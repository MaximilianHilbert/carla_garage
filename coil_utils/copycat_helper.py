import numpy as np
import heapq
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import os
import pickle
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from team_code.data import CARLA_Data
import datetime
from team_code.timefuser_model import TimeFuser
from coil_utils.baseline_helpers import extract_and_normalize_data,find_free_port, get_latest_saved_checkpoint, align_previous_prediction, set_not_included_ablation_args
from torch.utils.data import Sampler
class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
def norm(differences, ord):
    if ord==1:
        return np.sum(np.absolute(differences))
    if ord==2:
        return np.sqrt(np.sum(np.absolute(differences)**2))
def determine_copycat(args,predictions_lst,current_waypoints_gt,previous_prediction_aligned,previous_gt_lst,keyframe_correlation,current_index,params):
    detection_keyframes,detection_ours=False,False
    pred_residual=norm(predictions_lst[current_index]["pred"]["wp_predictions"]-previous_prediction_aligned, ord=args.norm)
    gt_residual=norm(current_waypoints_gt-previous_gt_lst[current_index], ord=args.norm)
    
    condition_value_1=params["avg_of_avg_baseline_predictions"]-params["avg_of_std_baseline_predictions"]*args.pred_tuning_parameter
    condition_1=pred_residual<condition_value_1
    condition_value_keyframes=params["avg_of_kf"]+params["std_of_kf"]*args.pred_tuning_parameter
    if args.keyframes_threshold=="absolute":
        condition_keyframes=True if keyframe_correlation==5.0 else False
    else:
        condition_keyframes=keyframe_correlation>condition_value_keyframes
    if args.second_cc_condition=="loss":
        condition_value_2=params["loss_avg_of_avg"]+params["loss_avg_of_std"]*args.tuning_parameter_2
        condition_2=predictions_lst[current_index]["loss"]>condition_value_2
    else:
        condition_value_2=params["avg_gt"]+params["std_gt"]*args.tuning_parameter_2
        condition_2=gt_residual>condition_value_2
    if condition_1 and condition_2:
        detection_ours=True
    if condition_keyframes:
        detection_keyframes=True
    return detection_ours, detection_keyframes, {"pred_residual":pred_residual, "gt_residual": gt_residual, "condition_value_1": condition_value_1, "condition_value_2": condition_value_2, "condition_value_keyframes": condition_value_keyframes}


def get_copycat_criteria(data_lst,prev_predictions_aligned,prev_gt_aligned_lst, which_norm):
    differences_pred=[]
    differences_gt=[]
    differences_loss=[]
    for index,(current, previous_pred_aligned,prev_gt_aligned) in enumerate(zip(data_lst, prev_predictions_aligned,prev_gt_aligned_lst)):
        try:
            differences_pred.append(norm(previous_pred_aligned-current["pred"]["wp_predictions"], ord=which_norm))
            differences_gt.append(norm(prev_gt_aligned-current["gt"], ord=which_norm))
            differences_loss.append(norm(data_lst[index-1]["loss"]-current["loss"], ord=which_norm))
        except TypeError:
            differences_pred.append(np.nan)
            differences_gt.append(np.nan)
            differences_loss.append(np.nan)

    differences_pred=np.array(differences_pred)
    differences_gt=np.array(differences_gt)
    differences_loss=np.array(differences_loss)

    std_value_gt=np.nanstd(differences_gt)
    std_value_pred=np.nanstd(differences_pred)

    mean_value_gt=np.nanmean(differences_gt)
    mean_value_pred=np.nanmean(differences_pred)

    mean_value_loss=np.nanmean(differences_loss)
    std_value_loss=np.nanstd(differences_loss)

    return {"mean_pred": mean_value_pred,"mean_gt": mean_value_gt, "std_pred":std_value_pred,
            "std_gt":std_value_gt, "loss_mean": mean_value_loss, "loss_std": std_value_loss}

def get_action_predict_loss_threshold(correlation_weights, ratio):
    _action_predict_loss_threshold = {}
    if ratio in _action_predict_loss_threshold:
        return _action_predict_loss_threshold[ratio]
    else:
        action_predict_losses = correlation_weights
        threshold = heapq.nlargest(int(len(action_predict_losses) * ratio), action_predict_losses)[-1]
        _action_predict_loss_threshold[ratio] = threshold
        return threshold

def preprocess(args):
    baseline_dict={}
    keyframe_correlations=np.load(os.path.join(
            os.environ.get("WORK_DIR"),
            "keyframes",
            "importance_weights",
            f"bcoh_weights_copycat_prev9_rep0_neurons300.npy",
        ))
    
    if args.keyframes_threshold=="absolute":
        threshold_ratio=0.1
        importance_sampling_threshold_weight=5.0
        importance_sampling_threshold=get_action_predict_loss_threshold(keyframe_correlations,threshold_ratio)
        keyframe_correlations=(keyframe_correlations > importance_sampling_threshold)* (importance_sampling_threshold_weight - 1) + 1        
    path_of_baselines=os.path.join(os.path.join(os.environ.get("WORK_DIR"),
                    "_logs",
        ))
    for root, dirs, files in os.walk(path_of_baselines):
        for file in files:
            if file=="predictions_std_all.pkl":
                with open(os.path.join(root,"predictions_std_all.pkl"), 'rb') as f:
                    criterion_dict = pickle.load(f)
                baseline_dict[root]={}
                baseline_dict[root].update({"std_pred": criterion_dict["std_pred"], "mean_pred": criterion_dict["mean_pred"],
                                                "loss_std": criterion_dict["loss_std"], "loss_mean":criterion_dict["loss_mean"]})
            
    ret_dict= {"avg_of_std_baseline_predictions": np.mean(np.array([per_baseline_stats["std_pred"] for per_baseline_stats in baseline_dict.values()])),
            "avg_of_avg_baseline_predictions":np.mean(np.array([per_baseline_stats["mean_pred"] for per_baseline_stats in baseline_dict.values()])),
            "std_gt":criterion_dict["std_gt"], "avg_gt":criterion_dict["mean_gt"], "loss_avg_of_std":np.mean(np.array([per_baseline_stats["loss_std"] for per_baseline_stats in baseline_dict.values()])),
            "loss_avg_of_avg":np.mean(np.array([per_baseline_stats["loss_mean"] for per_baseline_stats in baseline_dict.values()])),
            "avg_of_kf":np.mean(keyframe_correlations),"std_of_kf":np.std(keyframe_correlations), "keyframes_correlations": keyframe_correlations}
    return ret_dict


def evaluate_baselines_and_save_predictions(args, baseline_path):
    torch.cuda.empty_cache()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    print(f"World-size {world_size}, Rank {rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"env://127.0.0.1:{find_free_port()}",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=259200),
    )
    if rank == 0:
        print("Backend initialized")
    for root, dirs, files in os.walk(baseline_path):
        for file in files:
            if file=="config_training.pkl":
                basepath=root
                with open(os.path.join(basepath, "config_training.pkl"),'rb') as file:
                    config=pickle.load(file)
                set_not_included_ablation_args(config)
                config.number_previous_waypoints=1
                config.visualize_copycat=True
                config.initialize(root_dir=os.environ.get("DATASET_ROOT"),
                                   setting=args.setting)
                checkpoint_file = get_latest_saved_checkpoint(
                basepath
            )
                checkpoint = torch.load(
                    os.path.join(basepath,
                        "checkpoints",
                        f"{checkpoint_file}.pth",
                    ),
                    map_location=lambda storage, loc: storage,
                )
                print(f"loaded checkpoint_{checkpoint_file}")
                if "arp" in config.baseline_folder_name:
                    policy = TimeFuser("arp-policy", config)
                    policy.to("cuda:0")
                    policy = DDP(policy, device_ids=["cuda:0"])

                    mem_extract = TimeFuser("arp-memory", config)
                    mem_extract.to("cuda:0")
                    mem_extract = DDP(mem_extract, device_ids=["cuda:0"])
                    policy.load_state_dict(checkpoint["policy_state_dict"])
                    mem_extract.load_state_dict(checkpoint["mem_extract_state_dict"])
                   
                else:
                    model = TimeFuser(config.baseline_folder_name, config)
                    model.to("cuda:0")
                    model = DDP(model, device_ids=["cuda:0"])
                    model.load_state_dict(checkpoint["state_dict"])
                if Path(os.path.join(basepath,f"predictions_all.pkl")).exists():
                    print("already ran experiment")
                    continue
                val_set = CARLA_Data(root=config.val_data, config=config, shared_dict=None, rank=0,baseline=config.baseline_folder_name)

                if "keyframes" in config.baseline_folder_name:
                    filename = os.path.join(
                        os.environ.get("WORK_DIR"),
                            "_logs",
                            "keyframes",
                            "repetition_0",
                            "bcoh_weights_copycat_prev9_rep0_neurons300.npy")
                    val_set.set_correlation_weights(path=filename)
                    action_predict_threshold = get_action_predict_loss_threshold(
                        val_set.get_correlation_weights(), config.threshold_ratio
                    )
                sampler_val=SequentialSampler(val_set)
                data_loader_val = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=1,
                    num_workers=args.number_of_workers,
                    pin_memory=True,
                    shuffle=False,  # because of DDP
                    drop_last=True,
                    sampler=sampler_val,
                )
                
                print("Start of Evaluation")
                cc_lst=[]
                assert len(data_loader_val.dataset.images)!=0, "Selected validation dataset does not contain Town02 (validation) routes!"
                for data,image in zip(tqdm(data_loader_val), data_loader_val.dataset.images):
                    image=str(image, encoding="utf-8").replace("\x00", "")
                    all_images, all_speeds, target_point, targets, previous_targets, additional_previous_waypoints, bev_semantic_labels, targets_bb,bb=extract_and_normalize_data(args=args, device_id="cuda:0", merged_config_object=config, data=data)
                    
                    if "arp" in config.baseline_folder_name:
                        with torch.no_grad():
                            pred_dict_memory = mem_extract(x=all_images[:,:-1,...], speed=all_speeds[:,:-1,...] if all_speeds is not None else None, target_point=target_point, prev_wp=additional_previous_waypoints)

                        mem_extract_targets = targets - previous_targets
                        loss_function_params_memory = {
                            **pred_dict_memory,
                            "targets": mem_extract_targets,
                            "bev_targets": bev_semantic_labels,
                            "targets_bb": targets_bb,
                            "device_id": "cuda:0",
                        }

                        mem_extract_loss, detailed_losses,head_losses= mem_extract.module.compute_loss(params=loss_function_params_memory)
                        pred_dict = policy(x=all_images[:,-1:,...], speed=all_speeds[:,-1:,...] if all_speeds is not None else None, target_point=target_point, prev_wp=None, memory_to_fuse=pred_dict_memory["memory"].detach())
                            
                        loss_function_params_policy = {
                            **pred_dict,
                            "targets": targets,
                            "bev_targets": bev_semantic_labels,
                            "targets_bb": targets_bb,
                            "device_id": "cuda:0",
                            
                        }
                        policy_loss,detailed_losses,head_losses= policy.module.compute_loss(params=loss_function_params_policy)
                    else:
                        if "bcso" in config.baseline_folder_name or "bcoh" in config.baseline_folder_name or "keyframes" in config.baseline_folder_name:
                            with torch.no_grad():
                                pred_dict= model(x=all_images, speed=all_speeds, target_point=target_point, prev_wp=additional_previous_waypoints)

                        if "keyframes" in config.baseline_folder_name:
                            reweight_params = {
                                "importance_sampling_softmax_temper": config.softmax_temper,
                                "importance_sampling_threshold": action_predict_threshold,
                                "importance_sampling_method": config.importance_sample_method,
                                "importance_sampling_threshold_weight": config.threshold_weight,
                                "action_predict_loss": data["correlation_weight"].squeeze().to("cuda:0"),
                            }
                        else:
                            reweight_params = {}
                        
                        loss_function_params = {
                            **pred_dict,
                            "bev_targets": bev_semantic_labels if config.bev else None,
                            "targets": targets,
                            "targets_bb": targets_bb,
                            **reweight_params,
                            "device_id": "cuda:0",
                            
                        }
                        model_loss,detailed_losses,head_losses= model.module.compute_loss(params=loss_function_params)
                        
                    #this is only a viable comparison, if the batch_size is set to 1, because it will be marginalized over the batch dimension before the loss is returned!
                    cc_lst.append({"image": image, "pred":{key: value.squeeze().cpu().detach().numpy() for key, value in pred_dict.items() if not isinstance(value,tuple)},
                                   "pred_bb":{key: value for key, value in pred_dict.items() if isinstance(value,tuple)},
                                                "gt":targets.squeeze().cpu().detach().numpy(), "loss":detailed_losses["wp_loss"].cpu().detach().numpy(),
                                                "detailed_loss": {key: value.cpu().detach().numpy() for key, value in detailed_losses.items()},
                                                "head_loss": {key: value.cpu().detach().numpy() for key, value in head_losses.items()} if head_losses is not None else None,
                                                "previous_matrix": data["ego_matrix_previous"].detach().cpu().numpy()[0],
                                                "current_matrix": data["ego_matrix_current"].detach().cpu().numpy()[0]})
                prev_predictions_aligned_lst=[]
                prev_gt_aligned_lst=[]
                for i in range(len(cc_lst)):
                    if i==0:
                        prev_predictions_aligned_lst.append(np.nan)
                        prev_gt_aligned_lst.append(np.nan)
                    else:
                        prev_predictions_aligned=align_previous_prediction(pred=cc_lst[i-1]["pred"]["wp_predictions"], matrix_previous=cc_lst[i]["previous_matrix"],
                                                                        matrix_current=cc_lst[i]["current_matrix"])
                        prev_gt_aligned=align_previous_prediction(pred=cc_lst[i-1]["gt"], matrix_previous=cc_lst[i]["previous_matrix"],
                                                                        matrix_current=cc_lst[i]["current_matrix"])
                        prev_predictions_aligned_lst.append(prev_predictions_aligned)
                        prev_gt_aligned_lst.append(prev_gt_aligned)
                #data_df = pd.DataFrame.from_dict(wp_dict, orient='index', columns=['image','pred', 'gt', 'loss'])
                criterion_dict=get_copycat_criteria(cc_lst, prev_predictions_aligned_lst,prev_gt_aligned_lst,args.norm)
                
            
                with open(os.path.join(basepath,f"predictions_all.pkl"), "wb") as file:
                        pickle.dump(cc_lst, file)
            
                with open(os.path.join(basepath,f"aligned_predictions_all.pkl"), "wb") as file:
                        pickle.dump(prev_predictions_aligned_lst, file)
                with open(os.path.join(basepath,f"aligned_gt_all.pkl"), "wb") as file:
                        pickle.dump(prev_gt_aligned_lst, file)
                

                with open(os.path.join(basepath,"predictions_std_all.pkl"), "wb") as file:
                    pickle.dump(criterion_dict, file)
                with open(os.path.join(basepath,"config_cc.pkl"), "wb") as file:
                    pickle.dump(config, file)