from tools.video_generation import generate_video
from coil_utils.baseline_helpers import norm
import os
import pickle
from tqdm import tqdm
import numpy as np

def generate_cl_copycat_videos():
    main_root=os.path.join(os.environ.get("WORK_DIR"), "visualisation", "closed_loop")
    residuals={}
    tuning_parameter_1=0.6
    for root, dirs, files in tqdm(os.walk(main_root)):
        for file in files:
            if file.endswith("predictions.pkl"):
                file_path=os.path.join(root, file)
                with open(file_path, "rb") as f:
                    data=pickle.load(f)
                try:
                    residual=norm(data["prev"]-data["curr"], ord=2)
                except TypeError:
                    pass
                residuals.update({file_path: residual})
    residual_values=np.array(list(residuals.values()))
    mean=np.mean(residual_values)
    std=np.std(residual_values)
    data_dict={"residuals": residual_values, "mean": mean, "std": std}
    with open(os.path.join(main_root, "results.pkl"),"wb") as datafile:
        pickle.dump(data_dict, datafile)
    threshold=mean-std*tuning_parameter_1
    print(f"current threshold marked as copycat is: {threshold}")
    for index, (file, residual_value) in tqdm(enumerate(residuals.items())):
        if residual_value<threshold:
            rgb_folder=os.path.join(os.path.dirname(file), "with_rgb")
            generate_video(rgb_folder, save_path=os.path.join(main_root, "copycat_videos", f"{index}.avi"))
            
def main():
    generate_cl_copycat_videos()
if __name__=="__main__":
    main()