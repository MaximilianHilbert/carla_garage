import os
from functools import partial
import argparse
from datetime import datetime
from diskcache import Cache
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from team_code.data import CARLA_Data
from torch.nn.functional import mse_loss


class ActionModel(nn.Module):
    def __init__(self, input_dim, output_dim, neurons=None):
        super(ActionModel, self).__init__()
        self.units = neurons if neurons is not None else [300]

        layer_list = [
            nn.Linear(input_dim, self.units[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(len(self.units) - 1):
            layer_list.append(nn.Linear(self.units[i], self.units[i + 1]))
            layer_list.append(nn.LeakyReLU(0.2, inplace=True))
        layer_list.append(nn.Linear(self.units[-1], output_dim))

        self.main = nn.Sequential(*layer_list)
        self.max_loss, self.min_loss = 0, 0

    def forward(self, input):
        output = self.main(input)
        return output

    def load_margin(self, min_loss, max_loss):
        self.min_loss = min_loss
        self.max_loss = max_loss


def train(args, model, optimizer, train_loader, loss_func, epoch, all_epochs, logger):
    model.train()
    optimizer.zero_grad()
    accumulate_loss = []
    all_iterations = len(train_loader) - 1
    for idx, data in enumerate(tqdm(train_loader)):
        previous_wp = data["previous_ego_waypoints"].cuda().reshape(args.batch_size, -1)
        current_wp = data["ego_waypoints"].cuda().reshape(args.batch_size, -1)

        model.zero_grad()
        output = model(previous_wp)
        loss = loss_func(output, current_wp).mean()
        accumulate_loss.append(float(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    logger.add_scalar("train_loss", np.mean(accumulate_loss), epoch)
    print("epoch: {} train loss: {}".format(epoch, np.mean(accumulate_loss)))


def test(args, model, test_loader, loss_func, epoch, logger):
    model.eval()
    accumulate_loss = []
    min_loss, max_loss = np.inf, 0
    with torch.no_grad():
        for data in test_loader:
            previous_wp = data["previous_ego_waypoints"].cuda().reshape(args.batch_size, -1)
            current_wp = data["ego_waypoints"].cuda().reshape(args.batch_size, -1)

            output = model(previous_wp)
            loss = loss_func(output, current_wp)

            max_value = loss.max().item()
            min_value = loss.min().item()
            if min_value < min_loss:
                min_loss = min_value
            if max_value > max_loss:
                max_loss = max_value

            loss = loss.mean()
            accumulate_loss.append(float(loss.item()))
    logger.add_scalar("test_loss", np.mean(accumulate_loss), epoch)
    print("iter: {} test loss: {}".format(epoch, np.mean(accumulate_loss)))
    return min_loss, max_loss


def weighted_loss(output, target, weight=None):
    loss = (output - target).pow(2)
    if weight is not None:
        loss = loss * weight
        return loss.sum(dim=-1)
    else:
        return loss.mean(dim=-1)


def adjustlr(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay_rate


def train_ape_model(args, seed, repetition, merged_config_object, checkpoint_full_path, if_save):
    identifier = str(int(datetime.utcnow().timestamp())) + "".join([str(np.random.randint(10)) for _ in range(8)])
    if bool(args.use_disk_cache):
        # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
        # During training, we cache the dataset on the fast storage of the local compute nodes.
        # Adapt to your cluster setup as needed. Important initialize the parallel threads from torch run to the
        # same folder (so they can share the cache).
        tmp_folder = str(os.environ.get("SCRATCH", "/tmp"))
        print("Tmp folder for dataset cache: ", tmp_folder)
        tmp_folder = tmp_folder + "/dataset_cache"
        shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024**3))
    else:
        shared_dict = None
    dataset = CARLA_Data(
        root=merged_config_object.train_data,
        config=merged_config_object,
        shared_dict=shared_dict,
    )

    torch.manual_seed(seed)
    trainset, testset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    )

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        num_workers=args.number_of_workers,
        shuffle=True,
        drop_last=True,
    )
    testloader = DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        num_workers=args.number_of_workers,
        shuffle=False,
        drop_last=True,
    )

    model = ActionModel(
        input_dim=merged_config_object.number_previous_waypoints * merged_config_object.pred_len * 2,
        output_dim=merged_config_object.number_future_waypoints * merged_config_object.pred_len * 2,
        neurons=args.neurons,
    )
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    loss_func = mse_loss

    result_dir = os.path.join(
        os.environ.get("WORK_DIR"),
        "_logs",
        "keyframes",
        f"repetition_{str(repetition)}",
        "results",
        f"prev{merged_config_object.number_previous_waypoints}-repetition-{repetition}-{identifier}",
    )

    log_dir = os.path.join(result_dir, "run")
    os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    min_loss, max_loss = 0, 0
    for epoch in tqdm(range(1, merged_config_object.epochs_baselines + 1)):
        train(
            args,
            model,
            optimizer,
            trainloader,
            loss_func,
            epoch,
            merged_config_object.epochs,
            writer,
        )
        min_loss, max_loss = test(args, model, testloader, loss_func, epoch, writer)

        if epoch % (merged_config_object.epochs // 3) == 0:
            adjustlr(optimizer, 0.1)

    if if_save:
        os.makedirs(
            os.path.join(
                os.environ.get("WORK_DIR"),
                "_logs",
                "keyframes",
                f"repetition_{str(repetition)}",
                "checkpoints",
            ),
            exist_ok=True,
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "min_loss": min_loss,
            "max_loss": max_loss,
        }
        torch.save(checkpoint, checkpoint_full_path)
