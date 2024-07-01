import os

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, baseline_name, experiment, repetition, setting):
        self.dir_name = os.path.join(
            os.environ.get("WORK_DIR"),
            "_logs",
            baseline_name,
            experiment,
            f"repetition_{str(repetition)}",
            setting,
        )
        self.full_name = os.path.join(self.dir_name, "tensorboard")

    def add_scalar(self, name, scalar, step):
        self.writer.add_scalar(name, scalar, step)
    def flush(self):
        self.writer.flush()
    def close(self):
        self.writer.close()

    def create_tensorboard_logs(self):
        self.writer = SummaryWriter(log_dir=self.full_name)

    def create_checkpoint_logs(
        self,
    ):
        os.makedirs(os.path.join(self.dir_name, "checkpoints"), exist_ok=True)
