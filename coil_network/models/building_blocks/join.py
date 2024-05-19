import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


class Join(nn.Module):
    def __init__(self, params=None, module_name="Default"):
        super(Join, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if "mode" not in params:
            raise ValueError(" Missing the mode parameter ")
        if "after_process" not in params:
            raise ValueError(" Missing the after_process parameter ")

        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization

        self.after_process = params["after_process"]
        self.mode = params["mode"]

    def forward(self, x, m=None):
        # get only the speeds from measurement labels

        if self.mode == "cat" and m is not None:
            j = torch.cat((x, m), 1)
        elif m is None:
            j=x
        else:
            raise ValueError("Mode to join networks not found")

        return self.after_process(j)
