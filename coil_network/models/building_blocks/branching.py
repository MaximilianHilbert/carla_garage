import torch.nn as nn


class Branching(nn.Module):
    def __init__(self, branched_modules=None):
        """

        Args:
            branch_config: A tuple containing number of branches and the output size.
        """

        super(Branching, self).__init__()

        """ ---------------------- BRANCHING MODULE --------------------- """
        if branched_modules is None:
            raise ValueError("No model provided after branching")

        self.branched_modules = nn.ModuleList(branched_modules)

    def forward(self, x):
        # get only the speeds from measurement labels

        branches_outputs = []
        for i, branch in enumerate(self.branched_modules):
            branches_outputs.append(branch(x))

        return branches_outputs
