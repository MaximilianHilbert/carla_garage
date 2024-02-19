"""
    A simple factory module that returns instances of possible modules 

"""

from coil_network.models.coil_policy import CoILPolicy
from coil_network.models.coil_memory_extraction import CoILMemExtract
from coil_network.models.coil_icra import CoILICRA

def CoILModel(model_type, config):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """

    print(model_type)

    if model_type == 'coil-policy':

        return CoILPolicy(config)

    elif model_type == 'coil-memory':

        return CoILMemExtract(config)
    elif model_type=="coil-icra":
        return CoILICRA(config)

    else:

        raise ValueError(" Not found architecture name")
