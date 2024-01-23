"""
    A simple factory module that returns instances of possible modules 

"""

from coil_network.models.coil_policy import CoILPolicy
from coil_network.models.coil_memory_extraction import CoILMemExtract
from coil_network.models.coil_icra import CoILICRA

def CoILModel(architecture_name, architecture_configuration):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """

    print(architecture_name)

    if architecture_name == 'coil-policy':

        return CoILPolicy(architecture_configuration)

    elif architecture_name == 'coil-memory':

        return CoILMemExtract(architecture_configuration)
    elif architecture_name=="coil-icra":
        return CoILICRA(architecture_configuration)

    else:

        raise ValueError(" Not found architecture name")
