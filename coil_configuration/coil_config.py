from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from coil_utils.attribute_dict import AttributeDict
import copy
import numpy as np
import os
import yaml

from coil_configuration.namer import generate_name
from coil_logger.coil_logger import create_log, add_message



_g_conf = AttributeDict()
_g_conf.immutable(False)

"""#### GENERAL CONFIGURATION PARAMETERS ####"""
_g_conf.NUM_REPETITIONS=3
_g_conf.NUMBER_OF_LOADING_WORKERS = 12
_g_conf.FINISH_ON_VALIDATION_STALE = None

"""#### INPUT RELATED CONFIGURATION PARAMETERS ####"""
_g_conf.SENSORS = {'rgb': (3, 88, 200)}
_g_conf.MEASUREMENTS = {'float_data': (31)}
_g_conf.TARGETS = ['steer', 'throttle', "brake"]
#keyframes related
_g_conf.LIDAR_SEQ_LEN=1

_g_conf.EVERY_EPOCH=2
_g_conf.INPUTS = ['speed']
_g_conf.INTENTIONS = []
_g_conf.BALANCE_DATA = True
_g_conf.STEERING_DIVISION = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
_g_conf.PEDESTRIAN_PERCENTAGE = 0
_g_conf.SPEED_DIVISION = []
_g_conf.LABELS_DIVISION = [[0, 2, 5], [3], [4]]
_g_conf.SPLIT = None
_g_conf.REMOVE = None
_g_conf.AUGMENTATION = None
_g_conf.CARLA_FRAME_RATE=None

_g_conf.DATA_USED = 'all' #  central, all, sides,
_g_conf.USE_NOISE_DATA = True
_g_conf.TRAIN_DATASET_NAME = '1HoursW1-3-6-8'  # We only set the dataset in configuration for training
_g_conf.LOG_SCALAR_WRITING_FREQUENCY = 2
_g_conf.LOG_IMAGE_WRITING_FREQUENCY = 1000
_g_conf.EXPERIMENT_BATCH_NAME = "eccv"
_g_conf.EXPERIMENT_NAME = "default"
_g_conf.EXPERIMENT_GENERATED_NAME = None

_g_conf.PROCESS_NAME = "None"
_g_conf.EPOCHS = 30
#arp related
_g_conf.AUTO_LR=False
_g_conf.AUTO_LR_STEP=1000
_g_conf.ACTION_CORRELATION_MODEL_TYPE= 'prev9actions_weight'
_g_conf.ALL_FRAMES_INCLUDING_BLANK = None
_g_conf.IMAGE_SEQ_LEN = 1
_g_conf.PREFRAME_PROCESS = "None"  # None, blackhole, inpaint, blackhole+randombox
_g_conf.PREFRAME_PROCESS_NUM = 0
_g_conf.PREFRAME_PROCESS_PROBABILITY = 1.0  # The probability to mask out the objects
_g_conf.BLANK_FRAMES_TYPE = 'black'  # black (padd with all zeros) or copy (padd with the last image in frame sequence)
_g_conf.NUMBER_IMAGES_SEQUENCE = 1
_g_conf.SEQUENCE_STRIDE = 1
#keyframes related
_g_conf.NUMBER_FUTURE_ACTIONS=3#default 0
_g_conf.NUMBER_PREVIOUS_ACTIONS = 0
_g_conf.USE_COLOR_AUG=0
_g_conf.AUGMENT=0


_g_conf.VALIDATE_SCHEDULE = range(0, 2000, 200)
_g_conf.TEST_SCHEDULE = range(0, 2000, 200)
_g_conf.SPEED_FACTOR = 12.0

_g_conf.AUGMENT_LATERAL_STEERINGS = 6
_g_conf.NUMBER_OF_HOURS = 1
_g_conf.ASSIGN_WEATHER = None
#### Starting the model by loading another
_g_conf.PRELOAD_MODEL_BATCH = None
_g_conf.PRELOAD_MODEL_ALIAS = None
_g_conf.PRELOAD_MODEL_CHECKPOINT = None

"""#### Network Related Parameters ####"""

#all baselines related
_g_conf.CORRELATION_WEIGHTS=False
_g_conf.MODEL_TYPE = None
_g_conf.MODEL_CONFIGURATION = {}
#arp related
_g_conf.MEM_EXTRACT_MODEL_TYPE = None
_g_conf.MEM_EXTRACT_MODEL_CONFIGURATION = {}

_g_conf.PRE_TRAINED = False
_g_conf.WEIGHT_INIT_SEED = None  # the random seed dedicated to weight initialization
#keyframes related
_g_conf.TRAIN_WITH_ACTIONS_AS_INPUT=False
_g_conf.IMPORTANCE_SAMPLE_METHOD = 'mean'  # mean / softmax / threshold
_g_conf.SOFTMAX_TEMPER = 1.0
_g_conf.THRESHOLD_RATIO = 0.1  # set top 10% as THRESHOLD_WEIGHT and others as 1
_g_conf.THRESHOLD_WEIGHT = 5.0
_g_conf.SPEED_INPUT=True #also relevant for ARP!

_g_conf.OPTIMIZER = 'Adam'
_g_conf.LEARNING_RATE_DECAY_INTERVAL = 50000
_g_conf.LEARNING_RATE_DECAY_LEVEL = 0.5
_g_conf.LEARNING_RATE_THRESHOLD = 1000
_g_conf.LEARNING_RATE = 0.0002  # First
_g_conf.BRANCH_LOSS_WEIGHT = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95,0.05]
_g_conf.VARIABLE_WEIGHT = {'Steer': 0.5, 'Gas': 0.45, "Brake": 0.05}
_g_conf.USED_LAYERS_ATT = []

_g_conf.LOSS_FUNCTION = 'L1'

"""#### Simulation Related Parameters ####"""

_g_conf.USE_ORACLE = False
_g_conf.USE_FULL_ORACLE = False
_g_conf.AVOID_STOPPING = False
_g_conf.PLANNER_MIN_DISTANCE=10
_g_conf.PLANNER_MAX_DISTANCE=50

def merge_with_yaml(yaml_filename):
    """Load a yaml config file and merge it into the global config object"""
    global _g_conf
    with open(yaml_filename, 'r', encoding="utf-8") as f:

        yaml_file = yaml.safe_load(f)

        yaml_cfg = AttributeDict(yaml_file)


    _merge_a_into_b(yaml_cfg, _g_conf)

    if _g_conf.ALL_FRAMES_INCLUDING_BLANK is None:
        _g_conf.ALL_FRAMES_INCLUDING_BLANK = _g_conf.IMAGE_SEQ_LEN

    path_parts = os.path.split(yaml_filename)
    _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
    _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]
    _g_conf.EXPERIMENT_GENERATED_NAME = generate_name(_g_conf)


def get_names(folder):
    alias_in_folder = os.listdir(os.path.join('configs', folder))

    experiments_in_folder = {}
    for experiment_alias in alias_in_folder:

        g_conf.immutable(False)
        merge_with_yaml(os.path.join('configs', folder, experiment_alias))

        experiments_in_folder.update({experiment_alias: g_conf.experiment_generated_name})

    return experiments_in_folder


def set_type_of_process(process_type, shared_config_object,args=None, training_rep=None, param=None):
    """
    This function is used to set which is the type of the current process, test, train or val
    and also the details of each since there could be many vals and tests for a single
    experiment.

    NOTE: AFTER CALLING THIS FUNCTION, THE CONFIGURATION CLOSES

    Args:
        type:

    Returns:

    """

    if shared_config_object.process_name == "default":
        raise RuntimeError(" You should merge with some exp file before setting the type")

    if process_type == 'train':
        shared_config_object.process_name = process_type
    elif process_type == "validation":
        shared_config_object.process_name = process_type + '_' + param
    if process_type == "drive":  # FOR drive param is city name.
        shared_config_object.CITY_NAME = param.split('_')[-1]
        shared_config_object.process_name = process_type + '_' + param
    create_log(shared_config_object.baseline_folder_name,
            shared_config_object.baseline_name,
            training_rep,
            shared_config_object.process_name,
            shared_config_object.log_scalar_writing_frequency,
            shared_config_object.log_image_writing_frequency)
    if process_type == "train":
        if not os.path.exists(os.path.join(f'{os.environ.get("WORK_DIR")}/_logs', shared_config_object.experiment_batch_name,
                                            shared_config_object.experiment_name,f"repetition_{str(training_rep)}",
                                            'checkpoints') ):
                os.mkdir(os.path.join(f'{os.environ.get("WORK_DIR")}/_logs', shared_config_object.experiment_batch_name,
                                      shared_config_object.experiment_name,f"repetition_{str(training_rep)}",
                                      'checkpoints'))

    if process_type == "validation" or process_type == 'drive':
        if not os.path.exists(os.path.join(f'{os.environ.get("WORK_DIR")}/_logs', shared_config_object.experiment_batch_name,
                                           shared_config_object.experiment_name,
                                           shared_config_object.process_name + '_csv')):
            os.mkdir(os.path.join(f'{os.environ.get("WORK_DIR")}/_logs', shared_config_object.experiment_batch_name,
                                          shared_config_object.experiment_name,
                                           shared_config_object.process_name + '_csv'))



def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """

    assert isinstance(a, AttributeDict) or isinstance(a, dict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttributeDict) or isinstance(a, dict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if is it more than second stack
            if stack is not None:
                b[k] = v_
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)

        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts

        b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects


    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #

    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, type(None)):
        value_a = value_a
    elif isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_b, range) and not isinstance(value_a, list):
        value_a = eval(value_a)
    elif isinstance(value_b, range) and isinstance(value_a, list):
        value_a = list(value_a)
    elif isinstance(value_b, dict):
        value_a = eval(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a

g_conf = _g_conf

