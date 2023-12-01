import sys
import os
import timeit

# Set environment variables
os.environ['CARLA_ROOT'] = '/home/maximilian/Master/carla_garage/carla'
os.environ['WORK_DIR'] = '/home/maximilian/Master/carla_garage'

# Update sys.path
sys.path.append(os.path.join(os.environ['CARLA_ROOT'], 'PythonAPI'))
sys.path.append(os.path.join(os.environ['CARLA_ROOT'], 'PythonAPI', 'carla'))
sys.path.append(os.path.join(os.environ['CARLA_ROOT'], 'PythonAPI', 'carla', 'dist', 'carla-0.9.10-py3.7-linux-x86_64.egg'))

os.environ['SCENARIO_RUNNER_ROOT'] = os.path.join(os.environ['WORK_DIR'], 'scenario_runner')
os.environ['LEADERBOARD_ROOT'] = os.path.join(os.environ['WORK_DIR'], 'leaderboard')

# Update sys.path
sys.path.append(os.path.join(os.environ['CARLA_ROOT'], 'PythonAPI', 'carla'))
sys.path.append(os.environ['SCENARIO_RUNNER_ROOT'])
sys.path.append(os.environ['LEADERBOARD_ROOT'])
from data import CARLA_Data
from config import GlobalConfig
from diskcache import Cache
config=GlobalConfig()
config.initialize(root_dir=config.root_dir)
print(config.train_data)
use_disk_cache=True
if bool(use_disk_cache):
# NOTE: This is specific to our cluster setup where the data is stored on slow storage.
# During training, we cache the dataset on the fast storage of the local compute nodes.
# Adapt to your cluster setup as needed. Important initialize the parallel threads from torch run to the
# same folder (so they can share the cache).
    tmp_folder = str(os.environ.get('SCRATCH', '/tmp'))
    print('Tmp folder for dataset cache: ', tmp_folder)
    tmp_folder = tmp_folder + '/dataset_cache'
    shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024**3))
else:
    shared_dict = None
data=CARLA_Data(root=config.train_data,config=config, shared_dict=shared_dict)
callable_function=lambda: data.__getitem__(0)
timer=timeit.Timer(callable_function)
execution_time=timer.timeit(number=5)
print(execution_time)