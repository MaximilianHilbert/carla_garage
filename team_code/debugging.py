import sys
import os

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

config=GlobalConfig()
config.initialize(root_dir=config.root_dir)
print(config.train_data)
data=CARLA_Data(root=config.train_data,config=config)
print(data.images)