from ENMuS import SoundEventNavSim , SoundEventNavDataset
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class , AudioNavRLEnv
from configs.default import get_config
# ENV_CONFIG_PATH = './configs/audiogoal.yaml'
ENV_NAME = "AudioNavRLEnv"
config = get_config()

"""
从soundspaces中获取到env并暴漏出来
"""
ENV = construct_envs(config=config , env_class=get_env_class(ENV_NAME))