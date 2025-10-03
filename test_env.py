from env import ENV as env
from env import config
from col import COLLECTER
collecter = COLLECTER(config , env)
collecter.collect()
# collect data for foundation model training with saving into pickle files