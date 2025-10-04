"""
REGISTER
"""
from habitat import Config
from ss_baselines.common.environments import AudioNavRLEnv
from ENMuS.sen import SoundEventNavSim
from utils import draw_map
import pickle , copy , os ,random
class REGISTER:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f"{name} has already in registry")
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name):
        
        if name not in self._registry:
            raise KeyError(f"{name} not in registry : {list(self._registry.keys())}")
        return self._registry[name]

    def create(self, name, *args, **kwargs):
       
        cls = self.get(name)
        return cls(*args, **kwargs)

    def available(self):
        return list(self._registry.keys())
    
CollectRegister = REGISTER()

@CollectRegister.register("greedy")
class GreedyCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config):
        self.env = env
        self.sim: SoundEventNavSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            self.save_data = copy.deepcopy(self.save_data_struct)
            path_point = []
            obs = self.env.reset()
            action_id = self.sim.compute_oracle_actions()
            done = False
            self.save(sound_id = self.env._env.current_episode.sound_id)
            self.save(obs=obs,action_id=action_id,done=done)
            path_point.append(self.sim.get_agent_state().position)
            for action in action_id:
                obs , reward , done , info = self.env.step(action=action)
                self.save(obs=obs,reward=reward,done=done,info=info)
                path_point.append(self.sim.get_agent_state().position)
                if action == action_id[-1]:
                    print(f"action is {action} , action_list is {action_id} , done is {done} , info is {info}")
            
            self.save(map=draw_map(self.env , path_point))
            self.save(path_point=path_point)
            self.store(self.env._env.current_episode.scene_id , self.env._env.current_episode)
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, scene , id):
        os.makedirs(f"{self.save_data_dir}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None
        

@CollectRegister.register("collided")
class CollidedCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config):
        self.env = env
        self.sim: SoundEventNavSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            self.save_data = copy.deepcopy(self.save_data_struct)
            greedy_path_point = []
            collided_path_point = []
            obs = self.env.reset()
            
            done = False
            self.save(obs=obs,done=done)
            self.save(sound_id = self.env._env.current_episode.sound_id)
            greedy_path_point.append(self.sim.get_agent_state().position)
            while len(self.save_data['done']) < 50 and not done:
                action = random.choice([1,2,3])
                _ , _ , done , _ = self.env.step(action = action)
                greedy_path_point.append(self.sim.get_agent_state().position)
                collided = self.sim.previous_step_collided
                if collided:
                    greedy_action = self.sim.compute_oracle_actions()
                    for action in greedy_action[:5]:
                        obs , reward , done , info = self.env.step(action = action)
                        collided_path_point.append(self.sim.get_agent_state().position)
                        self.save(obs=obs,reward=reward,done=done,info=info,action_id=action)
                        if done:
                            break
            
            self.save(greedy_map=draw_map(self.env , greedy_path_point))
            self.save(collided_map=draw_map(self.env , collided_path_point))
            self.save(greedy_path_point=greedy_path_point)
            self.save(collided_path_point=collided_path_point)
            self.store(self.env._env.current_episode.scene_id , self.env._env.current_episode)
            
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, scene , id):
        os.makedirs(f"{self.save_data_dir}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None

@CollectRegister.register("random")
class RandomCollect:
    def __init__(self , env:AudioNavRLEnv , config:Config):
        self.env = env
        self.sim: SoundEventNavSim = env._env._sim
        self.save_data_struct = config.DATA_STRUCT
        self.save_data_dir = config.DATA_DIR
        self.save_path_img = config.IMG_DIR
        self.save_path_type = config.IMG_TYPE
    def collect(self):
        for _ in range(self.env._env.number_of_episodes):
            self.save_data = copy.deepcopy(self.save_data_struct)
            greedy_path_point = []
            collided_path_point = []
            obs = self.env.reset()
            
            done = False
            self.save(obs=obs,done=done)
            self.save(sound_id = self.env._env.current_episode.sound_id)
            greedy_path_point.append(self.sim.get_agent_state().position)
            while len(self.save_data['done']) < 50 and not done:
                action = random.choice([1,2,3])
                obs , reward , done , info = self.env.step(action = action)
                greedy_path_point.append(self.sim.get_agent_state().position)
                self.save(obs=obs,reward=reward,done=done,info=info,action_id=action)
                if done:
                    break
            
            self.save(greedy_map=draw_map(self.env , greedy_path_point))
            self.save(collided_map=draw_map(self.env , collided_path_point))
            self.save(greedy_path_point=greedy_path_point)
            self.save(collided_path_point=collided_path_point)
            self.store(self.env._env.current_episode.scene_id , self.env._env.current_episode)
            
    def save(self , **kwargs):
        for key , value in kwargs.items():
            self.save_data[key].append(value)
    def store(self, scene , id):
        os.makedirs(f"{self.save_data_dir}/{scene[-15:-4]}",exist_ok=True)
        
        with open(f"{self.save_data_dir}/{scene[-15:-4]}/{id.episode_id}.pkl" , 'wb' ) as f:
            pickle.dump(self.save_data , f)
        self.save_data = None
