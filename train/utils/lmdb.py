import lmdb , os , pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from train import Network
from train.utils.avtrans import AVtrans
class Transfor:
    def __init__(self , paths:list[str]):
        self.paths = paths
    
    def to_lmdb(self , **kwargs):
        """
        首先实现一个简单版本的。
        """
        if kwargs['type'] == "foundation_model_lmdb_save":
            files = self.get_files_by_scenes()
            step = 0
            env = lmdb.open(f"./dataset/lmdb/greedy", map_size=1024*1014*1024*70)
            txn = env.begin(write=True) 
            for file in tqdm(files):
                with open(file , 'rb') as f:
                    data = pickle.load(f)
                obs = data['obs']
                reward = data['reward']
                done = data['done']
                action_id = data['action_id']
                # step_angle =  self.get_angle_from_path_point(path_point)
                # import pdb;pdb.set_trace()
                lmdb_data = list(zip(obs[:-1] , action_id[0]))
                # import pdb;pdb.set_trace()

                for index , data in enumerate(lmdb_data):
                    key = f'index_{step}'.encode('utf-8')
                    data = pickle.dumps(data)
                    txn.put(key, data)
                    step+=1
                    if index % 100 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                key = f'__len__'.encode('utf-8')
                len_ = pickle.dumps(step)
                txn.put(key , len_)
                txn.commit()
                env.close()
        if kwargs['type'] == "foundation_model_torch_save":
            files = self.get_files_by_scenes(kwargs['b_s'] , kwargs['e_s'] , kwargs['b_f'] , kwargs['e_f'])
            shard_size = 10000  # 每个 shard 1w 样本
            shard_id = 0

            buffer_visuals, buffer_audios, buffer_actions , buffer_angles = [], [], [] , []

            for file in tqdm(files):
                with open(file, 'rb') as f:
                    data = pickle.load(f)

                obs = data['obs']
                
                # if "path_point" in data:
                #     path_point = data['path_point']
                # elif "greedy_path_point" in data:
                #     path_point = data['greedy_path_point']
                # else:
                #     print(f"file name is {file}")
                #     continue

                # if np.array(path_point).ndim == 3 and np.array(path_point).shape[0] == 1:
                #     path_point = path_point[0]

                action_id = data['action_id']
                action_id = np.array(action_id).reshape(-1).tolist()
                for v, a in zip(obs[:-1], action_id):
                    visual = torch.from_numpy(v['rgb']).float() / 255.0
                    audio = torch.from_numpy(v['spectrogram'][0]).float()
                    # audio = AVtrans.mel_audio(v['spectrogram'][1])
                    action = torch.tensor(a, dtype=torch.long)
                    angel = np.degrees(v['angle'][1])
                    buffer_visuals.append(visual)
                    buffer_audios.append(audio)
                    buffer_actions.append(action)
                    buffer_angles.append(torch.tensor(angel))
                    # 写一个 shard
                    if len(buffer_visuals) >= shard_size:
                        torch.save({
                            'visuals': torch.stack(buffer_visuals),
                            'audios': torch.stack(buffer_audios),
                            'actions': torch.stack(buffer_actions),
                            'angles':torch.stack(buffer_angles)
                        }, f"{kwargs['save_dir']}/foundation_model_shard_{shard_id}.pt")

                        print(f"保存 shard {shard_id}, size={len(buffer_visuals)}")

                        buffer_visuals, buffer_audios, buffer_actions = [], [], []
                        shard_id += 1

            # 保存最后一个不满 shard 的数据
            if buffer_visuals:
                torch.save({
                    'visuals': torch.stack(buffer_visuals),
                    'audios': torch.stack(buffer_audios),
                    'actions': torch.stack(buffer_actions),
                    'angles':torch.stack(buffer_angles)
                }, f"{kwargs['save_dir']}/foundation_model_shard_{shard_id}.pt")
                print(f"保存 shard {shard_id}, size={len(buffer_visuals)}")

        if kwargs['type'] == "offline_rl_embedding_save":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            foundation_model = Network().to(device)
            foundation_model.load_state_dict(torch.load('experiment/train/ckpt/model_epoch_600.pth'))
            foundation_model.eval()
            shard_size = 10000
            save_dir = kwargs['save_dir']
            os.makedirs(save_dir, exist_ok=True)

            shard_id = 0
            buffer_states, buffer_next_states, buffer_actions, buffer_rewards, buffer_dones = [], [], [], [], []

            files = self.get_files_by_scenes()

            for file in tqdm(files, desc="trans pickle to embedding"):
                with open(file, 'rb') as f:
                    episode_data = pickle.load(f)

                obs = episode_data['obs']
                reward = episode_data['reward']
                done = episode_data['done']
                action_id = np.array(episode_data['action_id']).reshape(-1).tolist()

                # 生成 transitions (s, a, r, s', done)
                for s, a, r, s_next, d in zip(obs[:-1], action_id, reward[:-1], obs[1:], done[:-1]):
                    # 用 foundation_model 提取 embedding
                    visual = torch.from_numpy(s['rgb']).float() / 255.0
                    audio = torch.from_numpy(s['spectrogram']).float()
                    next_visual = torch.from_numpy(s_next['rgb']).float() / 255.0
                    next_audio = torch.from_numpy(s_next['spectrogram']).float()
                    with torch.no_grad():
                        state_emb = foundation_model.embedding_forward(audio.unsqueeze(0).to(device), visual.unsqueeze(0).to(device)).squeeze(0).cpu()
                        next_state_emb = foundation_model.embedding_forward(next_audio.unsqueeze(0).to(device), next_visual.unsqueeze(0).to(device)).squeeze(0).cpu()

                    action = torch.tensor(a, dtype=torch.long)
                    reward_t = torch.tensor(r, dtype=torch.float32)
                    done_t = torch.tensor(d, dtype=torch.bool)

                    buffer_states.append(state_emb)
                    buffer_next_states.append(next_state_emb)
                    buffer_actions.append(action)
                    buffer_rewards.append(reward_t)
                    buffer_dones.append(done_t)

                    # shard 写盘
                    if len(buffer_states) >= shard_size:
                        torch.save({
                            'states': torch.stack(buffer_states),
                            'next_states': torch.stack(buffer_next_states),
                            'actions': torch.stack(buffer_actions),
                            'rewards': torch.stack(buffer_rewards),
                            'dones': torch.stack(buffer_dones),
                        }, os.path.join(save_dir, f"offline_model_shard_{shard_id}.pt"))

                        print(f"保存 shard {shard_id}, size={len(buffer_states)}")

                        buffer_states, buffer_next_states, buffer_actions, buffer_rewards, buffer_dones = [], [], [], [], []
                        shard_id += 1

            # 保存最后不足 shard_size 的数据
            if buffer_states:
                torch.save({
                    'states': torch.stack(buffer_states),
                    'next_states': torch.stack(buffer_next_states),
                    'actions': torch.stack(buffer_actions),
                    'rewards': torch.stack(buffer_rewards),
                    'dones': torch.stack(buffer_dones),
                }, os.path.join(save_dir, f"offline_model_shard_{shard_id}.pt"))
                print(f"保存 shard {shard_id}, size={len(buffer_states)})")


    def get_files_by_scenes(self , b_s , e_s , b_f , e_f):
        scenes = []
        for path in self.paths:
            scenes.extend([os.path.join(path , i ) for i in os.listdir(path)[b_s : e_s]])
        with open('split.txt' , 'w') as f:
            f.write(f"train/val split is : {scenes}")
        files = []
        for scene in scenes:
            files.extend([os.path.join(scene , i ) for i in os.listdir(scene)[b_f:e_f]])
        return files
    
    def get_angles_from_path(path_points):
        """
        path_points: list of [y, z, x]
        return: list of angles (degree) from each point to the last point
        """
        last_y, _, last_x = path_points[-1]
        angles = []
        for y, _, x in path_points[:-1]:  # 除了最后一个点
            dy = last_y - y
            dx = last_x - x
            angle_rad = math.atan2(dy, dx)  # [-pi, pi]
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            angles.append(angle_deg)
        return angles