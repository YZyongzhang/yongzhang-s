# dataloader
from torch.utils.data import DataLoader , Dataset
import os
import lmdb , pickle
from tqdm import tqdm
import sys
import pdb
import librosa
import torch
import numpy as np
from PIL import Image
import random
from train.network.foundation_model import Network
import time
import glob

class LoadLmdb:

    def __init__(self, path):

        self.paths = self._deal_path(path)


    @classmethod
    def load_lmdb(cls,paths):
        
        # 这里看需不需要使用这个dealpath ，也可以在外部处理，如果文件层级简单的话推荐使用内部的
        # paths = cls._deal_path(paths)
        dataset_name = input("please input the dataset name:")
        env = lmdb.open(f"./experiment/dataset/{dataset_name}", map_size=1024*1014*1024*30)
        sum_index = 0
        txn = env.begin(write=True) 
        for path_i in tqdm(paths):
            with open(path_i ,'rb') as f:
                
                data = pickle.load(f)

                audio , visual , action  = cls.get_values_tuple(data)
                for index in range(len(audio)):
                    key = f'index_{sum_index}'.encode('utf-8')
                    values = (audio[index],visual[index] , action[index])
                    values = pickle.dumps(values)
                    txn.put(key, values)
                    sum_index+=1

                    if sum_index % 5000 == 0: 
                        txn.commit()
                        txn = env.begin(write=True)
        
        # 在内部设置了长度大小
        key = f'__len__'.encode('utf-8')
        len_ = pickle.dumps(sum_index)
        txn.put(key , len_)
        txn.commit()
        env.close()
        print(f"dataset location the ./experiment/dataset/{dataset_name}")

    @classmethod
    def load_lmdb_offline_rl(cls, paths):
        """
        离线RL的LMDB数据加载
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset_name = input("please input the dataset name:")
        env = lmdb.open(f"./experiment/dataset/{dataset_name}", map_size=1024*1024*1024*30)
        foundation_model = Network().to(device)
        checkpoint = torch.load('./experiment/train/ckpt/model_epoch_1000.pth')
        foundation_model.load_state_dict(checkpoint)
        foundation_model.eval()
        sum_index = 0
        txn = env.begin(write=True) 
        for path_i in tqdm(paths):
            with open(path_i ,'rb') as f:
                data = pickle.load(f)

                audio , visual, next_audio, next_visual, action, rewards, dones = cls.get_values_offline(data)
                dones = [int(i[0]) for i in dones]

                for index in range(len(audio)):
                    key = f'index_{sum_index}'.encode('utf-8')
                    audio_t , visual_t = AVtrans.deal_data(audio[index], visual[index])
                    next_audio_t, next_visual_t = AVtrans.deal_data(next_audio[index], next_visual[index])
                    state = foundation_model(audio_t.unsqueeze(0).to(device), visual_t.unsqueeze(0).to(device))
                    next_state = foundation_model(next_audio_t.unsqueeze(0).to(device), next_visual_t.unsqueeze(0).to(device))
                    value = (state.squeeze(0).detach().cpu(), next_state.squeeze(0).detach().cpu(), action[index], rewards[index], dones[index])
                    values = pickle.dumps(value)
                    txn.put(key, values)
                    sum_index+=1

                    if sum_index % 5000 == 0: 
                        txn.commit()
                        txn = env.begin(write=True)
        
        # 在内部设置了长度大小
        key = f'__len__'.encode('utf-8')
        len_ = pickle.dumps(sum_index)
        txn.put(key , len_)
        txn.commit()
        env.close()
        print(f"dataset location the ./experiment/dataset/{dataset_name}")

    @classmethod
    def get_values_tuple(cls, data):
        audio = data[0][0]['audio'][:-1]
        visual = data[0][0]['camera'][:-1]
        action = data[0][0]['rl_pred']
        return audio ,visual , action

    @classmethod
    def get_values_offline(cls, data):
        """
        处理数据的函数
        """
        audio = data[0][0]['audio'][:-1]
        visual = data[0][0]['camera'][:-1]
        next_audio = data[0][0]['audio'][1:]
        next_visual = data[0][0]['camera'][1:]
        action = data[0][0]['rl_pred']
        rewards = data[0][0]['reward']
        dones = data[1]
        return audio , visual, next_audio, next_visual, action, rewards, dones
    @classmethod
    def _deal_path(cls,path):

        # 简单情况的deal，复杂的话估计要重复好几次，不如重新写一个def
        files = os.listdir(path)
        paths = [os.path.join(path , i) for i in files]
        return paths

class AVtrans:
    @classmethod
    def deal_data(cls, audio, visual):
        """
        处理音频和视觉数据
        """
        audio = audio.astype(float) / 32768.0
        audio_tensor = cls.mel_audio(audio)
        visual = visual[:,:,:-1]
        image_tensor = torch.tensor(np.array(visual), dtype=torch.float32) # 确保视觉数据是float32类型
        return  audio_tensor , image_tensor
    @classmethod
    def mel_audio(cls, audio):
        """
        处理音频数据，转换为梅尔频谱图
        """
        left_audio = audio[0]
        right_audio = audio[1]
        
        sr = 48000
        mel_spec_left = librosa.feature.melspectrogram(y=left_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spec_left = librosa.power_to_db(mel_spec_left, ref=np.max)

        mel_spec_right = librosa.feature.melspectrogram(y=right_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spec_right = librosa.power_to_db(mel_spec_right, ref=np.max)

        mel_spec = np.stack([mel_spec_left, mel_spec_right], axis=0)
        audio_tensor = torch.tensor(mel_spec, dtype=torch.float32)

        return audio_tensor

class VADE_Offline(Dataset):
    def __init__(self, path , transform = None):
        super().__init__()

        self.env = lmdb.open(
            path,
            readonly=True,
        )
        self.txn = self.env.begin()
        

        self.transform = transform

    def __getitem__(self, index):
        key = f"index_{index}".encode("utf-8")
        value = self.txn.get(key)

        if value is None:
            raise IndexError(f"Index {index} not found in LMDB dataset")

        sample = pickle.loads(value)
        state, next_state, action, rewards, done= sample
        
        return state, next_state, action, rewards, done

    def __len__(self):
        length_key = b'__len__'
        length = self.txn.get(length_key)
        r = pickle.loads(length)
        return r
    
class VADE(Dataset):
    def __init__(self, lmdb_env , txn):
        super().__init__()
        self.env = lmdb_env
        self.txn = txn


    def __getitem__(self, index):
        key = f"index_{index}".encode("utf-8")
        # b_time = time.time()
        value = self.txn.get(key)
        # e_time = time.time()
        # print(f"txn load index time {e_time - b_time}")

        if value is None:
            raise IndexError(f"Index {index} not found in LMDB dataset")
        
        # p_b_time = time.time()
        sample = pickle.loads(value)
        # p_e_time = time.time()
        # print(f"pickle load time {p_e_time - p_b_time}")
        # obs  , action = sample
        # visual = obs['rgb'] 
        # audio =  obs['spectrogram']
        # # d_b_time = time.time()
        # visual = torch.from_numpy(visual).float() / 255.0
        # audio = torch.from_numpy(audio).float()
        obs_batch, action_batch = zip(*sample)   # 相当于解压 list
        # obs_batch: (obs1, obs2, ...)
        # action_batch: (action1, action2, ...)
        visuals = torch.stack([torch.from_numpy(obs['rgb']).float() / 255.0 for obs in obs_batch])
        audios  = torch.stack([torch.from_numpy(obs['spectrogram']).float() for obs in obs_batch])
        # actions = torch.tensor(action_batch)  # 如果 action 是标量/小数组，可以直接转 tensor
        # d_e_time = time.time()
        # print(f"from array to torch{d_e_time - d_b_time}")
        return visuals ,audios  ,action_batch
    def __len__(self):
        length_key = b'__len__'
        length = self.txn.get(length_key)
        r = pickle.loads(length)
        return r

class ShardedPTDataset(Dataset):
    def __init__(self, shard_pattern="./dataset/pt/foundation_model_shard_*.pt", preload=True):
        """
        shard_pattern: shard 文件路径模式，比如 ./dataset/foundation_model_shard_*.pt
        preload: 是否把所有 shard 一次性加载到内存（大数据集建议 False）
        """
        super().__init__()
        self.shard_files = sorted(glob.glob(shard_pattern))
        assert len(self.shard_files) > 0, f"No shards found at {shard_pattern}"

        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_map = []    # 全局 index → (shard_id, local_index)

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.shard_files):
            print(f"scan shard id {shard_id}")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        self.total_size = sum(self.shard_sizes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        shard_id, local_idx = self.index_map[index]

        # 如果没预加载，就临时加载这个 shard
        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]

        visual = data["visuals"][local_idx]
        audio = data["audios"][local_idx]
        action = data["actions"][local_idx]

        return visual, audio, action
    

class ShardedPTDatasetOffline(Dataset):
    def __init__(self, shard_pattern="./dataset/pt/offline/offline_model_shard_*.pt", preload=True):
        """
        shard_pattern: shard 文件路径模式，比如 ./dataset/pt/foundation_model_shard_*.pt
        preload: 是否把所有 shard 一次性加载到内存（大数据集建议 False）
        """
        super().__init__()
        self.shard_files = sorted(glob.glob(shard_pattern))
        assert len(self.shard_files) > 0, f"No shards found at {shard_pattern}"

        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_map = []    # 全局 index → (shard_id, local_index)

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.shard_files):
            print(f"scan shard id {shard_id} ...")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        self.total_size = sum(self.shard_sizes)
        print(f"Total samples: {self.total_size}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        shard_id, local_idx = self.index_map[index]

        # 如果没预加载，就临时加载这个 shard
        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]

        # 取出一个 transition
        state       = data["states"][local_idx]
        next_state  = data["next_states"][local_idx]
        action      = data["actions"][local_idx]
        reward      = data["rewards"][local_idx]
        done        = data["dones"][local_idx]

        return state, action, reward, next_state, done
