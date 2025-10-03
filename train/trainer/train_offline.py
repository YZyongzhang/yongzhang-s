"""
利用offline RL进行训练。
"""
from train.network.OfflineNet import SAC_model
from train.utils.VEDA import VADE_Offline
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
from utils.log import logger
class SAC_train:
    def __init__(self, dataset, env , sac_model,foundation_model ,  writer , batch_size=64, device='cuda'):
        self.device = device
        self.agent = sac_model
        self.foundation_model = foundation_model
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True ,drop_last=True , num_workers = 16,pin_memory=True)
        self.writer = writer
        self.env = env

    def train(self, H = "offline" ,num_epochs=1000):
        
        global_step = 0
        for epoch in tqdm(range(1, num_epochs + 1),desc="epoch nums"):

            

            for batch in self.dataloader:
                
                global_step += 1
                states , actions , rewards , next_states , dones = batch
                loss_dict = self.agent.update(states, actions , rewards.unsqueeze(1), next_states ,  dones.unsqueeze(1))


                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"scalar/{key}", value, global_step=global_step)

                tqdm.write(f"step:{global_step} ,critic_1_loss:{loss_dict['critic_1_loss']} , critic_2_loss : {loss_dict['critic_2_loss']} , actor_loss : {loss_dict['actor_loss']} , alpha_loss:{loss_dict['alpha_loss']}")

            if epoch % 100 == 0 :
                path = f'./experiment/train/offline/ckpt/{H}/'
                times = time.localtime()
                time_str = f"{times.tm_year}-{times.tm_mon}-{times.tm_mday}-{times.tm_hour}_{times.tm_min}_{times.tm_sec}"
                os.makedirs(path,exist_ok=True)
                torch.save(self.agent.state_dict() , f'{path}/Foundation_model_{time_str}_{epoch}.pth')
            if epoch % 10 == 0:
                self.val(epoch)

    def val(self, epoch):
        self.agent.eval() 
        spl = 0
        sr = 0
        reward = 0

        for episode in tqdm(range(self.env._env.number_of_episodes)):
            logger.info(f"online val episode is {episode}/{self.env._env.number_of_episodes}")
            episode_reward = 0
            step = 0
            obs = self.env.reset()
            visual = torch.from_numpy(obs['rgb']).float() / 255
            audio = torch.from_numpy(obs['spectrogram']).float()
            done = False
            with torch.no_grad():
                state = self.foundation_model.embedding_forward(audio.unsqueeze(0).to(self.device) , visual.unsqueeze(0).to(self.device))
                action = torch.argmax(self.agent.actor(state),dim=1).item()
            while not done:
                obs , reward , done , info = self.env.step(action = action)
                visual = torch.from_numpy(obs['rgb']).float() / 255
                audio = torch.from_numpy(obs['spectrogram']).float()
                with torch.no_grad():
                    state = self.foundation_model.embedding_forward(audio.unsqueeze(0).to(self.device) , visual.unsqueeze(0).to(self.device))
                    action = torch.argmax(self.agent.actor(state),dim=1).item()
                episode_reward += reward
                step += 1
            logger.info(f"spl {info['spl']} , sr {info['success']} , reward {episode_reward}")
            spl+=info['spl']
            sr+=info['success']
            reward+=episode_reward
        logger.info(f"spl {spl} , sr {sr} , reward {reward}")  
        spl = spl/self.env._env.number_of_episodes
        sr = sr/self.env._env.number_of_episodes
        reward = reward/self.env._env.number_of_episodes
        logger.info(f"spl {spl} , sr {sr} , reward {reward}")

        self.writer.add_scalar("Val/spl", spl, global_step=epoch)
        self.writer.add_scalar("Val/sr", sr, global_step=epoch)
        self.writer.add_scalar("Val/reward", reward, global_step=epoch)
        

        self.agent.train()
