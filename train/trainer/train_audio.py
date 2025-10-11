from train.network.audio import AudioCRNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from utils.log import logger
import numpy as np

class Train:
    def __init__(self, model, Adam, dataset,val_dataset, epoch, writer, save_dir , device='cuda'):
        self.train_model: AudioCRNN = model.to(device)
        self.optimizer = Adam
        self.epoch = epoch
        self.writer = writer
        self.device = device
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.save_dir = save_dir
        print(self.dataset.__len__())
        self.train_loader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=True)

    def deal_batch_angle(self , batch_angle):
        # batch_angle: (batch , 1)
        return torch.stack([torch.tensor((i.item()-30 , i.item()+30) , dtype=float) for i in batch_angle])
    
    # def angle_error(self, angle_predict , batch_angle):
    #     return torch.nn.ReLU(angle_predict - batch_angle)
    # def compute_loss_by_guss(self , angle_predict , batch_angle):
    #     # angle_pre[0] for i in angle_predict
    #     # batch_angle[0]
    #     self.guss(angle , lable)
    
    # def guss(self , angle , lable):
    #     error = lable - angle
    #     loss = 
    def train(self):
        global_step = 0

        for ep in range(self.epoch):
            local_step = 0
            epoch_angle_loss = 0.0
            
            for batch in self.train_loader:
                # unpack 数据
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    batch_visual, batch_audio, batch_action , batch_angle = batch
                    batch_visual, batch_audio = batch_visual.to(self.device).squeeze(0), batch_audio.to(self.device).squeeze(0)
                    batch_angle = batch_angle.float().to(self.device)
                else:
                    raise ValueError("训练数据格式不正确，应为 (visual, audio, label) 三元组")
                # batch_angle = self.deal_batch_angle(batch_angle).float().to(self.device)


                # import pdb;pdb.set_trace()
                angle_predict = self.train_model(batch_audio)
                # error = self.angle_error(angle_predict.squeeze(1) , batch_angle)
                loss_angle = F.mse_loss(angle_predict.squeeze(1) , batch_angle)
                # loss_angle = self.compute_loss_by_guss(angle_predict , batch_angle)
                # loss = loss_angle
                self.optimizer.zero_grad()
                loss_angle.backward()
                self.optimizer.step()
                
                print(f"Epoch {ep}, Step {global_step} , angle loss {loss_angle:.6f}")
                epoch_angle_loss += loss_angle.item()

                if self.writer:
                    self.writer.add_scalar("Loss/step_angle", loss_angle.item(), global_step)
                global_step += 1
                local_step += 1

                # if global_step % 500 == 0:
                #     self.validate(global_step)
            self.validate(global_step)


            avg_angle_loss = epoch_angle_loss / local_step
            if self.writer:
                self.writer.add_scalar("Loss/epoch_angle", avg_angle_loss, ep)
            print(f"Epoch {ep+1} finished, average angle loss: {avg_angle_loss:.4f}")
            if (ep + 1) % 50 == 0:
                save_path = f"{self.save_dir}/model_epoch_{ep+1}.pth"
                torch.save(self.train_model.state_dict(), save_path)
                tqdm.write(f"Saved model checkpoint to {save_path}")

    def validate(self, step):
        self.train_model.eval()
        total = 0
        val_angle_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    batch_visual, batch_audio, batch_action , batch_angle = batch
                    batch_visual, batch_audio = batch_visual.to(self.device).squeeze(0), batch_audio.to(self.device).squeeze(0)
                    batch_angle = batch_angle.float().to(self.device)
                else:
                    raise ValueError("验证数据格式不正确，应为 (visual, audio, label) 三元组")
                # batch_angle = self.deal_batch_angle(batch_angle).float().to(self.device)
                angle_predict = self.train_model(batch_audio)
                angle_loss = F.mse_loss(angle_predict.squeeze(1) , batch_angle)

                val_angle_loss += angle_loss.item()
   


        avg_angle_loss = val_angle_loss / len(self.val_loader)


        if self.writer:
            self.writer.add_scalar("Val/angle_loss", avg_angle_loss, step)

        self.train_model.train()

