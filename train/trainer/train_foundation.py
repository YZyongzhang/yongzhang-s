from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
class Train:
    def __init__(self, model, Adam, dataset,val_dataset, epoch, writer, save_dir , device='cuda'):
        self.train_model = model.to(device)
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
    def train(self):
        global_step = 0

        for ep in range(self.epoch):
            local_step = 0
            epoch_action_loss = 0.0
            epoch_angle_loss = 0.0
            # pbar = tqdm(self.train_loader, desc=f"Epoch {ep+1}/{self.epoch}")
            for batch in self.train_loader:
                # unpack 数据
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    batch_visual, batch_audio, batch_action , batch_angle = batch
                    batch_visual, batch_audio = batch_visual.to(self.device).squeeze(0), batch_audio.to(self.device).squeeze(0)
                    batch_angle = batch_angle.float().to(self.device)
                else:
                    raise ValueError("训练数据格式不正确，应为 (visual, audio, label) 三元组")

                batch_action = torch.tensor([int(a) for a in batch_action], dtype=torch.long).to(self.device)
                # import pdb;pdb.set_trace()
                action_predict , angle_predict = self.train_model(batch_audio, batch_visual)
                loss_action = F.cross_entropy(action_predict, batch_action)
                loss_angle = F.mse_loss(angle_predict.squeeze(1) , batch_angle)
                loss = loss_action + 5*loss_angle
                # loss = loss_angle
                self.optimizer.zero_grad()
                # loss_action.backward(retain_graph=True)
                # loss_angle.backward()
                loss.backward()
                self.optimizer.step()
                
                print(f"Epoch {ep}, Step {global_step}, action Loss: {loss_action:.6f} , angle loss {loss_angle:.6f}")
                epoch_action_loss += loss_action.item()
                epoch_angle_loss += loss_angle.item()

                if self.writer:
                    self.writer.add_scalar("Loss/step_action", loss_action.item(), global_step)
                    self.writer.add_scalar("Loss/step_angle", loss_angle.item(), global_step)
                global_step += 1
                local_step += 1

            avg_action_loss = epoch_action_loss / local_step
            avg_angle_loss = epoch_angle_loss / local_step
            if self.writer:
                self.writer.add_scalar("Loss/epoch_action", avg_action_loss, ep)
                self.writer.add_scalar("Loss/epoch_angle", avg_angle_loss, ep)
            print(f"Epoch {ep+1} finished, average action loss: {avg_action_loss:.4f}")
            print(f"Epoch {ep+1} finished, average angle loss: {avg_angle_loss:.4f}")
            self.validate(ep)
            if (ep + 1) % 50 == 0:
                save_path = f"{self.save_dir}/model_epoch_{ep+1}.pth"
                torch.save(self.train_model.state_dict(), save_path)
                tqdm.write(f"Saved model checkpoint to {save_path}")

    def validate(self, epoch):
        self.train_model.eval()
        correct = 0
        total = 0
        val_action_loss = 0.0
        val_angle_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    batch_visual, batch_audio, batch_action , batch_angle = batch
                    batch_visual, batch_audio = batch_visual.to(self.device).squeeze(0), batch_audio.to(self.device).squeeze(0)
                    batch_angle = batch_angle.float().to(self.device)
                else:
                    raise ValueError("验证数据格式不正确，应为 (visual, audio, label) 三元组")

                batch_action = torch.tensor([int(a) for a in batch_action], dtype=torch.long).to(self.device)

                action_predict , angle_predict = self.train_model(batch_audio, batch_visual)
                action_loss = F.cross_entropy(action_predict, batch_action)
                angle_loss = F.mse_loss(angle_predict.squeeze(1) , batch_angle)

                val_action_loss += action_loss.item()
                val_angle_loss += angle_loss.item()
                preds = action_predict.argmax(dim=1)  # [batch]
                correct += (preds == batch_action).sum().item()
                total += batch_action.size(0)

        acc = correct / total if total > 0 else 0
        avg_action_loss = val_action_loss / len(self.val_loader)
        avg_angle_loss = val_angle_loss / len(self.val_loader)
        print(f"[Val] Epoch {epoch+1}: Loss={avg_action_loss:.4f}, Acc={acc:.4f}")

        if self.writer:
            self.writer.add_scalar("Val/action_loss", avg_action_loss, epoch)
            self.writer.add_scalar("Val/angle_oss", avg_angle_loss, epoch)
            self.writer.add_scalar("Val/Acc", acc, epoch)

        self.train_model.train()

