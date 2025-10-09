from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch
from train.network.foundation_model import Network
from train.utils.VEDA import ShardedPTDataset
from train.trainer.train_foundation import Train
import os
if  __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network().to(device)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr ,  weight_decay=1e-4)
    save_dir = './experiment/train/v2/ckpt_foundation_train_split_v_10_6'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='./experiment/train/v2/ckpt_foundation_train_split_v_10_7')
    dataset = ShardedPTDataset(shard_pattern="./dataset/pt/v2/foundation_audio_with_angle_train_split_v_10_6/foundation_model_shard_*.pt")
    val_dataset = ShardedPTDataset(shard_pattern="./dataset/pt/v2/foundation_audio_with_angle_val_train_split_v_10_6/foundation_model_shard_*.pt")
    trainer = Train(
        model=model,
        Adam=optimizer,
        dataset= dataset,
        val_dataset=val_dataset,
        epoch=1000,
        writer=writer,
        save_dir = save_dir
    )
    trainer.train()