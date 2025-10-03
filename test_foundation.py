from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch
from train import Network
from train import ShardedPTDataset
from train.trainer.train_foundation import Train
import os
if  __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network().to(device)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    save_dir = './experiment/train/ckpt_foundation_add_val_test'
    os.makedirs('./experiment/train/ckpt_foundation_add_val_test', exist_ok=True)
    writer = SummaryWriter(log_dir='./experiment/train/foundation_add_val_test')
    dataset = ShardedPTDataset(shard_pattern="./dataset/pt/foundation/foundation_model_shard_0.pt")
    val_dataset = ShardedPTDataset(shard_pattern="./dataset/pt/foundation_val/foundation_model_shard_0.pt")
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