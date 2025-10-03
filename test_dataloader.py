import torch
from torch.profiler import profile, record_function, ProfilerActivity
import lmdb
from train import network
from train import VADE
from torch.utils.data import DataLoader
path = '/data/Getuanhui/9_30_cvpr/lmdb/greedy'
lmdb_env = lmdb.open(path, readonly=True, lock=False)
txn = lmdb_env.begin()
dataset = VADE(lmdb_env = lmdb_env  , txn = txn)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True , num_workers=4, pin_memory=True)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 只跑前 10 个 batch
            break

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))