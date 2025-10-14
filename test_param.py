import torch
import torch.nn as nn
import numpy
import pickle
ckpt = './experiment/train/v3/audio_8/model_epoch_50.pth'
from train import Network
from train.network.foundation_model import Network , AnglePredictor
import numpy as np
np.set_printoptions(suppress=True)
foundation_model = AnglePredictor().to(torch.device('cuda'))
foundation_model.load_state_dict(torch.load(ckpt))
foundation_model.eval()
with open('val_loader.pkl' , 'rb') as f:
    data = pickle.load(f)
for batch in data:
    batch_visual, batch_audio, batch_action , batch_angle = batch
    break
batch_audio = batch_audio.to('cuda')
# for name , param in foundation_model.named_parameters():
#     if 'weight' in name and "fc1.1" in name:
#         f = param
embedding0 , embedding1  = foundation_model(batch_audio)
embedding0 = nn.Flatten()(embedding0)
embedding0_1 = embedding0[1]

# f = np.array(f.data.cpu().t())
import pdb;pdb.set_trace()