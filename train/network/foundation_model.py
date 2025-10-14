import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader
from train.network.ViT import ViTEncoder , AudioEncoder

def layer_init(cnn):
    for layer in cnn:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)

class Attention(nn.Module):
    def __init__(self , input_dim , visual_dim , audio_dim ,hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


        # self.attention_weight = nn.Parameter(torch.randn(self.output_dim) * 0.01)
        layer_init([self.fc1])
        layer_init([self.fc2])
        
    def forward(self, encode):
        x1 = self.dropout(torch.relu(self.fc1(encode)))
        x2 = self.dropout(torch.relu(self.fc2(x1)))
        # self.attention_weight.unsqueeze(1)
        # import pdb;pdb.set_trace()
        # y = torch.matmul(x2.unsqueeze(-1) , self.attention_weight)
        # y = x2 * self.attention_weight
        return x2
    
# class Vote(nn.Module):
#     def __init__(self,input_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         # self.hidden_dim = hidden_dim
#         # self.output_dim = output_dim
        
#         self.voter = nn.Parameter(torch.randn(self.input_dim , 32) * 0.01)

#     def forward(self,embeding):
#         output = torch.matmul(embeding , self.voter)
#         return output
    
# class Decision(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim=1):
#         super(Decision, self).__init__()
#         self.input_dim = input_dim
#         self.fc1 = nn.Linear(input_dim , hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, input_matrix):
#         x = input_matrix.squeeze(1)
#         # import pdb;pdb.set_trace()
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class AnglePredictor(nn.Module):
    def __init__(self, hidden_dim = 128):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*128 , hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(16*128 , hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(128 , 64),
        #     nn.ReLU(),
        #     nn.Linear(64 , 8)
        # )
        layer_init(self.fc1)
        layer_init(self.fc2)

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        # 简单平均池化
        x = self.audio_encoder(x)
        x1 = self.dropout(self.fc1(x))
        x2 = self.fc2(x1)
        return x2
    
    # def forward(self, x):
    #     # x: [batch, seq_len, hidden_dim]
    #     # 简单平均池化
    #     std_x = (x - x.mean()) / (x.std() + 1e-6)
    #     x = self.audio_encoder(std_x)
    #     f_x = self.fc1[0](x)
    #     x1 = self.fc1[1](f_x)
    #     return x , x1 
    
class Finnal_model(nn.Module):
    def __init__(self,input_dim , hidden_dim , output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(self.input_dim , self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        layer_init([self.fc1])
        layer_init([self.fc2])
    def forward(self,embeding):
        x1 = torch.relu(self.fc1(embeding))
        x2 = self.dropout(x1)
        x3 = self.fc2(x2)
        return x3
    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.vitpartnet = ViTEncoder()
        self.attention = Attention(input_dim=512,visual_dim=256,audio_dim=256,hidden_dim=256,output_dim=128)
        # self.mermory = Mermory(input_dim=128,hidden_dim=128)
        # self.vote = Vote(input_dim=128)
        # self.decision = Decision(input_dim=32,hidden_dim=16,output_dim=4)
        self.angle_part = AnglePredictor(hidden_dim=128)
        self.finnal = Finnal_model(input_dim = 128 , hidden_dim=32 , output_dim=4)


    def forward(self,audio , visual):
        visual = visual.permute(0,3, 1, 2)
        audio = audio.permute(0,3, 1, 2)
        avencoder , audio_encode , visual_encoder = self.vitpartnet(audio , visual)
        angle = self.angle_part(audio_encode)
        attentioned = self.attention(avencoder)

        finnal_output = self.finnal(attentioned)
        return finnal_output , angle
    
    def embedding_forward(self , audio ,visual):
        visual = visual.permute(0,3, 1, 2)
        # audio = audio.permute(0,3, 1, 2)
        
        avencoder= self.vitpartnet(audio , visual)
        attentioned = self.attention(avencoder)
        return attentioned
    
