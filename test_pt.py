from train import Transfor
import os
if __name__ == "__main__":
    paths = ['dataset/pickle/v2/greedy(val)','dataset/pickle/v2/collided(val)']
    transfor = Transfor(paths=paths)
    path = 'dataset/pt/v2/foundation_audio_with_angle_val/' 
    os.makedirs(path , exist_ok=True)
    transfor.to_lmdb(type = 'foundation_model_torch_save' , save_dir = path , step = 20)

