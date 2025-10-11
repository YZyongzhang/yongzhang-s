from train import Transfor
import os
if __name__ == "__main__":
    paths = ['dataset/pickle/v2/greedy(train_angle)']
    transfor = Transfor(paths=paths)
    path = 'dataset/pt/v2/foundation_spectmel/' 
    os.makedirs(path , exist_ok=True)
    transfor.to_lmdb(type = 'foundation_model_torch_save' , save_dir = path , b_s = 0 , e_s = -1 ,b_f = 0 , e_f = -1)

