from train import Transfor
if __name__ == "__main__":
    paths = ['dataset/pickle/greedy(val)','dataset/pickle/collided(val)']
    transfor = Transfor(paths=paths)
    transfor.to_lmdb(type = 'foundation_model_torch_save')

