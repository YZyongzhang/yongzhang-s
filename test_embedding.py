from train import Transfor
if __name__ == "__main__":
    paths = ['dataset/pickle/collided(train)']
    transfor = Transfor(paths=paths)
    transfor.to_lmdb(type = 'offline_rl_embedding_save')
