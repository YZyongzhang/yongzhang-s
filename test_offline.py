from train import SAC_model , Network , ShardedPTDatasetOffline
from train.trainer.train_offline import SAC_train
from env import ENV as env
import torch
import time

if __name__ == "__main__":
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f"./experiment/train/offline/loss")
    dataset = ShardedPTDatasetOffline()

    state_dim = 128
    action_dim = 4
    hidden_dim = 64
    lr = 1e-4
    target_entropy = -action_dim  
    tau = 0.005
    gamma = 0.99
    batch_size = 64
    num_epochs = 4000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    foundation_model = Network().to(device)
    foundation_model.load_state_dict(torch.load('experiment/train/ckpt/model_epoch_600.pth'))
    foundation_model.eval()

    sac_model = SAC_model(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        actor_lr=lr,
        critic_lr=lr,
        alpha_lr=lr,
        target_entropy=target_entropy,
        tau=tau,
        gamma=gamma,
        beta=1.0,
        device=device
    )



    trainer = SAC_train (
        dataset=dataset,
        env = env,
        sac_model=sac_model,
        foundation_model=foundation_model,
        batch_size=batch_size,
        device=device,
        writer=writer
    )

    trainer.train(num_epochs=num_epochs)
    # trainer.val(1)


    writer.close()