import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from tqdm import tqdm

from diffuser import Diffuser,UNet
from show_imgs import show_images

import pandas as pd
import wandb

if __name__ == "__main__":
    img_size   = 32
    batch_size = 128
    num_timesteps = 1000
    epochs = 300
    lr     = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())
    
    # wandbのセットアップ
    wandb.init(
        project="ddpm_cifar10",
        config={
        "learning_rate": 1e-3,
        "architecture": "DDPM",
        "dataset": "CIFAR-10",
        "epochs": 300,
        }
    )

    preprocess = transforms.ToTensor()
    dataset    = torchvision.datasets.CIFAR10(root='/root/data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    diffuser = Diffuser(num_timesteps, device=device)
    model    = UNet()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt      = 0

        # generate samples every epoch ===================
        # images = diffuser.sample(model)
        # show_images(images)
        # ================================================

        for images, labels in tqdm(dataloader):
            optimizer.zero_grad()
            x = images.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        images = diffuser.sample(model)
        show_images(images,epoch+1)

        loss_avg = loss_sum / cnt
        # wandbにlogの送信
        wandb.log({"loss":loss_avg})
        losses.append(loss_avg)
        print(f'Epoch {epoch+1} | Loss: {loss_avg}')

    # # plot losses
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig("/root/log/loss.png")
    
    log = {"epoch":range(epochs), "loss":losses}
    df_log = pd.DataFrame(log)
    df_log.to_csv('/root/log/loss.csv',index=False)
    
    model_path = '/root/weights/unet_cifar10.pth'
    torch.save(model.state_dict(), model_path)
    
    # generate samples
    images = diffuser.sample(model)
    show_images(images,"pred")
    
    wandb.alert(
        title = "実行終了",
        text  = "アーキテクチャ: DDPM, データセット: CIFAR10"
    )
    wandb.finish()