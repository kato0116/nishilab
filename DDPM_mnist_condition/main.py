import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from tqdm import tqdm

from diffuser import Diffuser,UNet
from nishilab.DDPM_mnist_condition.save_imgs import save_imgs
import pandas as pd
import wandb


if __name__ == "__main__":
    img_size = 28
    batch_size = 128
    num_timesteps = 1000
    epochs = 100
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())

    # wandbのセットアップ
    wandb.init(
        project="ddpm_cond_mnist",
        config={
        "learning_rate": 1e-3,
        "architecture": "DDPM_cond",
        "dataset": "MNIST",
        "epochs": 100,
        }
    )

    preprocess = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='/root/data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    diffuser = Diffuser(num_timesteps, device=device)
    model = UNet(num_labels=10)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        # generate samples every epoch ===================
        # images = diffuser.sample(model)
        # show_images(images)
        # ================================================

        for images, labels in tqdm(dataloader):
            optimizer.zero_grad()
            x = images.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)
            labels = labels.to(device)

            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t, labels)
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        images, labels = diffuser.sample(model)
        show_images(images,labels,epoch+1)

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch {epoch} | Loss: {loss_avg}')
        
        # wandbにlogの送信
        wandb.log({"loss":loss_avg})
        
    log = {"epoch":range(epochs), "loss":losses}
    df_log = pd.DataFrame(log)
    df_log.to_csv('/root/nishilab/DDPM_mnist_condition/log/loss.csv',index=False)
        
    model_path = '/root/nishilab/DDPM_mnist_condition/weights/unet_cond_mnist.pth'
    torch.save(model.state_dict(), model_path)

    # generate samples
    images,labels = diffuser.sample(model)
    show_images(images,labels,"pred")
    
    wandb.alert(
        title = "実行終了",
        text  = "アーキテクチャ: DDPM, データセット: MNIST"
    )
    wandb.finish()