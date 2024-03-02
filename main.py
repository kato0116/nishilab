import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# HuggingFaceのAccelerator
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
# Wandb
import wandb

from save_log import create_folder,save_imgs,save_model
from diffuser import Diffuser,UNet

class CFG:
    path          = "/root/volume"
    seed          = 42
    model_name    = 'ddpm'
    dataset       = 'mnist'
    img_size      = 28
    batch_size    = 256
    epochs        = 15
    lr            = 1e-3
    T_max         = 1000
    # device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Acceleratorの設定 (1/3)
    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device  = accelerator.device
    save_n_model  = 3
    save_n_imgs   = 1

if __name__ == "__main__":

    
    dir_path = os.path.join(CFG.path,"log",CFG.model_name+"_epochs_"+str(CFG.epochs))
    create_folder(dir_path)

    # wandbのセットアップ
    wandb.init(
        project="ddpm_cifar10",
        config={
        "learning_rate": CFG.lr,
        "architecture":  CFG.model_name,
        "dataset": CFG.dataset,
        "epochs":  CFG.epochs,
        }
    )
    
    preprocess = transforms.ToTensor()
    dataset    = torchvision.datasets.MNIST(root=CFG.path, download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)
    diffuser   = Diffuser(CFG.T_max, device=CFG.device)
    
    model = UNet()
    model.to(CFG.device)
    optimizer = Adam(model.parameters(), lr=CFG.lr)
    
    # Acceleratorの設定 (2/3)
    model, optimizer, dataloader = CFG.accelerator.prepare(
        model,optimizer,dataloader
    )
    
    losses = []
    for epoch in range(CFG.epochs):
        # loss_sum = 0.0
        # cnt = 0
        batch_losses = []

        for images, labels in tqdm(dataloader):
            optimizer.zero_grad()
            x = images.to(CFG.device)
            t = torch.randint(1, CFG.T_max+1, (len(x),), device=CFG.device)

            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise, noise_pred)

            # loss.backward()
            CFG.accelerator.backward(loss)
            optimizer.step()
            
            batch_losses.append(loss.item())
            # loss_sum += loss.item()
            # cnt += 1
            
        images = diffuser.sample(model)
        if (epoch+1)%CFG.save_n_imgs == 0:
            save_imgs(images,epoch+1,dir_path)
        if (epoch+1)%CFG.save_n_model == 0:
            save_model(model,epoch+1,dir_path)
        # accelerator.printを使うことでメインプロセスのみprintできる
        # loss_avg = loss_sum / cnt
        loss_avg = np.array(batch_losses).mean()
        # wandbにlogの送信
        wandb.log({"loss":loss_avg})
        losses.append(loss_avg)
        print(f'Epoch {epoch+1} | Loss: {loss_avg}')
        CFG.accelerator.print(f'Epoch: {epoch+1}\tloss: {np.array(batch_losses).mean()}')

    log = {"epoch":range(CFG.epochs), "loss":losses}
    df_log = pd.DataFrame(log)
    log_path = os.path.join(dir_path,'loss.csv')
    df_log.to_csv(log_path,index=False)

    # generate samples
    images = diffuser.sample(model)
    save_imgs(images,"pred",dir_path)
    save_model(model,epoch+1,dir_path)
    
    wandb.finish()