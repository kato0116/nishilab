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
    dataset       = 'CelebA'
    img_size      = 128
    channel       = 3
    batch_size    = 128
    epochs        = 100
    lr            = 1e-3
    T_max         = 1000
    # device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Acceleratorの設定 (1/3)
    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device  = accelerator.device
    save_n_model  = 3
    save_n_imgs   = 1
    wandb_num_images = 16 # wandbに送信する画像の数
    
if __name__ == "__main__":
    dir_path = os.path.join(CFG.path,"log",CFG.model_name+"_epochs_"+str(CFG.epochs)+":"+CFG.dataset)
    create_folder(dir_path)

    # wandbのセットアップ
    wandb.init(
        project="ddpm_celebA",
        config={
        "learning_rate": CFG.lr,
        "architecture":  CFG.model_name,
        "dataset": CFG.dataset,
        "epochs":  CFG.epochs,
        }
    )
    
    transform = transforms.Compose([
        transforms.Resize((CFG.img_size,CFG.img_size)),
        transforms.ToTensor(),
    ])
    dataset    = torchvision.datasets.ImageFolder(root="/root/volume/img_align_celeba",transform=transform)
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
    for epoch in range(1,CFG.epochs+1):
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
            
        images = diffuser.sample(model)    
        # accelerator.printを使うことでメインプロセスのみprintできる
        loss_avg = np.array(batch_losses).mean()
        
        # wandbにimages,logの送信
        wandb_images = [wandb.Image(images[i]) for i in range(CFG.wandb_num_images)]
        wandb.log(
            {"generated_images": wandb_images, "loss":loss_avg}
            )
        
        losses.append(loss_avg)
        print(f'Epoch {epoch} | Loss: {loss_avg}')
        # CFG.accelerator.print(f'Epoch: {epoch}\tloss: {np.array(batch_losses).mean()}')
        
        if (epoch)%CFG.save_n_imgs == 0:
            save_imgs(images,epoch+1,dir_path)
        if (epoch)%CFG.save_n_model == 0:
            save_model(model,epoch+1,dir_path)
            
    log = {"epoch":range(CFG.epochs), "loss":losses}
    df_log   = pd.DataFrame(log)
    log_path = os.path.join(dir_path,'loss.csv')
    df_log.to_csv(log_path,index=False)

    # generate samples
    images = diffuser.sample(model,x_shape=(20,CFG.channel, CFG.img_size, CFG.img_size))
    save_imgs(images,"pred",dir_path)
    save_model(model,epoch+1,dir_path)
    
    wandb.finish()