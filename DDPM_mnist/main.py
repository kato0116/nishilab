import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from tqdm import tqdm

from diffuser import Diffuser,UNet
from nishilab.DDPM_mnist.save_imgs import save_imgs

if __name__ == "__main__":
    img_size = 28
    batch_size = 128
    num_timesteps = 1000
    epochs = 10
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())

    preprocess = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='/root/data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    diffuser = Diffuser(num_timesteps, device=device)
    model = UNet()
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

            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        images = diffuser.sample(model)
        save_imgs(images,epoch+1)

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch {epoch} | Loss: {loss_avg}')

    # plot losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("/root/log/loss.png")

    # generate samples
    images = diffuser.sample(model)
    save_imgs(images,"pred")