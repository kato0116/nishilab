import torch
import wandb
print(torch.cuda.is_available())
print("test")
wandb.init()
wandb.finish()