from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.multiprocessing as mp

import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import time

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'

  # initialize the process group
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

def train(ddp_model, rank, world_size, train_loader, optimizer, epoch):
  ddp_model.train()
  for batch_idx, (data, label) in enumerate(train_loader):
    data, label = data.to(rank), label.to(rank)
    optimizer.zero_grad()
    output = ddp_model(data)
    loss = F.nll_loss(output, label)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
      print('rank: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          rank, epoch + 1, batch_idx * len(data) * world_size, len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

def test(ddp_model, rank, test_loader):
  ddp_model.eval()
  test_loss = 0
  correct = 0
  data_num = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(rank), target.to(rank)
      output = ddp_model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      data_num += len(data)

  test_loss /= data_num

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) [rank: {}]\n'.format(
      test_loss, correct, data_num,
      100. * correct / data_num, rank))

def main(rank, world_size, batch_size, epochs):
  print(f"Running basic DDP example on rank {rank}.")
  setup(rank, world_size)

  # create model and move it to GPU with id rank
  model = Net().to(rank)
  ddp_model = DDP(model, device_ids=[rank])

  transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

  dataset = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
  dataset2 = datasets.MNIST('../data', train=False,
                      transform=transform)

  optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      dataset, num_replicas=world_size, rank=rank, shuffle=False
  )
  train_loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler
  )

  test_sampler = torch.utils.data.distributed.DistributedSampler(
      dataset2, num_replicas=world_size, rank=rank, shuffle=False
  )
  test_loader = torch.utils.data.DataLoader(
      dataset2, batch_size=batch_size, shuffle=test_sampler is None, sampler=test_sampler
  )

  for epoch in range(epochs):
    train(ddp_model=ddp_model, rank=rank, world_size=world_size,
      train_loader=train_loader, optimizer=optimizer, epoch=epoch)
    test(ddp_model=ddp_model, rank=rank, test_loader=test_loader)

  cleanup()


def run(demo_fn, world_size=1, batch_size=2, epochs=1):
  mp.spawn(demo_fn,
    args=(world_size,batch_size,epochs),
    nprocs=world_size,
    join=True)

if __name__ == "__main__":
  start = time.time()
  run(main, world_size=4, batch_size=128, epochs=100)
  print("Process time: ", time.time() - start)
