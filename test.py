# http://meg.aalip.jp/python/BrainVISA_nibabel.html
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import math

path = "/root/volume/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00121-000/BraTS-GLI-00121-000-seg.nii.gz"
image = nib.load(path)
img0=image.get_fdata()
print(img0.shape)
plt.imshow(img0[:,:,70].T,cmap='gray',origin='lower')
plt.show()
plt.savefig("./test.png")