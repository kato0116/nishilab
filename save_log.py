import os
import matplotlib.pyplot as plt
import torch

def create_folder(dir_path):
    """
    in
    ==============================
    dir_path:   ディレクトリのパス

    out
    ==============================
    log記録用のディレクトリ作成
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    imgs_dir_path = os.path.join(dir_path,"imgs")
    if not os.path.exists(imgs_dir_path):
        os.mkdir(imgs_dir_path)
        
    weights_dir_path = os.path.join(dir_path,"weights")
    if not os.path.exists(weights_dir_path):
        os.mkdir(weights_dir_path)
        
def save_imgs(imgs, epoch, dir_path, rows=2, cols=10):
    """
    in
    ==============================
    imgs: 画像
    epoch: 現在のエポック数
    dir_path: ディレクトリのパス
    
    out
    ==============================
    画像保存
    """
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    imgs_path = os.path.join(dir_path,"imgs",f"epoch_{epoch}.png")
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(imgs[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.savefig(imgs_path)

def save_model(model,epoch,dir_path):
    """
    in
    ===============================
    model: モデルの重み
    epoch: 現在のエポック数
    dir_path: ディレクトリを指定
    
    out
    ===============================
    重み保存
    """
    model_path = os.path.join(dir_path,"weights",f"weight_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)