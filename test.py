import torch
import wandb
import shutil
import os

# 元のディレクトリと移動先ディレクトリのパス
src_directory = '/root/volume/img_align_celeba/'
dest_directory = '/root/volume/img_align_celeba/images/'

# 移動先ディレクトリが存在しない場合は作成
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

# 元のディレクトリ内の全ファイルに対して移動処理を実行
for filename in os.listdir(src_directory):
    # ファイルのフルパスを取得
    file_path = os.path.join(src_directory, filename)
    
    # ディレクトリ（フォルダ）でないことを確認してから移動
    if os.path.isfile(file_path):
        # ファイルを移動
        shutil.move(file_path, dest_directory)
        
        
# print(torch.cuda.is_available())
# print("test")
# wandb.init()
# wandb.finish()
# print("接続確認")