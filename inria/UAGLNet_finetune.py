import os
import collections
import collections.abc
# catalyst==20.09 references collections.MutableMapping, which was removed in Python 3.10.
# Restore the alias so `from catalyst...` imports below don't crash on Python 3.10+.
collections.MutableMapping = collections.abc.MutableMapping

import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from geoseg.losses import *
from tools.metric import Evaluator
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import math
from tqdm import tqdm
from datetime import datetime
from geoseg.models.UAGLNet import UAGLNet
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ==========================================
# 1. 베이스라인 유틸리티 함수
# ==========================================
def rle_decode(mask_rle, shape=(1024, 1024)):
    if str(mask_rle) == '-1': 
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def get_training_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MultiplicativeNoise(
        multiplier=(0.7, 0.9), # 70%~90% 밝기로 무작위 저하
        elementwise=True,      # 픽셀마다 다르게 적용 (매우 중요)
        p=0.5
        ),
        A.CLAHE(
        clip_limit=4.0,
        tile_grid_size=(4, 4), 
        p=0.5
        ),
        A.Sharpen(alpha=(0.1, 0.3), p=0.3),
        A.Normalize(),
        ToTensorV2()
    ])

# ==========================================
# 2. 오답 패치 전면 제거 커스텀 데이터셋
# ==========================================
class SatelliteDataset(Dataset):
    def __init__(self, csv_file, data_root='/workspace', transform=None, failure_csv=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.data_root = data_root
        
        self.patch_size = 224
        self.n_patches_per_side = 5
        self.patches_per_image = 25
        
        self.total_indices = list(range(len(self.data) * self.patches_per_image))
        
        # [핵심] 오답 노트에 이름이 있으면 무조건 제외 대상을 생성 (Set으로 최적화)
        self.failure_set = set()
        if failure_csv and os.path.exists(failure_csv):
            fail_df = pd.read_csv(failure_csv)
            self.failure_set = set(fail_df['image_name'].unique())
            print(f"[*] 오답 노트 로드 완료: {len(self.failure_set)}개 패치를 학습에서 제외합니다.")

        # [핵심] 오답 노트에 없는 '깨끗한' 패치만 active_indices에 저장
        self.active_indices = []
        for idx in self.total_indices:
            name = self._get_patch_name(idx)
            if name not in self.failure_set:
                self.active_indices.append(idx)
        
        print(f"[필터링 결과] 전체 {len(self.total_indices)}개 중 {len(self.active_indices)}개 패치로 학습 진행.")

    def _get_patch_name(self, idx):
        r_idx = idx // self.patches_per_image
        p_idx = idx % self.patches_per_image
        img_path = self.data.iloc[r_idx, 1]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        return f"{img_id}_patch_{p_idx:02d}"

    def set_stage(self, stage):
        # train.py 에폭별 로직과의 호환성을 위해 유지하되, 아무 작업도 수행하지 않음
        return None

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx]
        real_idx = actual_idx // self.patches_per_image
        patch_idx = actual_idx % self.patches_per_image
        
        img_path = os.path.join(self.data_root, self.data.iloc[real_idx, 1].replace('./', ''))
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rle = self.data.iloc[real_idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        stride = 200 
        row, col = patch_idx // self.n_patches_per_side, patch_idx % self.n_patches_per_side
        y1, x1 = row * stride, col * stride
        y2, x2 = y1 + self.patch_size, x1 + self.patch_size

        image_patch, mask_patch = image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

        if self.transform:
            augmented = self.transform(image=image_patch, mask=mask_patch)
            image_patch, mask_patch = augmented['image'], augmented['mask'].long()

        return {'img': image_patch, 'gt_semantic_seg': mask_patch, 'name': self._get_patch_name(actual_idx)}

# ==========================================
# 3. 모델 및 학습 설정
# ==========================================
now = datetime.now().strftime('%m%d_%H%M')
task_name = f'UAGLNet_CleanOnly_{now}'
save_path = os.path.join('./results', task_name)
save_freq = 2
dev_freq = 1000
num_workers = 28
class_num, n_epochs = 2, 30
batch_size, evaluator = 56, Evaluator(num_class=2)
loss_func = UAGLloss()

net = UAGLNet(drop_path_rate=0.2, pretrained_backbone=None)
try:
    ckpt_path = hf_hub_download(repo_id="ldxxx/UAGLNet_Inria", filename="model.safetensors")
    net.load_state_dict(load_file(ckpt_path))
    print("성공: 가중치 로드 완료")
except Exception as e:
    print(f"가중치 로드 실패: {e}")

# ==========================================
# 4. 데이터 로더 초기 설정
# ==========================================
data_root = '/workspace'
failure_csv_path = f"{data_root}/UAGLNet/results/UAGLNet_224_Patch_Overlap_0129_0355/cumulative_failure_report_final.csv" 

train_dataset = SatelliteDataset(
    csv_file=f"{data_root}/train.csv",
    data_root=data_root,
    failure_csv=failure_csv_path,
    transform=get_training_transform()
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=28, pin_memory=True, shuffle=True)
val_loader = None 

layerwise_params = {"CoE.*": dict(lr=4e-3, weight_decay=0.0025)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
optimizer = Lookahead(torch.optim.AdamW(net_params, lr=4e-3, weight_decay=0.0025))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
