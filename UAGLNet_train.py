import os
import logging
import torch
import numpy as np
import argparse
import random
import copy
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from tools.cfg import py2cfg

# 1. 시드 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Current seed is {seed}")

seed_everything(42)

# 2. 모델 저장 함수
def model_save(model, save_path):
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)

# 3. 오답 노트 저장 함수
def save_failure_report(failure_tracker, save_path, suffix="final"):
    if not failure_tracker:
        return
    sorted_failures = sorted(failure_tracker.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_failures, columns=['image_name', 'failure_count'])
    output_file = os.path.join(save_path, f'clean_run_failure_{suffix}.csv')
    df.to_csv(output_file, index=False)

# 4. 검증 함수
def dev(model, val_loader, evaluator):
    model.eval()
    evaluator.reset()
    tbar = tqdm(val_loader, position=0, leave=True, desc="Validating")
    for batch in tbar:
        img, mask = batch['img'], batch['gt_semantic_seg']
        img, mask = img.cuda(), mask.cuda()
        with torch.no_grad():
            outputs = model(img)
            pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            pre_mask = torch.nn.Softmax(dim=1)(pred).argmax(dim=1)
            for i in range(mask.shape[0]):
                evaluator.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        
        mIoU = evaluator.Intersection_over_Union().mean()
        tbar.set_postfix(mIoU=f'{mIoU:.4f}')
    return evaluator.Intersection_over_Union()

# 5. 수정된 학습 함수 (Clean Only)
def train(config, model, train_loader, val_loader, loss_func, optimizer, scheduler, evaluator):
    epoch = 0
    best_ep = -1
    best_record = 0.
    n_epoch = config.n_epochs
    train_eval = copy.deepcopy(evaluator)
    
    dataset = train_loader.dataset
    current_loader = train_loader # 에폭별 교체 없이 고정된 로더 사용
    
    strategy_failure_tracker = {}
    threshold = 0.35

    while epoch < n_epoch:
        epoch += 1
        
        # [삭제됨] 에폭 1, 4에서 발생하던 단계별 로더 교체 로직 제거

        model.train()
        loss_records = []
        train_eval.reset()

        print(f'\nTraining Epoch {epoch:3d} (Samples: {len(dataset)}):')
        tbar = tqdm(current_loader, position=0, leave=True)
        
        for batch in tbar:
            img, mask = batch['img'], batch['gt_semantic_seg']
            img_names = batch['name']
            img, mask = img.cuda(), mask.cuda()
            
            optimizer.zero_grad()
            outputs = model(img)
            
            if isinstance(outputs, (tuple, list)):
                pred, prob_high, prob_low = outputs
                loss = loss_func(pred, prob_high, prob_low, mask)
            else:
                pred = outputs
                loss = loss_func(pred, mask)
                
            loss.backward()
            optimizer.step()
            loss_records.append(loss.item())
            
            # 실시간 성능 및 오답 추적
            pre_mask = torch.nn.Softmax(dim=1)(pred).argmax(dim=1)
            for i in range(mask.shape[0]):
                evaluator.reset()
                evaluator.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
                indiv_miou = evaluator.Intersection_over_Union().mean()
                
                if indiv_miou < threshold:
                    strategy_failure_tracker[img_names[i]] = strategy_failure_tracker.get(img_names[i], 0) + 1
                
                train_eval.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
            
            tbar.set_postfix(loss=f'{np.mean(loss_records):.4f}', mIoU=f'{train_eval.Intersection_over_Union().mean():.4f}')
        
        scheduler.step()
        save_failure_report(strategy_failure_tracker, config.save_path, suffix=f"epoch_{epoch}")
        
        # 검증 (val_loader가 None일 경우 안전하게 패스)
        if val_loader is not None:
            dev_records = dev(model, val_loader, evaluator)
            dev_miou = dev_records.mean()
            logging.info(f'Epoch {epoch:d}: train_mIoU = {train_eval.Intersection_over_Union().mean():.4f}, val_mIoU = {dev_miou:.4f}')
            
            if best_record < dev_miou:
                best_record = dev_miou
                best_ep = epoch
                model_save(model, os.path.join(config.save_path, 'ckpts', 'best_epoch.pth'))
        
        if epoch % config.save_freq == 0:
            model_save(model, os.path.join(config.save_path, 'ckpts', f'{epoch}.pth'))

    save_failure_report(strategy_failure_tracker, config.save_path, suffix="final")
    print(f"Best mIoU: {best_record:.4f} at epoch {best_ep}")

# 6. 메인 실행부
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=Path, default="inria/UAGLNet_finetune.py")
    return parser.parse_args()

def create_save_path(save_path):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ckpts'), exist_ok=True)

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    create_save_path(config.save_path)

    logging.basicConfig(filename=os.path.join(config.save_path, 'log.txt'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    model = config.net
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    # config에 val_loader가 없을 경우 None으로 처리하여 에러 방지
    train_loader = getattr(config, 'train_loader', None)
    val_loader = getattr(config, 'val_loader', None)

    train(config, model, train_loader, val_loader, config.loss_func, 
          config.optimizer, config.lr_scheduler, config.evaluator)

if __name__ == "__main__":
    main()
