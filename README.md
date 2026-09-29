# UAGLNet 기반 위성 이미지 건물 영역 분할

**2023 위성 이미지 건물 영역 분할 경진대회** 참가를 위해 [UAGLNet](https://ieeexplore.ieee.org/document/11322680)(*Uncertainty-Aggregated Global-Local Fusion Network with Cooperative CNN-Transformer for Building Extraction*, IEEE TGRS) 논문 코드를 대회 데이터 형식(RLE 인코딩 마스크, 1024×1024 위성 이미지)에 맞게 커스터마이징한 학습 파이프라인임.

메인 학습 스크립트는 [UAGLNet_train.py](UAGLNet_train.py) 이고, 대응하는 설정 파일은 [inria/UAGLNet_finetune.py](inria/UAGLNet_finetune.py) 임.

실험 기록과 설계 근거는 [프로젝트 노트](https://chb2066.github.io/projects/satellite-buildings/)에 있음.

## Results

| 구성 | Test |
|---|---:|
| DINOv2 frozen + segmentation head | 0.03 ~ 0.07 |
| Prithvi frozen + segmentation head | 0.03 ~ 0.07 |
| UAGLNet fine-tune, 해상도 정렬 전 | baseline 이하 |
| UAGLNet fine-tune, 해상도 정렬 후 | 0.65 |
| + 224 패치 학습으로 전환 | **0.79** |
| + 실패 패치 정제 | **0.80** |

train 1024×1024 / test 224×224 라는 해상도 차이를 맞추는 방식이 성능을 갈랐음.
학습과 추론의 겉보기 객체 크기를 일치시키자 0.03~0.07 에서 0.65 로 올랐고,
224 패치로 학습해 추론 시 해상도 조정을 없애자 0.79 가 됐음.

## 원본 코드 대비 변경 사항

- **패치 기반 학습 데이터셋**: 1024×1024 원본 이미지를 224×224 크기, stride 200(overlap 있음)으로 잘라 이미지당 5×5=25개 패치를 생성해 학습에 사용 ([inria/UAGLNet_finetune.py](inria/UAGLNet_finetune.py)의 `SatelliteDataset`).
- **오답 노트 기반 데이터 클리닝**: 학습 중 패치 단위 mIoU가 threshold(0.35) 미만인 패치를 `clean_run_failure_*.csv`로 기록하고, 이전 학습에서 누적된 실패 패치 목록(`failure_csv`)을 다음 학습에서 제외해 정제된 데이터로만 재학습.
- **사전학습 가중치 초기화**: Hugging Face에 공개된 `ldxxx/UAGLNet_Inria` checkpoint를 불러와 초기화한 뒤 대회 데이터로 fine-tuning.
- **검증 없이 학습(val-free)**: 대회 제출 성능 극대화를 위해 별도 validation split 없이 전체 학습 데이터를 사용(`val_loader=None`).

## 설치

```bash
git clone https://github.com/chb2066/satellite-building-segmentation.git
cd satellite-building-segmentation

conda create -n uaglnet python=3.11 -y
conda activate uaglnet
pip install -r requirements.txt
```

CUDA 환경에 맞는 PyTorch 빌드가 필요하다면 [pytorch.org](https://pytorch.org/get-started/locally/) 안내에 따라 `torch`/`torchvision`을 먼저 설치한 뒤 나머지 의존성을 설치하면 됨.

## 데이터 준비

`inria/UAGLNet_finetune.py`는 다음 경로를 기준으로 데이터를 읽음.

- `data_root`(기본값 `/workspace`) 아래 `train.csv`: 대회에서 제공하는 학습 메타데이터로, 다음 3개 컬럼을 이 순서(인덱스 0, 1, 2)로 포함해야 함: `이미지 ID`, `이미지 경로`, `RLE 인코딩된 마스크`.
- 이미지 경로는 `train.csv`에 기록된 상대 경로(`./` 접두사는 제거됨)를 `data_root` 와 결합해 참조함.

본인 환경에 맞게 `inria/UAGLNet_finetune.py`의 `data_root`와 `failure_csv_path`를 수정한 뒤 사용하면 됨. `failure_csv_path`가 가리키는 파일이 없으면(첫 학습 시) 오답 노트 필터링 없이 전체 데이터로 학습함.

## 학습

```bash
python UAGLNet_train.py -c inria/UAGLNet_finetune.py
```

- 체크포인트: `results/<task_name>/ckpts/`에 `save_freq` 주기로 저장되며, validation이 활성화된 경우 최고 성능 모델은 `best_epoch.pth` 로 별도 저장됨.
- 학습 로그: `results/<task_name>/log.txt`
- 오답 노트: 매 epoch 종료 시 `results/<task_name>/clean_run_failure_epoch_<N>.csv`, 학습 종료 시 `clean_run_failure_final.csv` 로 저장됨.

> **멀티 GPU 참고**: `UAGLNet_train.py`는 GPU가 2개 이상 감지되면 자동으로 `torch.nn.DataParallel` 로 모델을 감쌈. 일부 멀티 GPU 환경(GPU 간 P2P/NCCL 통신 제약이 있는 경우 등)에서는 이 경로가 멈추는(hang) 현상이 관찰됐음. 학습이 첫 iteration에서 진행되지 않는다면 `CUDA_VISIBLE_DEVICES=0`으로 단일 GPU를 지정해 실행해 볼 것.

## Citation

이 저장소는 아래 원 논문의 코드를 기반으로 함.

```
@article{UAGLNet,
  title   = {UAGLNet: Uncertainty-Aggregated Global-Local Fusion Network with Cooperative CNN-Transformer for Building Extraction},
  author  = {Siyuan Yao and Dongxiu Liu and Taotao Li and Shengjie Li and Wenqi Ren and Xiaochun Cao},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2025}
}
```

## Acknowledgement

이 프로젝트는 [UAGLNet](https://github.com/Dstate/UAGLNet) 원본 코드를 기반으로 하고, 원본은 [BuildingExtraction](https://github.com/stdcoutzrh/BuildingExtraction), [GeoSeg](https://github.com/WangLibo1995/GeoSeg/tree/main), [SMT](https://github.com/AFeng-x/SMT) 를 참고해 작성됐음.

## License

이 프로젝트는 [Apache License 2.0](LICENSE) 을 따름.
