# massive_pretrained_cnn_model_v1

객체 탐지의 백본을로 사용하기 위해 CNN을 PyTorch로 직접 설계하고, **500-class 커스텀 ImageNet 데이터셋(약 26만장)으로 처음부터(from scratch) 사전학습**한 CNN 모델입니다.

torchvision의 pretrained weight 또는 외부 모델을 사용하는 대신 백본 구조 설계 → 데이터 파이프라인 → 학습 까지 과정을 구현했습니다.

# 사전학습 모델을 만든 이유

- 객체 탐지에서 **warm start** 시작하여 초기화된 상태부터 시작하지않아 학습 속도가 높아집니다.
- 백본 내부 구조에 대한 설계 결정을 직접 작성하고 CNN이 실제로 무엇을 학습하는지 확인해 볼 수 있습니다.

## 아키텍처

`backbone.py`

```
Input 
 └─ Stem      Conv3×3 s2 → BN → SiLU            32ch  @160   (stride 2)
 └─ Stage1    ResidualBlock ×2                   32ch  @160
 └─ Stage2    ResidualBlock ×2 (첫 블록 s2)      64ch  @80    (stride 4)
 └─ Stage3    ResidualBlock ×2 (첫 블록 s2)     128ch  @40    (stride 8)   
 └─ Stage4    ResidualBlock ×2 (첫 블록 s2)     256ch  @20    (stride 16) 
 └─ Stage5    ResidualBlock ×2 (첫 블록 s2)     512ch  @10    (stride 32)  
 └─ Head      AdaptiveAvgPool → Flatten → Dropout(0.3) → Linear(512, 500)
```

**ResidualBlock**

```
Input
└─ pipe_x     Conv → BN or nn.Identity()(Input 그대로 넘김)
└─ out        Conv → BN → SiLU
└─ out        Conv → BN
└─ return     pipe_x + out → SiLU 
```

# 설계 결정과 이유

| 결정 | 이유 |
|---|---|
| **MaxPool 제거** | MaxPool은 n*n중 하나만 최댓값을 고르고 추가연산이 들어 주변 픽셀을 가중치를 반영하고 추가연산 없는 stride로 채널 변환 합니다. |
| **SiLU 활성함수** | 음수 영역에서 완전히 0값으로 만들지 않아 죽은 뉴런이 생깁니다. 많은 모델들이 채택한 활성화 함수입니다. |
| **Stage별 채널 32/160→64/80→128/40→256/20→512/10** | 탐지기가 사용하는 스케일 관례에 맞추었습니다. |

# 데이터 파이프라인

- 500 클래스, 약 25만장 train/ 3만장 test / 3만장 val
- `split_image.py`: 클래스별 8:1:1 분할(seed 고정), 320×320 리사이즈 후 PNG로 사전 저장
- `extension_convert.py`: jpg/bmp/webp 혼재 → PNG 통일

# GPU 증강으로 CPU 병목 해결

`transform.py`

torchvision transform은 배치내 처리를 각각 수행하지 못하기 때문에 **CPU가 이미지 변환까지 담당하게되어 GPU활용도가 낮아지는 문제점이 있습니다**

CPU는 **uint8 텐서 변환만 수행하고 증강 단계는 Kornia를 이용하여 GPU 처리하도록** 구성했습니다.

```
class GPUTrainTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = K.AugmentationSequential(
            K.RandomResizedCrop((320, 320), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            K.RandomRotation(degrees=10.0),
            data_keys=["input"],
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.aug(x)
```
- kornia는 배치 내 **샘플별 독립적인 수행**하므로 배치내 다양한 증강이 이루어집니다.
- cpu 스로틀링과 병목을 막기위해 GPU가 처리합니다.

## 학습 설정

`train.py`, `train_val_code.py`

| 항목 | 설정 |
|---|---|
| Optimizer | AdamW (lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.02) |
| Scheduler | CosineAnnealingLR (T_max=150, eta_min=1e-6) |
| Loss | CrossEntropyLoss (weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.1,) |
| Precision | AMP (fp16 autocast + GradScaler) |
| Batch size | 256 |
