# massive_pretrained_cnn_model_v1

객체 탐지의 백본을로 사용하기 위해 CNN을 PyTorch로 직접 설계하고, **500-class 커스텀 ImageNet 데이터셋(약 26만장)으로 처음부터(from scratch) 사전학습**한 CNN 모델입니다.

torchvision의 pretrained weight 또는 외부 모델을 사용하는 대신 백본 구조 설계 → 데이터 파이프라인 → 학습 까지 과정을 구현했습니다.

# 사전학습 모델을 만든 이유

- 객체 탐지에서 **warm start** 시작하여 초기화된 상태부터 시작하지않아 학습 속도가 높아집니다.
- 백본 내부 구조에 대한 설계 결정을 직접 작성하고 CNN이 실제로 무엇을 학습하는지 확인해 볼 수 있습니다.

## 아키텍처

`backbone.py`

```
Input 3×320×320
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
ResidualBlock(in_ch, out_ch, stride=s)

  Input                                 in_ch  @ H×W
    │
    ├──── Main ──────────────────────────────────────┐
    │     Conv3×3  s=s  p=1  bias=False              │
    │     BN → SiLU                     out_ch @ H/s │
    │     Conv3×3  s=1  p=1  bias=False              │
    │     BN                            out_ch @ H/s │
    │                                                │
    └──── Shortcut ───────────────────────────────┐  │
          [s≠1 or in_ch≠out_ch]                   │  │
            Conv1×1  s=s  bias=False → BN         │  │
          [else]                                  │  │
            Identity                 out_ch @ H/s │  │
                                                  ▼  ▼
                                              Add → SiLU
                                              out_ch @ H/s×W/s
```
