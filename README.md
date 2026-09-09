# massive_pretrained_cnn_model_v1

객체 탐지의 백본을로 사용하기 위해 CNN을 PyTorch로 직접 설계하고, **500-class 커스텀 ImageNet 데이터셋(약 26만장)으로 처음부터(from scratch) 사전학습**한 CNN 모델입니다.

torchvision의 pretrained weight 또는 외부 모델을 사용하는 대신 백본 구조 설계 → 데이터 파이프라인 → 학습 까지 과정을 구현했습니다.
