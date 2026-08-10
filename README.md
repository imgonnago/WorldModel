# WorldModel

## 프로젝트 개요

- 최근 VLA 중에서 World Model이라는 모델이 뜨고있음. 현재의 상태와 환경을 보고 다음 환경을 예측하고 다음 액션을 도출하는 모델임. 단순히 행동만 예측하는것이 아닌 모든 물리법칙을 고려하여 환경까지 예측함. 즉 환경의 상태를 고려한 행동을 예측하는것.
- 모델은 V(Vision), M(Memory,Prediction), C(Control) 세 가지 부분으로 나뉨.
- 최종적으론 3가지 모델을 합쳐서 상황에 대한 다음 상황을 예측하고 다음 행동을 결정하는 모델.
---

- V(Vision): high dimension observation을 low dimension latant vector로 인코딩(z). 출력은 이미지를 latant dim z값으로 압축한 값. 
- M(memory,Prediction): 과거의 요소들로 미래의 환경을 예측하는 모델. 입력은 V에서 압축한 z값을 받고, 출력은 예측한 z값. 
- C(control): V,M을 통해 좋은 액션을 선택하는 작은 모델. 해당 논문에선 Linear레이어 하나를 사용함. 본 프로젝트는 일단 V,M 까지 설계하는 것을 목표로 함.

---
## 코드 설명

```
WorldModel/
├── Memory/                     # M 모듈 (LSTM 기반 월드 모델)
│   ├── m_model/
│   │   ├── __init__.py
│   │   └── LSTM.py             # LSTM 모델 정의
│   ├── __init__.py
│   ├── encode_data.py          # 잠재 벡터 인코딩
│   ├── lstm_show_result.py     # 예측 결과 시각화
│   ├── m_config.py             # M 모듈 하이퍼파라미터
│   ├── main.py
│   ├── SequenceDataset.py      # 시퀀스 데이터셋
│   └── train.py                # M 모듈 학습
│
├── Vision/                     # V 모듈 (VAE 기반 시각 인코더)
│   ├── Data/
│   │   ├── __init__.py
│   │   └── Dataset.py          # 데이터셋 정의
│   ├── v_model/
│   │   ├── __init__.py
│   │   ├── Decoder.py          # VAE 디코더
│   │   ├── Encoder.py          # VAE 인코더
│   │   └── VAE.py              # VAE 전체 모델
│   ├── __init__.py
│   ├── Collect.py              # 데이터 수집
│   ├── main.py
│   ├── show_result.py          # 재구성 결과 시각화
│   ├── train.py                # V 모듈 학습
│   └── v_config.py             # V 모듈 하이퍼파라미터
│
├── z_data/                     # VAE 인코딩된 잠재 벡터 데이터
│   ├── train/
│   │   ├── next_z.npy
│   │   └── z.npy
│   └── val/
│       ├── next_z.npy
│       └── z.npy
│
├── .gitignore
├── predicted.png               # M 모듈 예측 결과 샘플
├── pyproject.toml
├── README.md
└── reconstruction.png          # V 모듈 재구성 결과 샘플
```
## 참고 논문
World Model: https://arxiv.org/pdf/1803.10122
