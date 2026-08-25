# WorldModel

## 프로젝트 개요

- 최근 VLA 중에서 World Model이라는 모델이 뜨고있음. 현재의 상태와 환경을 보고 다음 환경을 예측하고 다음 액션을 도출하는 모델임. 단순히 행동만 예측하는것이 아닌 모든 물리법칙을 고려하여 환경까지 예측함. 즉 현재 환경의 상태와 다음 환경의 상태를 고려한 다음 행동을 예측하는것.
- 모델은 V(Vision), M(Memory,Prediction), C(Control) 세 가지 부분으로 나뉨.
- 최종적으론 3가지 모델을 합쳐서 상황에 대한 다음 상황을 예측하고 다음 행동을 결정하는 모델.
---

- V(Vision): high dimension observation을 low dimension latent vector로 인코딩(z). 출력은 이미지를 latent dim z값으로 압축한 값. 
- M(memory,Prediction): 과거의 요소들로 미래의 환경을 예측하는 모델. 입력은 V에서 압축한 z값을 받고, 출력은 예측한 z값. 
- C(control): V,M을 통해 좋은 액션을 선택하는 작은 모델. 해당 논문에선 Linear레이어 하나를 사용함. **본 프로젝트는 일단 V,M 까지 설계하는 것을 목표로 함.**


## 코드 설명

</details>
<summary>📁 figures</summary>

모델 loss 그래프와 VAE 재구성 시각화 이미지, predicted 이미지 등.

<details>

<details>
<summary>📁 Memory</summary>

LSTM 기반 M 모듈. 인코딩한 z값 시퀀스를 학습하여 다음 환경을 예측함.

<details>
<summary>📁 m_model</summary>

- `__init__.py`
- `LSTM.py` - 레이어는 깊게 쌓지 않았고 nn.LSTM을 사용함 input_size=(latent_dim=48 + action=2)

</details>

- `__init__.py`
- `encode_data.py` - 시퀀스 데이터를 z값으로 인코딩 하는 코드 CNN모델을 사용
- `lstm_show_result.py` - LSTM이 예측한 z값을 이미지로 시각화. 디코더를 사용
- `m_config.py` - M모듈 전용 config 파일
- `main.py` - 모델 학습 main 코드
- `SequenceDataset.py` - LSTM 학습용 시퀀스 데이터 사용을 위한 데이터 생성 코드
- `ShowDataset.py` - 시각화용 원본 이미지 시퀀스 데이터 생성 코드
- `train.py` - train 코드

</details>

<details>
<summary>📁 Vision</summary>

CNN 기반 V 모듈. 이미지를 z값으로 인코딩

<details>
<summary>📁 Data</summary>

- `__init__.py`
- `Dataset.py` - 데이터셋 생성 코드

</details>

<details>
<summary>📁 old_structure</summary>

U-Net 이전 구조(skip connection 없음). LSTM이 예측한 z값을 시각화할 때, U-Net 디코더는 skip connection이 필수라 사용할 수 없어 대신 사용

- `old_Decoder.py` - skip connection 없이 z값만으로 이미지 복원
- `old_Encoder.py` - skip connection 없이 z값만 출력
- `old_main.py` - old 구조 학습 main 코드
- `old_VAE.py` - old_Encoder, old_Decoder를 합친 VAE 모델

</details>

<details>
<summary>📁 v_model</summary>

V 모듈에서 VAE를 학습을 시킴.

- `__init__.py`
- `Decoder.py` - 인코더에서 인코딩한 z값을 transpose로 역변환 시켜 원래 차원으로 늘림
- `Encoder.py` - CNN 기반 latent_dim=48 로 설정하여 z값 인코딩
- `VAE.py` - 인코더 디코더를 합친 VAE 모델 이미지를 재구축함

</details>

- `__init__.py`
- `Collect.py` - 기본 데이터 수집 코드
- `main.py` -
- `show_result.py` - 디코더가 재구축한 z값을 이미지로 시각화
- `train.py` -
- `v_config.py` - V 모듈에서 사용한 config

</details>

</details>

- `pyproject.toml`
-`README.md`

<details>

## 개선점
기존 CNN기반의 VAE 모델은 train loss가 val loss보다 높고 train loss가 16.xxx 대로 완전히 학습되지 않는 문제가 있었음.
이를 해결하고자 U-net 구조의 skip층을 사용하여 encoder 레이어 마다 출력값을 decoder의 레이어에 concat 해주고 입력으로 사용함.
그리고 latent dim을 48에서 24로 대폭 축소 시킴. 너무 큰 잠재 공간은 PushT 같은 단순한 이미지와 문제에서 너무 낭비되고 복잡한 결과를 낼거라고 생각.

### 기존 CNN 기반 VAE 모델 summary
![old_model_structure](https://github.com/user-attachments/assets/4421adf1-4893-43a8-ba01-c4c40893b8ca)

### U-net 기반 VAE 모델 summary
![unet_structure](https://github.com/user-attachments/assets/710267aa-f704-4ec2-ad58-c64c31f23368)
- 기존 CNN과 달리 각 층을 block으로 나눠 정의하고 각 층의 출력을 반환하여 decoder에 concat하여 입력으로 들어감


## 성능
![reconsstructure](https://github.com/user-attachments/assets/41b92148-bc43-4a93-b6dd-d11275b3c7c3)
- 오리지널 이미지와 VAE로 복원한 이미지를 시각화

![predicted](https://github.com/user-attachments/assets/97363bc6-a1f0-4ebc-9d26-3b2862ef2746)
- LSTM으로 예측한 z값을 Decoder로 복원한 시각화 이미지

- 학습 성능표

Epoch 138/1000 - Train: 3.6281, Val: 3.3629, LR: 0.000001
  → 베스트 모델 갱신 (val loss: 3.3629)
-> unet 구조 학습결과

| Model | epochs | train loss | val loss | 
|-------|--------|------------|----------|
| VAE | 95 | 15.0893 | 6.6171 |
| LSTM | 497 | 15.9591 | 18.1231 |

## 참고 논문
World Model: https://arxiv.org/pdf/1803.10122
