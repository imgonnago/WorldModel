# WorldModel

## 프로젝트 개요

최근 VLA 중에서 World Model이라는 모델이 뜨고있음. 현재의 상태와 환경을 보고 다음 환경을 예측하고 다음 액션을 도출하는 모델임. 단순히 행동만 예측하는것이 아닌 모든 물리법칙을 고려하여 환경까지 예측함. 즉 현재 환경의 상태와 다음 환경의 상태를 고려한 다음 행동을 예측하는것.
모델은 V(Vision), M(Memory,Prediction), C(Control) 세 가지 부분으로 나뉨.
최종적으론 3가지 모델을 합쳐서 상황에 대한 다음 상황을 예측하고 다음 행동을 결정하는 모델.

---

- V(Vision): high dimension observation을 low dimension latent vector로 인코딩(z). 출력은 이미지를 latent dim z값으로 압축한 값. 
- M(memory,Prediction): 과거의 요소들로 미래의 환경을 예측하는 모델. 입력은 V에서 압축한 z값을 받고, 출력은 예측한 z값. 
- C(control): V,M을 통해 좋은 액션을 선택하는 작은 모델. 해당 논문에선 Linear레이어 하나를 사용함. **본 프로젝트는 일단 V,M 까지 설계하는 것을 목표로 함.**


## 코드 설명

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
U-Net(skip connection) 적용 이전의 VAE 구조. M의 예측 결과를 시각화할 때, skip connection 없이 z만으로 이미지를 복원해야 하므로 별도로 학습하여 사용함

- `old_Decoder.py` - skip connection 없이 z만으로 이미지를 복원하는 디코더
- `old_Encoder.py` - skip connection 없는 인코더
- `old_main.py` - old 구조 학습 main 코드
- `old_VAE.py` - old 구조 인코더/디코더를 합친 VAE

</details>

<details>
<summary>📁 v_model</summary>
V 모듈에서 VAE를 학습을 시킴. U-Net 구조(skip connection)를 적용해 재구성 품질을 개선함

- `__init__.py`
- `Decoder.py` - 인코더에서 인코딩한 z값을 transpose로 역변환 시켜 원래 차원으로 늘림. 인코더의 conv1,2,3 값을 skip connection으로 받아 공간 정보를 보존함
- `Encoder.py` - CNN 기반 latent_dim=48 로 설정하여 z값 인코딩. 각 conv 블록의 출력을 skip connection용으로 함께 반환함
- `VAE.py` - 인코더 디코더를 합친 VAE 모델 이미지를 재구축함

</details>

- `__init__.py`
- `Collect.py` - 기본 데이터 수집 코드
- `main.py` - 모델 학습 main 코드
- `show_result.py` - 디코더가 재구축한 z값을 이미지로 시각화
- `train.py` - train 코드
- `v_config.py` - V 모듈에서 사용한 config

</details>

<details>
<summary>📁 figures</summary>
학습 loss 그래프, 재구성/예측 결과 시각화 이미지 저장 폴더

</details>

- `pyproject.toml` - 프로젝트 패키지 설치 설정 (pip install -e .)
- `README.md`


## 1차 실험 

### 기본 CNN 기반 VAE 모델 summary
![old_model_structure](https://github.com/user-attachments/assets/4421adf1-4893-43a8-ba01-c4c40893b8ca)

---

### LSTM 모델 summary
![LSTM Model](https://github.com/user-attachments/assets/39680469-1328-4e33-a62c-65af5f4b2d47)
> 너무 단순한 환경과 이미지를 예측하기 때문에 간단하게 레이어 1개로 구성.

---

### 1차 실험 결과 
![reconsstructure](https://github.com/user-attachments/assets/41b92148-bc43-4a93-b6dd-d11275b3c7c3)
> 기본 CNN 기반 VAE 모델로 오리지널 이미지와 복원한 이미지 시각화

![predicted](https://github.com/user-attachments/assets/b73b5382-7be9-4b23-b77a-4e6d7730a137)
![predicted1](https://github.com/user-attachments/assets/140cb37b-4c12-48fa-bf93-88393fc8007d)
> LSTM으로 예측한 z값을 Decoder로 복원한 시각화 이미지



- 학습 성능표(1차 실험에서 loss 그래프를 그리지 않음)

| Model | epochs | train loss | val loss | 
|-------|--------|------------|----------|
| VAE | 95 | 15.0893 | 6.6171 |
| LSTM | 497 | 15.9591 | 18.1231 |

---

### 1차 실험 결과 해석

기본 CNN 구조로 train을 했을 때 train loss가 val loss 보다 높고 train loss가 16.xxx 대로 제대로 학습되지 않는 문제가 있음.
기본 CNN 구조로는 이미지를 **z값으로 잘 압축하지 못함.**
이미지에서 에이전트가 없거나 회색 T가 없어지거나 흐려지는 현상이 많았음.
CNN 구조에서 low, mid, high level의 특징을 잘 표현하지 못했고, 단순한 이미지에 비해 latent dim이 커서 제대로 잠재 공간을 표현하지 못함.

-> latent dim의 크기를 줄이고, 학습 단계에서 train이 더 잘되도록 U-net 구조를 활용하여 encoder에서 decoder까지 conv레이어의 값이 잘 전달되도록 개선하는 방향으로 2차 실험 구성

---

## 2차 실험 

**U-net 구조의 skip층**을 사용하여 encoder 레이어 마다 출력값을 decoder의 레이어에 concat 해주고 입력으로 사용함.
그리고 **latent dim을 48에서 24**로 대폭 축소 시킴. 너무 큰 잠재 공간은 PushT 같은 단순한 이미지와 문제에서 넓은 공간을 최적화 하는 과정이 더 어려울 것이라 생각.
LSTM 모델은 1차 실험의 모델 구조를 그대로 사용함.

---

### U-net 기반 VAE 모델 summary
![unet_structure](https://github.com/user-attachments/assets/0f2c3110-5009-467d-a485-d7572508255f)
> 기존 CNN과 달리 각 층을 block으로 나눠 정의하고 각 층의 출력을 반환하여 decoder에 concat하여 입력으로 들어감

---

### 2차 실험 결과

- 학습 성능표

| Model | epochs | train loss | val loss | 
|-------|--------|------------|----------|
| U-net VAE | 138 | 3.4043 | 3.1640 |
| LSTM | 116 | 0.0003 | 0.0001 |

- U-net VAE Loss

![unet vae loss](https://github.com/user-attachments/assets/72cba224-db90-4bdd-b764-f8b8725268aa)

- LSTM Loss trained with U-net VAE
  
![LSTM with U-net](https://github.com/user-attachments/assets/996418ef-993f-405b-9277-3dbe0537562c)

- U-net 구조 기반 VAE 모델로 오리지널 이미지와 복원한 이미지 시각화
  
![unet reconstruction](https://github.com/user-attachments/assets/b820e3c6-71fc-45c2-b656-cbb82bf30d49)

> 흐린 부분도 있지만 전체적으로 양호하게 복원됨.

- LSTM으로 예측한 z값을 Decoder로 복원한 시각화 이미지
  
![LSTM with unet](https://github.com/user-attachments/assets/109ee265-2bbd-4745-ab93-04de88afc9e2)

> 전체적으로 양호하게 예측함. 

---

### 2차 실험 결과 해석

1차 실험에서 문제 되었던 train loss가 크게 떠있는 현상을 U-net 구조의 skip층을 통해 encoder에 각 레이어의 값을 decoder에 전달하여 학습이 잘 흐르도록 함.
결과적으로 각 conv층의 값을 decoder에서 받아 decoder가 encoder가 뽑아둔 level들을 재사용하여 loss를 줄이는데 수월했고, gradient가 보다 짧은 경로로 전달되어 소실이 줄어들었을 것이라 해석. 
latent dim을 줄인 효과로는 현재 실험만으로 정확히 설명하기 어렵지만, 단순한 이미지에 적당한 공간을 할당하여 최적화 하는데 수월했을 것이라고 해석됨.

## 향후 개선점

1. LSTM을 학습 할 때 VAE를 통해 z값 데이터를 만들어 사용했는데, 이 때 logvar는 사용하지 않고 mu만 사용한 노이즈 없는 데이터로 학습을 시킴. 
이로 인해 LSTM모델이 노이즈 없이 항상 정답인 값만 학습을 하였고, 그럼에도 M모듈의 예측은 노이즈가 있던 시각화 과정에서도 강건했지만, 이는 분포적 의미를 잃어버렸다고 할 수 있음. 
향후 LSTM 모델이 노이즈를 같이 학습 했을 때의 성능과 비교해보면 좋을 것.

2. 현재 시간과 프로젝트 난이도를 위해 Controller 모듈은 만들지 않았음. 향후 프로젝트에서는 C모듈을 만들어서 논문에서 사용한 CMA-ES 알고리즘이나 강화학습 알고리즘을 이용하여 학습 시키고 성능을 확인하면 좋을 것.

3. 모델이 같은 환경에서 다른 액션이 나왔을 때도 다양한 경우의 수로 커버를 할 수 있는지 실험해보면 좋을 것. 즉 모델이 융통성을 확인.

## 참고 논문
World Model: https://arxiv.org/pdf/1803.10122

U-net: https://arxiv.org/abs/1505.04597
