# WorldModel

## 프로젝트 개요

- 최근 VLA 중에서 World Model이라는 모델이 뜨고있음. 현재의 상태와 환경을 보고 다음 환경을 예측하고 다음 액션을 도출하는 모델임. 단순히 행동만 예측하는것이 아닌 모든 물리법칙을 고려하여 환경까지 예측함. 즉 환경의 상태를 고려한 행동을 예측하는것.
- 모델은 V(Vision), M(Memory,Prediction), C(Control) 세 가지 부분으로 나뉨.
- 최종적으론 3가지 모델을 합쳐서 상황에 대한 다음 상황을 예측하고 다음 행동을 결정하는 모델.
---

- V(Vision): high dimension observation을 low dimension latent vector로 인코딩(z). 출력은 이미지를 latent dim z값으로 압축한 값. 
- M(memory,Prediction): 과거의 요소들로 미래의 환경을 예측하는 모델. 입력은 V에서 압축한 z값을 받고, 출력은 예측한 z값. 
- C(control): V,M을 통해 좋은 액션을 선택하는 작은 모델. 해당 논문에선 Linear레이어 하나를 사용함. 본 프로젝트는 일단 V,M 까지 설계하는 것을 목표로 함.

---
## 코드 설명

<details>
<summary>📁 WORLDMODEL</summary>

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

<details>
<summary>📁 z_data</summary>
시퀀스 데이터로 인코딩한 z 값 데이터

<details>
<summary>📁 train</summary>

- `next_z.npy`
- `z.npy`

</details>

<details>
<summary>📁 val</summary>

- `next_z.npy`
- `z.npy`

</details>

</details>

- `predicted.png`
- `pyproject.toml`
- `README.md`
- `reconstruction.png`

</details>

## 참고 논문
World Model: https://arxiv.org/pdf/1803.10122
