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

<details>
<summary>📁 WORLDMODEL</summary>

<details>
<summary>📁 Memory</summary>

<details>
<summary>📁 m_model</summary>

- `__init__.py`
- `LSTM.py`

</details>

- `__init__.py`
- `encode_data.py`
- `lstm_show_result.py`
- `m_config.py`
- `main.py`
- `SequenceDataset.py`
- `train.py`

</details>

<details>
<summary>📁 Vision</summary>

<details>
<summary>📁 Data</summary>

- `__init__.py`
- `Dataset.py`

</details>

<details>
<summary>📁 v_model</summary>

- `__init__.py`
- `Decoder.py`
- `Encoder.py`
- `VAE.py`

</details>

- `__init__.py`
- `Collect.py`
- `main.py`
- `show_result.py`
- `train.py`
- `v_config.py`

</details>

<details>
<summary>📁 z_data</summary>

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

- `checkpoints/`
- `.gitignore`
- `predicted.png`
- `pyproject.toml`
- `README.md`
- `reconstruction.png`

</details>

## 참고 논문
World Model: https://arxiv.org/pdf/1803.10122
