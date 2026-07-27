# WorldModel

## 프로젝트 개요

- 최근 VLA 중에서 World Model이라는 모델이 뜨고있음. 개념은 어떤 행동을 통해 다음 행동을 예측한다는 것. 단순히 행동만 예측하는것이 아닌 모든 물리법칙을 고려하여 환경까지 예측함. 즉 환경의 상태를 고려한 행동을 예측하는것.
- 모델은 V(Vision), M(Memory,Prediction), C(Control) 세 가지 부분으로 나뉨.

---

- V(Vision): 말 그대로 Vision encoder를 사용해서 영상을 프레임 단위로 쪼개서 벡터형태로 더한 후 M으로 넘어감. 여기에선 CNN 모델을 사용할 예정.
- M(memory,Prediction): 이 부분은 기억을 하고 예측을 하는 단계. 이전 행동들을 기억하는 능력이 필요하고 이를 토대로 다음 상황을 예측해야함. 여기에선 RNN이나 Transformer를 사용할 예정.
- C(control): 이 부분은 논문에선 간단한 선형 linear 모델을 사용함. 행동을 뽑아내는 부분.

---
## 참고 논문
World Model: https://arxiv.org/pdf/1803.10122
