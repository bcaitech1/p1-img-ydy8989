# P-Stage 1. 마스크 이미지 분류

## :raising_hand: 작성자

- 윤도연 T1134

## 🏆최종 성적

- `Public LB` :  F1 0.6800 | Accuracy 74.4300%
- `Private LB` :  F1 0.6738 | Accuracy 73.9841% 

## 📚Task Description

마스크를 착용하는 건 COVID-19의 확산을 방지하는데 중요한 역할을 합니다. 제공되는 이 데이터셋은 사람이 마스크를 착용하였는지 판별하는 모델을 학습할 수 있게 해줍니다. 모든 데이터셋은 아시아인 남녀로 구성되어 있고 나이는 20대부터 70대까지 다양하게 분포하고 있습니다.

- ***기간*** : 2021.03.29~2021.04.08(2주)

- ***Data description*** : 

	- 2700명에 대한 이미지(한 명당 7가지의 마스크 착용 모습)

	- `input` : Image shape : 384x512 크기의 마스크를 쓴 사람 이미지

	- `output` : 18개의 클래스(마스크 착용 종류, 성별, 나이의 18가지 조합)

		![image](https://user-images.githubusercontent.com/38639633/121912341-78fd6800-cd6b-11eb-8c23-fd2dc34597d9.png)

- ***Metric :*** F1 Score

	- $F_1 = 2 * \frac{precision*recall}{precision+recall}$
	- $precision = \frac{TP}{TP+FP}, ~~recall = \frac{TP}{TP+FN}$

	

## 🛠️프로젝트 구조

```
p1-img-ydy8989>
├─src
│      dataset.py
│      evaluation.py
│      inference.py
│      loss.py
│      model.py
│      optimizer.py
│      sample_submission.ipynb
│      train.py
│      train_multilabel.py
│
├─winner_code
│      README.md
│      winner_code.ipynb
│
└─wrap_up
│      wrap_up.pdf
├─README.md
└─requirements.txt
```



### File overview

- `src`
	- `dataset.py` : 베이스라인 데이터셋 클래스 두 개(기본 베이스라인 클래스, 사람 기준으로 나누는 데이터셋 클래스)와 multilabel을 위한 custom dataset class  
	- `evaluation.py` : evaluation print
	- `inference.py` : 기본 inference 방식 + 멀티 레이블 방식의 모델을 inference
	- `loss.py` : custom loss를 포함한 여러가지 loss 함수 클래스들`model.py` :
	- `model.py` : model이 구현된 모듈
	- `optimizer.py` : 옵티마이저 클래스 모듈(AdamP, RAdam)
	- `train.py` : single model 학습 실행
	- `train_multilabel.py` : multilabel model 학습 실행
- `winner_code` : Competition winner's code
- `wrap_up` : 대회를 진행하면서 경험했던 내용들에 대한 랩업 리포트입니다.



## 🧬Model pipeline

- Main model : EfficientNet

- EDA : 

	- Label filtering : Imbalanced한 분포를 보이는 데이터를 

	- Augmentations : 

		

## Dependencies

- torch==1.6.0
- torchvision==0.7.0                                                              



## Install Requirements

- `pip install -r requirements.txt`



## Training

```
SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py
```



## Inference

```
SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py
```



## Evaluation

```
SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py
```

