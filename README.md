# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`

### File overview

- `Dataset.py` : 
	- 베이스라인 데이터셋 클래스 두 개(기본 베이스라인 클래스, 사람 기준으로 나누는 데이터셋 클래스)와 multilabel을 위한 custom dataset class
- `inference.py` : 
	- `inference()` : 기본 inference 방식
	- `multilabel_inference() `: 멀티 레이블 방식의 모델을 inference
- `loss.py`:
	- custom loss를 포함한 여러가지 loss 함수 클래스들
- `model.py` :
	- 모델 아키텍쳐들의 모임
- `optimizer.py` ;
	- AdamP와 RAdam을 사용하기 위한 파일
- `train.py` and `train_multilabel.py` :
	- Single model과 multi label(branch) model을 각각 실행하는 파일

