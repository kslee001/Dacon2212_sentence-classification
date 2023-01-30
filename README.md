# DACON_2210 : 문장 유형 분류 AI 경진대회
- Natural Language Processing  
- Classification  


## 1. 대회 개요
- 주최 : 성균관대학교
- 주관 : 데이콘
- 대회기간 : 2022.12.12 ~ 2022.12.23
- Task
    - **text classification**
    - 문장을 입력으로 받아 문장의 '유형', '시제', '극성', '확실성'을 AI 분류 모델 생성
- Data
    - train data
        - 16540개
    - test data
        - 7090개

- 상금 : 총 500만원
    - 1위 : 200
    - 2위 : 100만원
    - 3~6위 : 50만원

## 2. 대회 결과
- 최종 성적
    - Public  :
        - **Weighted F1 score : 0.74121  ( 49 / 333 )**
            - 1위 : 0.76075
    - Private :
        - **Weighted F1 score : 0.74415  ( 49 / 333 , top 15% )**
            - 1위 : 0.75854

## 3. 진행 과정
### 데이터 전처리
- 숫자, 특수문자 등을 제외한 한글, 영문만을 남기고 중복된 공백 제거
    - lambda x: re.sub('[^a-zA-Zㄱ-ㅎ가-힣]', ' ',x)  
    - replace("  ", " ")  
- 유형 (사실형, 추론형, 예측형, 대화형), 극성(긍정, 부정, 미정), 시제(현재, 과거, 미래), 확실성(확실, 불확실) 등의 multi-label의 multi-class를 0, 1, 2, 3 등의 discrete value로 처리

### 학습
- backbone : klue/roberta-large  
    - max_len = 128  
    - time_len = 16  
    - dim_ff = 256  
    - dropout = 0.35   
- loss function : focal loss  
    - loss = 유형loss + 극성loss + 시제loss + 확실성loss
- optimizer : SAM optimizer  
    - learning rate : 0.05  
    - batch size : 256  
    - epochs : 61  

## 4. Self-feedback?
- 시간이 없는 관계로 대충 참여했는데 나름 괜찮은 성적을 거둔 것 같아 만족함
