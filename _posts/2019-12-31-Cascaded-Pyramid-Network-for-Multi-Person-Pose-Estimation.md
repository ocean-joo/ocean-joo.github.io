---
title: Cascaded Pyramid Network for Multi-Person Pose Estimation
date : 2019-12-31
category: PoseEstimation
tags: PoseEstimation
use_math: true
---
  
Human pose estimation과 관련된 논문을 하나하나 정리하고 있는데 Multi human pose estimation 중에서는 제일 먼저 읽었던 Cascaded pyramid network, CPN이다.  
  
  
  
## 1. Introduction  

Human pose estimation과 관련된 연구는 계속 진행중이지만, 키포인트들끼리 겹쳐있다던가, 다른 object에 가려서 보이지 않는 키포인트라던가, 뒷 배경이 복잡하게 생겨서 object의 정확한 검출에 악영향을 미치는 경우가 꽤 많이 존재한다.  

이 논문에서는 이러한 한계를 극복하고 분류하기 어려운 **hard keypoints**를 잘 검출하기 위해서 두 개의 stage로 이루어진 CPN을 제안한다. *(GlobalNet + RefineNet)*  

GlobalNet은 간단한 키포인트는 잘 localize 시키지만, 뭉쳐있거나 보이지 않는 키포인트들은 틀릴 수도 있는 feature pyramid network이다. 말 그대로 global한 특징을 잡아서 localize를 시키는 네트워크라고 생각하면 될 것 같다.  

RefineNet은 GlobalNet에서 제대로 localize시키지 못한 **online hard keypoint mining loss**을 이용해서 hard keypoints를 localize 시키는 네트워크이다.  

최종적으로 논문에서 말하는 contribution은 총 3가지 이다.  
- GlobalNet과 RefineNet으로 이루어진 새로운 CPN이라는 아키텍쳐를 제안  
- top-down pipeline 기반의 multi-person pose estimation에 영향을 주는 다양한 factor 조사  
  - (top-down approach와 bottom-up approach가 있다.)  
- COCO multi-person keypoint benchmark에서 SOTA 찍음  
  
  
  
  
## 2. Cascaded Pyramid Network  
  
<img width="1099" alt="image" src="https://user-images.githubusercontent.com/40735375/71621114-d4d43f80-2c10-11ea-93f6-ee753f8353b5.png">  

<img width="1063" alt="image" src="https://user-images.githubusercontent.com/40735375/71622326-609d9a00-2c18-11ea-85e2-486b943746f4.png">  
  
  
여기에서 제안하는 CPN은 top-down 기반의 아키텍처이다. Top-down pipeline이라는 것은 먼저 detector가 주어진 이미지에서 사람이라고 판단되는 bounding box를 뽑아내고, 이 각 bounding box마다 key point를 localize 시키는 방법이다.  
  
  
### 	1) Human Detector  
여기에서 사용한 base object detector는 FPN(Feature Pyramid Network)인데, 여기에서 RoI pooling을 Mask R-CNN의 RoI Align으로 바꾸었다. 그리고 학습할 때는 COCO dataset을 이용했다.  
  
  
### 	2) GlobalNet  
먼저 GlobalNet은 ResNet 기반의 아키텍처이다. Last residual blocks의 output을 다양한 resolution에서의 feature map이라고 생각했을 때, 각 feature map에 3x3 convolution filter 적용해 keypoints의 heat map을 얻을 수 있었다. 그런데 FPN에서 발생했던 이슈와 마찬가지로, 이런식으로 하면 낮은 resolution의 feature map과 높은 resolution의 feature map에서 trade off가 발생한다. (high semantic information이냐 high resolution이냐)  

그래서 여기에서도 U모양의 구조를 도입해서 upsampling할때 1x1 convolution으로 채널 수 맞춰주고, element-wise로 더하는 과정을 추가적으로 더해주었다. (FPN이랑 완전 똑같다.)  

여기까지 설명하면 상단의 Figure들에서 GlobalNet 부분은 대강 이해할 수 있을 것 같다.  
  
  
### 	3) RefineNet  
하지만 논문에서는 이거 말고도 아키텍쳐 뒤에 RefineNet을 붙인다. 아까 introduction에서 이야기했던 hard keypoint를 잘 잡아내기 위함인데, 아까 다양한 resolution에서 뽑아냈던 feature map을 기반으로 key point를 좀 더 정교하게 localize한다.  

RefineNet에서는 단순히 모든 pyramid feature를 다 concat한다. 학습이 진행되면서 네트웤크는 simple keypoint에 더 집중하지 hard keypoint를 무시하는 경향을 보이는데, 네트워크가 두 종류의 keypoint를 다 잘 학습할 수 있도록 online hard keypoint mining을 이용해 hard keypoints를 골라주고 걔네들만 이용해서 학습을 진행했다고 한다.  
  
  
  


## 3. Experiment  
### 1) Training detail  
- Human detect를 할 때 height과 width를 정해진 비율(256:192)로 잘랐고, 이미지를 늘이거나 줄이거나 하는 왜곡은 하지 않았다고 한다.  
- Data augmentation할 때 random flip, random rotation, random scaling 전부 다 했다.  
- Data 쪽만 detail을 가져온 이유는 내가 pose estimation에서 데이터 전처리를 어떻게 해야 좋을지에 대해서 궁금했기 때문이다. ㅎㅁㅎ
- Human Detector에서 NMS를 사용했다.  
  - 성능에 유의미한 차이 O  
  
  
### 2) Experiment result  
  
<img width="611" alt="image" src="https://user-images.githubusercontent.com/40735375/71622671-cdb22f00-2c1a-11ea-80ba-a1d6f5fc1da5.png">  
  
여기에서 baseline 으로 잡았던 hourglass나 ResNet + dilation보다 더 향상된 성능을 보인다. 다만 2-stage hourglass와 비교해 보았을 때 AP는 3-4정도 차이가 나는데 parameter size가 거의 다섯배가 차이가 나서 성능은 좋지만 그래도 좀 더 가볍게 할 수 있는 방법이 있으면 좋을 것 같다는 생각을 했다.  
  
  
그리고 아키텍처에서 조절할 수 있는 것들을 가지고 실험한 다양한 표들이 있는데, 이 부분은 간단하게 설명하겠다.  
- Human Detector  
    -   다양한 Human Detector를 사용했을 때 성능 차이 (앙상블이 제일 좋음)  
    -   다양한 NMS threshold 사용했을 때 성능 차이 (Soft NMS가 제일 좋음)  

- RefineNet  
  - concatenate operation이 바로 진행되는지 아닌지  
  - bottleneck block이 각 레이어 뒤에 하나만 붙는지 아닌지  
  - various bottleneck number  
  

논문에서 중점적으로 제안하는 Online Hard Keypoints Mining의 실험결과는 다음과 같다.  
<img width="599" alt="image" src="https://user-images.githubusercontent.com/40735375/71654429-24338080-2d75-11ea-802c-1656e46f46aa.png">  
  
<img width="596" alt="image" src="https://user-images.githubusercontent.com/40735375/71654460-4a592080-2d75-11ea-9b00-0aa028c13cac.png">  

개인적으로는 RefineNet에 OHEM을 적용하더라도 AP가 1이 조금 안되게 올라서 성능에 그렇게 유의미한 차이가 있다고 할 수 있나? 라는 생각이 들었다. 뭔가 데이터별로 예측이 잘 되는 데이터, 예측이 잘 되지 않는 데이터를 나누어서 정량적으로 RefineNet이 hard keypoints를 잘 예측한다는 내용을 볼 수 없어서 아쉬웠다.  
  
그리고 개인적으로 재밌었던 부분은 GlobalNet에 OHEM을 적용하면 전체 성능이 떨어졌는데, GlobalNet이 해야할 general information을 잡아내야하는 역할을 제대로 수행하지 못해서 그런 것 같았다!!  
  
  
  
## 4. Conclusion  
이 논문에서는 hard keypoints를 잘 찾기 위한 top-down pipeline의 Cascaded Pyramid Network를 제안한다. 이 모델은 GlobalNet과 RefineNet으로 나누어져 있는데, RefineNet에서는 명시적으로 online hard keypoint mining을 이용해 hard keypoints를 중심으로 학습한다.  
  
  
  
  
(+ 원래 이거 읽기는 크리스마스 이브에 읽었었는데 이거 읽다가 FPN 모르겠어서 FPN 읽고오고 online hard keypoint mining loss 모르겠어서 online hard keypoint mining loss 논문 읽고 오느라 이제 올린다 ...^^ 배움의 세계란^^ )  
  
  
### Reference  
  
- https://arxiv.org/abs/1711.07319  