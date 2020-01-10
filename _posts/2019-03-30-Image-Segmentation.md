---

title: Image Segmentation
date: 2019-03-30
category: DeepLearning
use_math: true
---

# Image Segmentation

Computer vision에서 중요하게 다루는 task 중에는 Image Classification, Object Detection, Recognition, Localiztion 등 다양한 문제들이 있다. 그리고 이번 포스팅에서는 그 중에서도 내가 제일 관심있는 Image Segementation에 대해서 다루려고 한다. 



![image](https://user-images.githubusercontent.com/40735375/55273347-4d6a4080-530d-11e9-86b2-74c9c34a0c95.png)



위의 사진을 보자. 우리는 자연스럽게 사진의 왼쪽에는 개가 있고, 오른쪽에는 고양이가 있다는 것을 인지할 수 있다. 하지만 컴퓨터에게 단순히 주어진 픽셀들의 3차원 값(RGB)에서 의미적으로 어디에 어떤 물체가 있다고 판단하는 것은 정말 어려운 일이다. 그리고 이러한 문제를 **Image Segmentation**이라고 한다.

![image](https://user-images.githubusercontent.com/40735375/55273376-c79ac500-530d-11e9-8cce-0760a207b29d.png)



스탠포드의 강의 CS231n 중 Image Segmentation과 관련된 강의에서 가져온 사진이다. 분류한 것을 보면 Semantic Segmentation과 Instance Segmentation을 분리한 것을 알 수 있다. Semantic Segmenation의 경우에는 주어진 사진에서 의미적으로 물체의 위치를 찾고, 그 물체가 해당하는 class를 찾는다. 따라서 한 사진 내에 동일 class가 여러개 존재한다고 해도, 이 물체들을 단일의 class로 구분한다. 하지만 Instance Segmentation의 경우에는 동일한 class라고 하더라도 instance 단위로 구분을 한다. 사진을 보면 쉽게 이해할 수 있는데, 제일 오른쪽 사진에서 Instance Segmentation의 경우에는 똑같이 DOG의 class에 속한다고 하더라도 서로 다른 instance이기 때문에 다르게 분류한 것을 확인할 수 있다. 동일한 이미지를 Semantic Segmentation 했다면 두 강아지가 같은 class으로 분류되었을 것이다.



## Image Segmentation Architecture

1. Fully Convolution Network (2014. 11)

   - Semantic Segmentation 분야에서 처음으로 등장한 End-to-End Model

   - data의 input과 output의 크기가 같아야 하는 segmentation task의 특성을 잘 고려

   -  임의의 size의 input에 대해 동일한 크기의 output map을 생성

     

     ![image](https://user-images.githubusercontent.com/40735375/55274814-05084e00-5320-11e9-83e5-aa976d9d8692.png)

   

   

   

2. U - Net (2015.5)

   - Symmetric한 구조의 encoder와 decoder를 사용
   - input을 encoding 할때의 context를 decoding할때 concate해서 사용

![image](https://user-images.githubusercontent.com/40735375/55274826-400a8180-5320-11e9-84f0-2aee5403c9bb.png)



3. SegNet (2016.10)
   - pooling 연산을 진행할 때 pooling한 위치를 기억했다가 dimension을 키울때 사용
   - pixel 단위로 예측을 진행

![image](https://user-images.githubusercontent.com/40735375/55274838-616b6d80-5320-11e9-9351-b50f51ba7cbb.png)





4. Global Convolutional Network (2017.3)
   - Nonlinearity function을 가지지 않는 GCN block을 사용
   - Boundary Refinement를 위해 Residual structure 사용

![image](https://user-images.githubusercontent.com/40735375/55274843-76e09780-5320-11e9-8d08-10f343378743.png)



5. DeepLab v3+ (2018.8)
   - Semantic Segmentation 분야에서 SOTA 성능
   - 기존 DeepLab v3의 아키텍쳐와 Xception을 결합하여 만듦
   - ASPP(Atrous Spatial Pyramid Pooling)과 Depthwise Separable Convolution을 이용해 다양한 크기의 object를 잡아낼 수 있도록 했고, 연산량을 줄임

![image](https://user-images.githubusercontent.com/40735375/55274852-8fe94880-5320-11e9-99da-335ab21ce2fa.png)

