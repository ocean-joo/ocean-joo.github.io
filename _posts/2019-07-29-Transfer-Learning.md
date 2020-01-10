---
title: Transfer Learning
date : 2018-07-29
category: DeepLearning
use_math: true
---

매일 공부한 내용을 블로그에 올려 정리하고 싶었는데 거의 한학기에 포스팅을 하나씩 올리는 것 같다 … 블로그 처음 메인화면도 좀 더 가독성 있게 바꾸려고 (1년전부터) 생각하고 있는데 언제쯤 수정할지,.. 그래도 남은 방학은 적어도 일주일에 한번, 3-4일에 한번씩은 글을 올리려고 한다 !

![image-20190730150405734](https://user-images.githubusercontent.com/40735375/62104728-4e22db80-b2db-11e9-9271-26c39f177601.png)

*이건 진짜로 블로그에 올리겠다는 내 다짐 .., 약속 ,.. *



오늘 정리하려고 하는 내용은 **Transfer Learning** 이다. 최근 semantic segmentation과 object detection과 관련된 논문을 읽으면서 공부를 시작하고 있는데, 공통적으로 등장하는 내용이 무슨 모델(VGG나 ResNet 등)로 pretrained된 weight 값을 가져왔다는 얘기이다. 보통 이미지를 처리하는 모델을 만들 때는 이처럼 기존에 트레이닝 된 weight 값을 가져와 내가 적용하려는 task에 







### Reference

- https://cs231n.github.io/transfer-learning/
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html