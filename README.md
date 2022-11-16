# 🚗 Steering wheel Defect Detection 
### [디지털스마트부산아카데미 1기] GAN 알고리즘을 이용한 Steering whieel 결함검출
<br/>

***

<br/>
<div><h1>📚 Development Environment</h1></div>
<div>
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white">
<img src="https://img.shields.io/badge/YOLO-00FFFF.svg?&style=for-the-badge&logo=YOLO&logoColor=white">
<img src="https://img.shields.io/badge/Google Colab-F9AB00.svg?&style=for-the-badge&logo=Google Colab&logoColor=white">
<img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/GAN-3776AB?style=for-the-badge&logo=Gitpod&logoColor=white">
<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=white">
<img src="https://img.shields.io/badge/XceptionNet-2C3E50?style=for-the-badge&logo=Xamarin&logoColor=white">
<img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=MongoDB&logoColor=white">
  

</div>

<br/>

## 프로젝트 목표
:heavy_check_mark:  DCGAN, ANOGAN을 이용하여 데이터 증강  <br/>
:heavy_check_mark:  Yolov5를 이용하여 결함검출 모델 학습 <br/>
:heavy_check_mark:  학습한 모델을 웹페이지로 제작

* <h2>Dataset</h2>
    - 핸들 버튼 데이터 <br/>
![IMG_8594 (2)](https://user-images.githubusercontent.com/93966720/199412090-76cb557e-deda-4da7-9fc4-7494f1578d21.jpg)

* <h2>TrueData Details</h2>
    - 정상 <br/>
![9](https://user-images.githubusercontent.com/93966720/200234413-7d6619ce-2978-4b52-8ceb-1a8dac2e6024.jpg)
* <h2>FalseData Details</h2>
    - 비정상 <br/>
![f5](https://user-images.githubusercontent.com/93966720/200717582-61cd780c-cf2d-4366-af86-a955ae5cf091.jpg)

* <h2>Result<h2>
    
  
- YOLO Detect  </br>
![yolov5l불량검출이미지 (1)](https://user-images.githubusercontent.com/93966720/202052917-c056bd6d-002b-4924-962e-24b29b5e68c3.jpg)
  
- XceptionNet Detect </br>
![XceptionNet불량검출실패 (1)](https://user-images.githubusercontent.com/93966720/202052957-5b6deddf-26d4-4c4f-9b90-0c1a18999f02.jpg)
  
- DCGAN 생성이미지 </br>
![DCGAN생성이미지 (1)](https://user-images.githubusercontent.com/93966720/202052854-e838bef0-c8fb-4d64-8cbb-ad9840fa0271.jpg)

- Web 구현 </br>
![KakaoTalk_20221116_093534819](https://user-images.githubusercontent.com/93966720/202055315-3928d2bb-56d2-4e50-ab38-134e99c38ac6.jpg)
![KakaoTalk_20221116_093534819_01](https://user-images.githubusercontent.com/93966720/202055360-5563ef53-18c6-4d88-887f-059ff383886a.jpg)

## :pushpin: 참고
*  YOLOv5 https://github.com/ultralytics/yolov5.git
*  AnoGAN, DCGAN  https://github.com/yjucho1/anoGAN.git  
