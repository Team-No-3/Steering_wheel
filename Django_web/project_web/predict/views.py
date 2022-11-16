# html에 응답을 보낸다 return render(request, "predict/index.html")
# py는 주석이 #

#render는 url을 확인하는 역할, httpresponse 객체를 쉽게 변환한다
from django.shortcuts import render
#request->httpresponse->view
from django.http import HttpResponse, FileResponse

import json, os
import numpy as np
from PIL import Image
from django.core.files.base import ContentFile

import torch, cv2, io

label_list = ["o","x"]

#torch.hub.load 모델호출(git 주소, 받는 레포지토리, 모델의 위치), git의 yolov5->custom-> 저장->로드
model = torch.hub.load("ultralytics/yolov5", "custom", path="predict/model/yolov5.pt")

#contentfile은 파일의 이름, 경로, 크기 이런거 없이 정보만 읽는다
def preprocessing(select_model, file):
    file = Image.open(ContentFile(file.read()))
    #8비트 부호없는 정수배열
    #yolov5의 특성을 고려하여 값을 불러오기 위함
    img = np.array(file, dtype='uint8')
    #model에 640 image의 (좌표4개,confidence,클래스,이름) 리스트로 변환하여 
    #이를 nms거쳐 plot_boxes로 리턴 -> iou값과 이미지좌표를 리턴
    results = (model(img, size=640).pandas().xyxy[-1]).values.tolist()
    results = nms(results)
    img = plot_boxes(results, img)
    
    return img

#box=(x1,y1,x2,y2)
#nms 연산량 감소와 mAP향상 목적, non-maximum suppression, 그냥 iou, 1일수록 검출성공
#intersection=box의 겹치는 영역, overlap=넓이, union=(box1넓이+box2넓이)-겹친넓이=
def nms(boxes):
    def iou(box1, box2):
        intersection_x_length = min(box1[2], box2[2]) - max(box1[0], box2[0])
        intersection_y_length = min(box1[3], box2[3]) - max(box1[1], box2[1])
        overlap = intersection_x_length * intersection_y_length
        union = ((box1[2] - box1[0]) * (box1[3] - box1[1])) + ((box2[2] - box2[0]) * (box2[3] - box2[1])) - overlap
        return overlap / union

    #박스들을 호출하여 iou > 0.85 에서 confidence가 적은것을 삭제할 리스트에 넣는다
    remove_index = []
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j: continue
            if j in remove_index:
                continue

            calc = iou( (int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])),
                        (int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2]), int(boxes[j][3])) )
            if calc > 0.85:
                if boxes[i][4] > boxes[j][4]:
                    remove_index.append(j)
                else:
                    remove_index.append(i)
                    break
    
    #중복을 제거하고 삭제하고 남은 것을 리턴
    for i in set(remove_index):
        del boxes[i]
    return boxes


#plot_boxes(넣은 이미지, 그리는 박스)
def plot_boxes(predicts, frame):
    for row in predicts:
        #confidence>=0.1
        if row[4] >= 0.1:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            #맞으면 녹색
            rgb = (0, 255, 0)
            #아니면 빨강, row[5]는 클래스, [1]-> 이름'x' 
            if row[5] == 1:
                rgb = (0, 0, 255)


            #cv2.rectangle(이미지,왼상,우하,컬러,선두께)
            #cv2.putText(이미지, 이름, 문자시작위치, 폰트, 크기, 색상, 두께)
            cv2.rectangle(frame, (x1, y1), (x2, y2), rgb, 2)
            cv2.putText(frame, row[6], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rgb, 2)
    return frame

# index.html로 여기선 기능이 없음, request는 사용자에 의해 입력받은 인자
def index(request):
    return render(request, "predict/index.html")


#request.post.get('select_model') 변수를 받으면 리턴, 없으면 none
#request.files['file']는 'file'이라는 이름으로 파일불러옴
#model과 파일을 가지고옴
def api_predict(request):
    select_model = request.POST.get("select_model")
    file = request.FILES["file"]
    res = preprocessing(select_model, file)

    #imencode binary형태로 이미지를 읽는다(압축), (확장자, 이미지)
    #imencode return:(retval-yes/false,buf-인코딩이미지)
    #tostring-java,문자열로리턴,
    imencoded = cv2.imencode(".jpg", res)[1]
    return HttpResponse(imencoded.tostring(), content_type="image/jpeg")

