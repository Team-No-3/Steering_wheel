import os
import sys

real_path = os.path.dirname(os.path.realpath(__file__))
sub_path = os.path.split(real_path)[0]
os.chdir(sub_path)

from flask import Flask, Response, escape, g, make_response, request
from flask.templating import render_template
from werkzeug.utils import secure_filename

from pymongo import MongoClient
import cv2, pandas, time
import numpy as np
import yolov5
import base64

conn = MongoClient('127.0.0.1', port=27017)
db = conn.upload_log    # db 선택
collect = db.data       # table 선택

app = Flask(__name__)
app.debug = True

def root_path():
	'''root 경로 유지'''
	real_path = os.path.dirname(os.path.realpath(__file__))
	sub_path = "\\".join(real_path.split("\\")[:-1])
	return os.chdir(sub_path)

''' Main page '''
@app.route('/')
def index():
	return render_template('index.html')

''' AnoGAN input page '''
@app.route('/ano_get')
def ano_get():
	return render_template('ano_get.html')


# @app.route('/ano_post', methods=['GET','POST'])
# def ano_post():
# 	if request.method == 'POST':
# 		root_path()

# 		# User Image (target image)
# 		user_img = request.files['user_img']
# 		user_img.save('./flask_deep/static/images/anogan'+str(user_img.filename))
# 		user_img_path = '/images/'+str(user_img.filename)

# 		# Anogan_G
# 		AnoGAN_G_img = AnoGAN_G.main(user_img_path) #수정 필요
# 		AnoGAN_G_img_path = '/images/'+str(AnoGAN_G_img.split('/')[-1])

#         # Anogan_D
# 		AnoGAN_D_img = AnoGAN_D.main(user_img_path) #수정 필요
# 		AnoGAN_D_img_path = '/images/'+str(AnoGAN_D_img.split('/')[-1])

# 		# Anogan_R
# 		AnoGAN_R_img = AnoGAN_R.main(user_img_path) #수정 필요
# 		AnoGAN_R_img_path = '/images/'+str(AnoGAN_R_img.split('/')[-1])


# 	return render_template('ano_post.html', 
# 					user_img=user_img_path, AnoGAN_G_img=AnoGAN_G_img_path, AnoGAN_D_img=AnoGAN_D_img_path, AnoGAN_R_img=AnoGAN_R_img_path)

'''yolov5'''
@app.route('/yolo_get')
def yolo_get():
	return render_template('yolo_get.html')

@app.route('/yolo_post', methods=['GET','POST'])
def file_upload():
	if request.method == 'POST':
		file = request.files['file']
		filename = secure_filename(file.filename)

		file_string = file.read()
		file_bytes = np.fromstring(file_string, np.uint8)
		file_bytes = np.frombuffer(file_bytes, np.uint8)
		img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

		boxes = yolov5.model(img, size=640).pandas().xyxy[-1].values.tolist()
		print(boxes)
		yolo_img = yolov5.plot_boxes(boxes, img)
		yolo_img_bytes = cv2.imencode('.jpg', yolo_img, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()
		collect.insert_one({"original_img": file_string, "yolo_img": yolo_img_bytes })


		original = base64.b64encode(file_string).decode('utf-8')
		yolo_img = base64.b64encode(yolo_img_bytes).decode('utf-8')
		# return response
		return render_template('yolo_post.html', original=original, result=yolo_img, time=time.time())
	return 'No Method'
