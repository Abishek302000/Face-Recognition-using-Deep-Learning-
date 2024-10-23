# -*- coding: ulf-8 -*-
"""

@author: Abishek
"""

import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition


webcam_video_stream = cv2.VideoCapture(0)

face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r", encoding="utf-8").read())
face_exp_model.load_weight('ataset/facial_expression_model_weight.h5')
emotions_label = ('angry', 'fear', 'happy', 'sad', 'surprise', 'neutral')
all_face_location = []

while True:
  ret, current_frame = webcam_video_stream.read()
  current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25,fy=0.25)
  all_face_location = face_recognition.face_locations(current_frame_small,model='hog')


for index
