#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@author:yujianghua
@file:eye.py
@time:2022/03/29
"""
'''
功能：眨眼疲劳检测
'''

from scipy.spatial import distance as dist
import cv2
import mediapipe as mp
import time
import os


#calculate ear
def eye_aspect_ratio(faceLm):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    p1 = [faceLm.landmark[160].x, faceLm.landmark[160].y]
    p2 = [faceLm.landmark[144].x, faceLm.landmark[144].y]
    p3 = [faceLm.landmark[158].x, faceLm.landmark[158].y]
    p4 = [faceLm.landmark[153].x, faceLm.landmark[153].y]
    p5 = [faceLm.landmark[33].x, faceLm.landmark[33].y]
    p6 = [faceLm.landmark[133].x, faceLm.landmark[133].y]

    A = dist.euclidean(p1, p2)
    B = dist.euclidean(p3, p4)

    C = dist.euclidean(p5, p6)
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    print(ear)
    # return the eye aspect ratio
    return ear


# resize frame
def rescaleFrame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# get video
cap = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils
mp_facemesh = mp.solutions.face_mesh
facemesh = mp_facemesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)


COUNTER = 0 #眨眼总次数
EYE_AR_THRESH = 0.35 #计算眨眼阈值，可调。
NUM_THRESH = 3 #单秒眨眼次数阈值,一秒内眨眼大于3次，认为疲劳（眨眼疲劳）
NUM_3THRESH = 71 #75帧眨眼四次，即3秒内眨眼次数少于4次，认为走神（眨眼走神）
waring = '' #眨眼警告


num_blink_per = 0 #单秒眨眼次数，每秒清零
num_per = 0 #1秒

num_3per = 0 #3秒 每3秒清零
num_blink_3per = 0 #3

frame = -1

while True:
    #一秒
    if num_per > 25:
        num_blink_per = 0
        num_per = 0
    #三秒
    if num_3per > 75:
        num_blink_3per = 0
        num_3per = 0

    num_per += 1
    num_3per += 1

    pre_time = time.time()
    success, img = cap.read()
    frame += 1
    scaled = rescaleFrame(img, 150)
    imgRGB = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mp_draw.draw_landmarks(scaled, faceLm, mp_facemesh.FACEMESH_CONTOURS, draw_spec, draw_spec)
            ear1 = eye_aspect_ratio(faceLm) #left ear
            if ear1 < EYE_AR_THRESH:
                COUNTER += 1
                num_blink_per += 1 #记录单秒眨眼次数
            else:
                num_blink_3per += 1

            if num_blink_per > NUM_THRESH:
                waring = 'Tired out'
            elif num_blink_3per > NUM_3THRESH:
                waring = 'Absent minded'
            else:
                waring = ''

    cv2.putText(scaled, str(int(COUNTER)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    cv2.putText(scaled, waring, (150, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    frame_path = os.path.join('../data', 'eye'+str(frame) + '.jpg')
    cv2.imwrite(frame_path, scaled)
    cv2.imshow("Image", scaled)
    cv2.waitKey(1)









