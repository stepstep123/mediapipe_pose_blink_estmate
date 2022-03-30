#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@author:yujianghua
@file:pose.py
@time:2022/03/29
"""
'''
功能：低头检测、摇头检测
'''

import math
import os
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

def rotation_matrix_to_angles(rotation_matrix):

    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

frame = -1
while cap.isOpened():
    success, image = cap.read()
    frame += 1

    # Convert the color space from BGR to RGB and get Mediapipe results
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Convert the color space from RGB to BGR to display well with Opencv
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_coordination_in_real_world = np.array([
        [285, 528, 200],
        [285, 371, 152],
        [197, 574, 128],
        [173, 425, 108],
        [360, 574, 128],
        [391, 425, 108]
    ], dtype=np.float64)

    h, w, _ = image.shape
    face_coordination_in_image = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 57, 130, 287, 359]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_coordination_in_image.append([x, y])

            face_coordination_in_image = np.array(face_coordination_in_image,
                                                  dtype=np.float64)

            # The camera matrix
            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Use solvePnP function to get rotation vector
            success, rotation_vec, transition_vec = cv2.solvePnP(
                face_coordination_in_real_world, face_coordination_in_image,
                cam_matrix, dist_matrix)

            # Use Rodrigues function to convert rotation vector to matrix
            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

            result = rotation_matrix_to_angles(rotation_matrix)
            for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                k, v = info
                text = f'{k}: {int(v)}'
                # text = ''
                if k == 'pitch' and int(v) < 0:
                    text = text + " tired out"
                if k == 'yaw' and (int(v) > 30 or int(v) < -30) :
                    text = text + " absent-minded"
                cv2.putText(image, text, (2, i * 30 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)
    frame_path = os.path.join('../data', str(frame) + '.jpg')
    # cv2.imwrite(frame_path, image)
    cv2.imshow('Head Pose Angles', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()

