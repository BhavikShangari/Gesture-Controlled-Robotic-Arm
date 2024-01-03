import cv2
import time
import mediapipe as mp
import numpy as np
import math
from cvzone.SerialModule import SerialObject

arduino = SerialObject('/dev/ttyACM0')
front = cv2.VideoCapture(0)
side = cv2.VideoCapture(2)

mphands_front = mp.solutions.hands
hands_front = mphands_front.Hands(max_num_hands=1, static_image_mode = False, min_tracking_confidence = 0.5, min_detection_confidence=0.5)

mphands_side = mp.solutions.hands
hands_side = mphands_side.Hands(max_num_hands=1, static_image_mode = False, min_tracking_confidence = 0.5, min_detection_confidence=0.5)
drawingUtils = mp.solutions.drawing_utils

mppose_front = mp.solutions.pose
pose_front = mppose_front.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mppose_side = mp.solutions.pose
pose_side = mppose_side.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

he, w = 480, 640
while True:
    init = time.time()
    ret, frame_front = front.read()
    suc, frame_side = side.read()

    
    rgb_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
    rgb_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)

    results_hand_front = hands_front.process(rgb_front)
    results_hand_side = hands_side.process(rgb_side)
    results_pose_front = pose_front.process(rgb_front)
    results_pose_side = pose_side.process(rgb_side)

    if results_hand_front.multi_hand_landmarks and results_pose_front.pose_landmarks and results_hand_side.multi_hand_landmarks and results_pose_side.pose_landmarks:
        drawingUtils.draw_landmarks(frame_front, results_hand_front.multi_hand_landmarks[0], mphands_front.HAND_CONNECTIONS)
        drawingUtils.draw_landmarks(frame_front, results_pose_front.pose_landmarks, mppose_front.POSE_CONNECTIONS)
        drawingUtils.draw_landmarks(frame_side, results_hand_side.multi_hand_landmarks[0], mphands_front.HAND_CONNECTIONS)
        drawingUtils.draw_landmarks(frame_side, results_pose_side.pose_landmarks, mppose_side.POSE_CONNECTIONS)
        
        
        # For Big Vertical Motor Movement
        elbow_side = (results_pose_side.pose_landmarks.landmark[14].x * w, results_pose_side.pose_landmarks.landmark[14].y * he)
        wrist_side = (results_hand_side.multi_hand_landmarks[0].landmark[0].x*w, results_hand_side.multi_hand_landmarks[0].landmark[0].y*he)
        elbow2wrist = (elbow_side[0] - wrist_side[0], -(elbow_side[1] - wrist_side[1]))
        horizontal_vector= (1, 0)
        big_vertical_angle = int(abs( 180 - math.acos(np.dot(elbow2wrist, horizontal_vector) / np.linalg.norm(elbow2wrist)) * 180 / math.pi))
    
        # Dor small angled motor
        middle_finger_side = (results_hand_side.multi_hand_landmarks[0].landmark[12].x*w, results_hand_side.multi_hand_landmarks[0].landmark[12].y*he)
        wrist2middle = (middle_finger_side[0] - wrist_side[0], -(middle_finger_side[1] - wrist_side[1]))
        between_small_angle = int(np.clip(abs( 180 - (math.acos(np.dot(wrist2middle, elbow2wrist)/ (np.linalg.norm(wrist2middle) * np.linalg.norm(elbow2wrist))) * 180 / np.pi)), 0, 180))

        #For gripper in left motor and abs(180 - this angle) in right motor__________THIS IS FROM FRONT CAMERA
        index_base = (results_hand_front.multi_hand_landmarks[0].landmark[5].x*w, results_hand_front.multi_hand_landmarks[0].landmark[5].y*he)
        index_tip = (results_hand_front.multi_hand_landmarks[0].landmark[8].x*w, results_hand_front.multi_hand_landmarks[0].landmark[8].y*he)
        thumb_tip = (results_hand_front.multi_hand_landmarks[0].landmark[4].x*w, results_hand_front.multi_hand_landmarks[0].landmark[4].y*he)
        indexbase2indextip = (index_tip[0] - index_base[0], -(index_tip[1] - index_base[1]))
        indexbase2thumbtip = (thumb_tip[0] - index_base[0], -(thumb_tip[1] - index_base[1]))
        gripper_left_angle = int(np.clip(abs(math.acos(np.dot(indexbase2indextip, indexbase2thumbtip) / (np.linalg.norm(indexbase2thumbtip) * np.linalg.norm(indexbase2indextip))) * 180 / np.pi - 10), 0, 90))
        

        #for Big Horizontal motor
        elbow3d = (results_pose_front.pose_landmarks.landmark[14].x * w, results_pose_front.pose_landmarks.landmark[14].y * he, results_pose_side.pose_landmarks.landmark[14].x * w)
        middle3d = (results_hand_front.multi_hand_landmarks[0].landmark[12].x*w, results_hand_front.multi_hand_landmarks[0].landmark[12].y*he, results_hand_side.multi_hand_landmarks[0].landmark[12].x*w)
        #Y component is zero (projection on xz plane)
        elbow2middle3dproj = (middle3d[0] - elbow3d[0], 0, middle3d[2] - elbow3d[2])
        zcomponentvector = (0, 0, 1)
        if results_hand_front.multi_hand_landmarks[0].landmark[12].x <= results_pose_front.pose_landmarks.landmark[14].x:
            big_hor_angle = int(np.clip(90 - math.acos((np.dot(zcomponentvector, elbow2middle3dproj) / np.linalg.norm(elbow2middle3dproj)) )* 180 / np.pi, 0, 180))
        else:
            big_hor_angle = int(np.clip(90 + math.acos((np.dot(zcomponentvector, elbow2middle3dproj) / np.linalg.norm(elbow2middle3dproj)) )* 180 / np.pi, 0, 180))

        arduino.sendData([big_hor_angle, big_vertical_angle, between_small_angle, gripper_left_angle])

   
    cv2.imshow('front', frame_front)
    cv2.imshow('side', frame_side)
    end = time.time()
    if cv2.waitKey(1) == ord('d'):
        break

    front.release()
    side.release()