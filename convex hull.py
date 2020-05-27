# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:24:15 2020

@author: gmlgn
"""


import cv2

import numpy as np

lower = np.array([0,40,70], dtype="uint8")

upper = np.array([20,255,255], dtype="uint8")

#kernel = np.ones((5,5),np.float32)/(5*2)

#배경제거를 위한 GMG
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

capture = cv2.VideoCapture(0)

capture.set(3, 600)
capture.set(4, 400)

#ret, frame = capture.read()
#cv2.imwrite('background.jpg',frame)

#sub1 =cv2.imread('background.jpg')


while True:
    
    ret, frame = capture.read()

    #gaussian blur 적용
    frame1 = cv2.blur(frame,(6,6))
    
    #영상을 RGB에서 HSV로 바꿈
    img_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    #살색 히스토그램에 대한 Range적용
    img_hand = cv2.inRange(img_hsv, lower, upper)
    
    #이진화된 이미지의 잡음 제거
    kernel = np.ones((7, 7), np.uint8)
    result = cv2.erode(img_hand, kernel, iterations = 1)

    #이미지의 contour 찾기
    ret, thr = cv2.threshold(result, 127 ,255, 0)
    _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(frame,contours,-1,(0,0,255),3)
    
    #이미지 Convex Hull 구하기
    for i in contours:
        hull = cv2.convexHull(i, clockwise=True)
        cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)

    cv2.imshow("Source", frame)
    
    cv2.imshow("Gaussian", frame1)
    
    cv2.imshow("hand", img_hand)
    
    cv2.imshow("contours", result)

    
    if cv2.waitKey(1) > 0: break





capture.release()
cv2.destroyAllWindows()