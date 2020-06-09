# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:26:47 2020

@author: gmlgn
"""
import numpy as np
import cv2
import pyautogui
import finger_log


def start():

    capture = cv2.VideoCapture(0)

    capture.set(3, 600)
    capture.set(4, 400)

    lower_red = np.array([0, 100, 100],  dtype="uint8") 
    upper_red = np.array([4, 255, 255],  dtype="uint8")

    max1 = 0
    maxcX = 0
    maxcY = 0
    beforecX = 0
    beforecY = 0
    
    
    beforecX = 0
    beforecY = 0

    tick = 0

    framenum = 0

    center_x = int(300)
    center_y = int(200)

    location = (center_x - 200, center_y - 100)
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # hand-writing style font
    fontScale = 3.5



    while True:
    
    
        ret, frame = capture.read()
    
        framenum = framenum+1
        #gaussian blur 적용
        frame1 = cv2.blur(frame,(10,10))
    
        #영상을 RGB에서 HSV로 바꿈
        img_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

        red_result = cv2.inRange(img_hsv, lower_red, upper_red)
    
        #이진화된 이미지의 잡음 제거
        kernel = np.ones((17, 17), np.uint8)
        result = cv2.erode(red_result, kernel, iterations = 1)

        #red_result = cv2.bitwise_and(frame1, frame1, mask=red_range)
        #pyautogui.typewrite("hello test")


        #이미지의 contour 찾기
        ret, thr = cv2.threshold(result, 127 ,255, 0)
        _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        #cv2.drawContours(frame1,contours,-1,(0,0,255),3)
        for i in contours:
        
            M = cv2.moments(i, False)
            if M['m00']==0 : M['m00'] = 1
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            if(M['m00']>max1): 

                max = M['m00']
                maxcX = cX
                maxcY = cY
                contour = contours
        
    
        cv2.circle(frame1, (maxcX, maxcY), 3, (255, 0, 0), -1)
        cv2.drawContours(frame1,contours,-1,(0,0,255),3)
        
        if maxcY > 200:
            tick += 1 
            if(tick > 30):
                pyautogui.typewrite("hello test")
                pyautogui.press('enter')
        else:
            tick = 0
        
        cv2.putText(frame1, str(maxcX), location, font, fontScale, (255,255,255), 2)
        cv2.putText(frame1, str(maxcY), (center_x - 200, center_y ), font, fontScale, (255,255,255), 2)
        
        cv2.imshow("Source", frame1)
    
        cv2.imshow("red",red_result)
        
        if(framenum == 1):
            pass
        else:
            finger_log.fingerLog(framenum, beforecX, beforecY, maxcX, maxcY)
    
        beforecX = maxcX
        beforecY = maxcY
        if cv2.waitKey(1) > 0: break
    
    capture.release()
    cv2.destroyAllWindows()