import cv2

import numpy as np

lower = np.array([0,40,70], dtype="uint8")

upper = np.array([20,255,255], dtype="uint8")

#kernel = np.ones((5,5),np.float32)/(5*2)

capture = cv2.VideoCapture(0)

capture.set(3, 640)
capture.set(4, 480)

while True:
    
    ret, frame = capture.read()

    #gaussian blur 적용
    frame1 = cv2.blur(frame,(10,10))
    
    #영상을 RGB에서 HSV로 바꿈
    img_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    
    #img_hsv = cv2.fastNlMeansDenoisingColoredMulti(img_hsv,2,5,None,3,3,7,21)
    
    #살색 히스토그램에 대한 Range적용
    img_hand = cv2.inRange(img_hsv, lower, upper)
    
    #x축 contour 검출
    img_sobel_x = cv2.Sobel(img_hand, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

    #y축 contour 검출
    img_sobel_y = cv2.Sobel(img_hand, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)



    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0);

    

    cv2.imshow("Source", frame1)
    
    
    cv2.imshow("VideoFrame", img_sobel)
    
    
    
    
    
    
    if cv2.waitKey(1) > 0: break





capture.release()
cv2.destroyAllWindows()