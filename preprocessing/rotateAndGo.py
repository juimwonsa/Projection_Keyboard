import cv2
import imutils
import numpy as np

WIDTH = 400
HEIGHT = 300

#1 - load the image
image_ori = cv2.imread("C:/Users/jw969/Desktop/keyb.jpg")

image_ori = cv2.resize(image_ori,(WIDTH,HEIGHT))
cv2.imshow("1-Original", image_ori)


#2 - Gary
image_gray = cv2.cvtColor(image_ori,cv2.COLOR_BGR2GRAY)
cv2.imshow("2-grayscale", image_gray)


#3 - Blur
image_blur = cv2.medianBlur(image_gray,19)#<<odd numbers only, prefer ~(WIDTH/20)~
#image_blur = cv2.GaussianBlur(image_gray,(9,9),0)
#medianBlur를 이용하여 아래에서 컨투어를 구할 때 키보드 자판 하나하나들로 나뉘는 것을 막음
# TODO: 가우시안 블러는 혹시 쓸 수도 있어서 남겨놓은것 나중에 지울것
cv2.imshow("3-blur", image_blur)


#4 - edge detection (Canny)
image_edge = cv2.Canny(image_blur, 100, 300, 3)
cv2.imshow("4-edge dection", image_edge)


#5 - Rotate
image_rotated = imutils.rotate_bound(image_edge, 45)  
cv2.imshow("5-Rotate 45 Degrees", image_rotated)


#6 - find contours, get largest one, get extrime points
cnts = cv2.findContours(image_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

image_rotated_color = imutils.rotate_bound(image_ori, 45)
image_rotated_color2 = image_rotated_color.copy()
#위의 두개는 바로 아래에서 이미지에 낙서를 해서 하나더만든거임
#TODO: 아래의 코드는 확인용이므로 지워도됨 그러나 프로젝트 끝날 때 까지는 살림
# draw the outline of the object, then draw each of the
cv2.drawContours(image_rotated_color2, [c], -1, (0, 255, 255), 2)
cv2.circle(image_rotated_color2, extLeft, 6, (0, 0, 255), -1)
cv2.circle(image_rotated_color2, extRight, 6, (0, 255, 0), -1)
cv2.circle(image_rotated_color2, extTop, 6, (255, 0, 0), -1)
cv2.circle(image_rotated_color2, extBot, 6, (255, 255, 0), -1)
cv2.imshow("6-contour image", image_rotated_color2)


#7 - affine 
#add extra pixel 
OFFSET = 20
aextLeft = np.asarray(extLeft)
aextRight = np.asarray(extRight)
aextTop = np.asarray(extTop)
aextBot = np.asarray(extBot)
print(aextLeft)
print(aextRight)
print(aextTop)
print(aextBot)
aextLeft += [-OFFSET,0]
aextRight += [OFFSET,0]
aextTop += [0,-OFFSET]
aextBot += [0,OFFSET]

pts1 = np.float32([aextBot,aextLeft,aextRight,aextTop])
pts2 = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
image_affine = cv2.warpPerspective(image_rotated_color,matrix,(WIDTH,HEIGHT))

cv2.imshow("7-affine ", image_affine)

cv2.waitKey(0)
cv2.destroyAllWindows()