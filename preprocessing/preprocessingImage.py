import numpy as np
import cv2
import imutils
import copy
import math

WIDTH = 400
HEIGHT = 300

cap = cv2.VideoCapture(0)

#*********************************************************
#**************************setup**************************
#*********************************************************

while(True):
    #1 - load the image
    ret, image_ori = cap.read()

    #2 - Gary
    image_gray = cv2.cvtColor(image_ori,cv2.COLOR_BGR2GRAY)
    
    #3 - Blur
    image_blur = cv2.medianBlur(image_gray,9)#<<odd numbers only, prefer ~(WIDTH/20)~
    #image_blur = cv2.GaussianBlur(image_gray,(9,9),0)
    #medianBlur를 이용하여 아래에서 컨투어를 구할 때 키보드 자판 하나하나들로 나뉘는 것을 막음
    # TODO: 가우시안 블러는 혹시 쓸 수도 있어서 남겨놓은것 나중에 지울것
    #cv2.imshow("3-blur", image_blur)
    
    #4 - edge detection (Canny)
    image_edge = cv2.Canny(image_blur, 100, 200, 3)
    #cv2.imshow("4-edge dection", image_edge)
    
    #5 - Rotate
    image_rotated = imutils.rotate_bound(image_edge, 45)  
    #cv2.imshow("5-Rotate 45 Degrees", image_rotated)
    
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
    OFFSET = 10
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

    # Display the resulting frame
    cv2.imshow("3-blur", image_blur)
    cv2.imshow("4-edge dection", image_edge)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
#*********************************************************
#**************************loop***************************
#*********************************************************
    
while(True):
    #1 - load the image
    ret, image_ori = cap.read()

    #2 - Gary
    image_gray = cv2.cvtColor(image_ori,cv2.COLOR_BGR2GRAY)
    
    #3 - Rotate
    image_rotated_gray = imutils.rotate_bound(image_gray, 45) 
    
    #4 - Affine
    image_affine = cv2.warpPerspective(image_rotated_gray,matrix,(WIDTH,HEIGHT))    
    
    cv2.imshow("7-affine ", image_affine)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#********************************************************************



# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)


        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            
            #cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print (cnt)
                    #app('System Events').keystroke(' ')  # simulate pressing blank space
                    

        cv2.imshow('output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
        
#********************************************************************
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()