import cv2

pMOG2 = cv2.createBackgroundSubtractorMOG2()


img_source = cv2.imread('myHand.jpg',0)

#이진화
img_result1 = cv2.threshold(img_source, 127, 255, cv2.THRESH_BINARY)
#적응형 이진화
img_result2 = cv2.adaptiveThreshold(img_source, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
#otsu 이진화
img_result3 = cv2.threshold(img_source, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow("thresholding", img_result1)

cv2.imshow("adapt thresholding", img_result2)

cv2.imshow("otsu thresholding", img_result3)

while(1):    
    
    if cv2.waitKey(1) & 0xFF == 27:
        
        cv2.destroyAllWindows()

