# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 02:03:38 2020


@author: gmlgn
"""
def fingerLog(framenum, beforecX, beforecY, cX, cY):
    
    #로그


    #logging.basicConfig(filename = './log/test.log', level=logging.DEBUG,format = "frame", filemode = 'w')
    
    
    #logging.debug( ' ' +str(framenum) + ' ' + str(cX) +' '+ str(cY))    
    
    f = open('./log/test.txt','a')
    
    #if list(enumerate(f))[-1][0]+1>:
        
    vecX = cX - beforecX
    vecY = cY - beforecY
    vec = None
    
    if(isVaild(vecX, vecY)):
        if(vecY > 0):
            vec = "down"
        else:
            vec = "up"
    else:
        vec = "unvaild"
    
    f.write("frame: " + str(framenum) + "\n" + "beforeX: " + str(beforecX)+ "\n" 
            + "beforeY: " + str(beforecY) + "\n" + "currentX: " + str(cX) + "\n" 
            + "currentY: "+ str(cY) + "\n" + vec + "\n" + "\n\n")
    
    f.close()
    

def isVaild(vecX, vecY):
    if(vecX<30 & vecY<30):
        return True
