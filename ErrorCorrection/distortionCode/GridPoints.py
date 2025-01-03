import numpy as np
import cv2

def findingIntersection(path):
    pS = 6
    pS2=3
    #getting the intersections
    img= cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray=gray.max()*np.ones(np.shape(gray),dtype=np.uint8())-gray
    #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    #ret, thres= cv2.adaptiveThreshold(gray, 100, np.max(gray), cv2.THRESH_BINARY)
    thres= cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    thres2=thres.copy()
    thres2[thres>0]=0
    thres2[thres==0]=255
    thres=thres2
    #cv2.imshow('thres', thres)


    k1=np.ones((1,pS), np.uint8)
    k2=np.ones((pS,1), np.uint8)
    k3=np.ones((pS2,pS2),np.uint8)

    mask = cv2.morphologyEx(thres, cv2.MORPH_OPEN, k1)
    #cv2.imshow('open', mask)

    mask2 = cv2.morphologyEx(thres, cv2.MORPH_OPEN, k2)
    #cv2.imshow('open2', mask2)

    for x in range(401):
        for y in range(301):
            if(mask[x,y]==0 or mask2[x,y]==0):
                mask[x,y]=0
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    # cv2.imshow('close', mask)

    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list=[]
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        #cv2.rectangle(img, (x, y),(x+w,y+h),(0, 255, 0), 2)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        #cv2.circle(img,(cx, cy),10,(0,255,0),2, cv2.LINE_AA)
        cv2.circle(img,(cx, cy),1,(0,0,255),-1, cv2.LINE_AA)
        img[cy,cx]=(0,255,0)
        list.append([cx,cy])
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #cv2.imwrite('final_artiInter_8_8.jpg',img)

    cv2.circle(img,(151,201),3,(255,0,0),2,cv2.LINE_AA)
    #cv2.imshow('img2',img)
    #cv2.waitKey(0)
    return list