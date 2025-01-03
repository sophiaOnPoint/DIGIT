import cv2
import GridPoints
import math
import numpy as np
import heapq
midX=151.
midY=201.
def getOneGridPixel():
    for i in range(50):
         #print('number:')
         j=i+1
         #print(j)
         path='../initialPic/p#_'+str(j)+'.jpg'
         if (j==1) : img_all=cv2.imread(path)
         else: img_all=(img_all*float(i)+cv2.imread(path))/float(j)
         #point_OnePic=SelectPoints.SelectPoints(path)
         #print(point_OnePic)
         #point_all+=point_OnePic
     #img_all=img_all/float(picNum)
    cv2.imwrite('addupImage.jpg',img_all)
    list = GridPoints.findingIntersection('addupImage.jpg')
    distance = []
    for point in list:
        distance.append(math.sqrt(math.pow(point[0] - midX, 2) + math.pow(point[1] - midY, 2)))
    distance = np.array(distance)
    biggest = heapq.nlargest(11, range(len(distance)), distance.take)
    smallest = heapq.nsmallest(4, range(len(distance)), distance.take)
    minX = list[smallest[0]][0]
    # print('minX')
    # print(minX)
    minY = list[smallest[0]][1]
    minX2 = list[smallest[1]][0]
    minY2 = list[smallest[1]][1]
    minX3 = list[smallest[2]][0]
    minY3 = list[smallest[2]][1]
    minX4 = list[smallest[3]][0]
    minY4 = list[smallest[3]][1]
    testX = list[biggest[10]][0]
    testY = list[biggest[10]][1]

    '''
    maxl=0
    maxX=0
    maxY=0
    minl=100000
    minX=0
    minY=0
    minl2=15000
    minX2=0
    minY2=0
    minl3=16000
    minX3=0
    minY3=0
    for point in list:
        if math.pow(point[0]-midX,2)+math.pow(point[1]-midY,2)>math.pow(maxl,2) :
            maxl=math.sqrt(math.pow(point[0]-midX,2)+math.pow(point[1]-midY,2))
            maxX=point[0]
            maxY=point[1]
        elif math.pow(point[0]-midX,2)+math.pow(point[1]-midY,2)<math.pow(minl,2) :
            minl= math.sqrt(math.pow(point[0]-midX,2)+math.pow(point[1]-midY,2))
            minX=point[0]
            minY=point[1]
        elif math.pow(point[0]-midX,2)+math.pow(point[1]-midY,2)<math.pow(minl2,2) :
            minl2= math.sqrt(math.pow(point[0]-midX,2)+math.pow(point[1]-midY,2))
            minX2=point[0]
            minY2=point[1]
    '''
    img=cv2.imread('addupImage.jpg')
    cv2.circle(img, (testX, testY), 1, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (minX, minY), 1, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (minX2, minY2), 1, (255, 50, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (minX3, minY3), 1, (255, 100, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (minX4, minY4), 1, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (int(midX), int(midY)), 1, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (0, 0), 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (0, 401), 3, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (301, 0), 3, (255, 0, 0), -1, cv2.LINE_AA)
    # cv2.imshow('img2', img)
    # cv2.waitKey(0)

    OneUmInPixelY = 0.5 * (math.sqrt(math.pow(minX - minX2, 2) + math.pow(minY - minY2, 2)) + math.sqrt(
        math.pow(minX4 - minX3, 2) + math.pow(minY4 - minY3, 2)))
    OneUmInPixelX = 0.5 * (math.sqrt(math.pow(minX3 - minX, 2) + math.pow(minY3 - minY, 2)) + math.sqrt(
        math.pow(minX4 - minX2, 2) + math.pow(minY4 - minY2, 2)))
    OneUmInPixel = 0.5 * (OneUmInPixelX + OneUmInPixelY)
    # OneUmInPixelX=OneUmInPixel
    # OneUmInPixelY=OneUmInPixel
    # if(abs(minY2-minY)<abs(minX2-minX)):
    #     r=OneUmInPixelX
    #     OneUmInPixelX=OneUmInPixelY
    #     OneUmInPixelY=r
    print('OneUmInPixelX:')
    print(OneUmInPixelX)
    print('OneUmInPixelY:')
    print(OneUmInPixelY)
    #minMidDisX=((minX-midX)*1+(minY-midY)*posSlope)/math.sqrt(1+math.pow(posSlope,2))#PIXEL
    #minMidDisY=((minX-midX)*1+(minY-midY)*negSlope)/math.sqrt(1+math.pow(negSlope,2))#PIXEL
    return [OneUmInPixel,OneUmInPixelX,OneUmInPixelY,minX,minY]

print(getOneGridPixel())