import numpy as np
import cv2
import GridPoints
import math
import heapq
import RelativeGridNumber
rho0=1
aGrid=1
theta0=np.pi/180
houghThres=100
minLineLength=200
maxLineGap=100
midX=151.
midY=201.
def SelectPoints(path,OneUmInPixel,OneUmInPixelX,OneUmInPixelY,minX,minY):
    #finding the lines
    img= cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres= cv2.adaptiveThreshold(gray, gray.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    thres2=thres.copy()
    thres2[thres>0]=0
    thres2[thres==0]=255
    thres=thres2
    #cv2.imshow('thresadp', thres2)
    #cv2.waitKey(0)
    ret,thres= cv2.threshold(gray, 100,gray.max(),  cv2.THRESH_TOZERO_INV)
    ##print(thres)
    #cv2.imshow('thres', thres)
    #cv2.waitKey(0)
    gray=thres2
    lines=cv2.HoughLinesP(gray,rho0,theta0,houghThres,minLineLength=minLineLength,maxLineGap=maxLineGap)
    #lines = cv2.HoughLines(gray, rho0, theta0, houghThres)

    ##print(lines)
    img1=img.copy()
    img4=img.copy()
    ##print(lines)
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 1)
        #k,d= line[0]
        ##print(k)
        ##print(d)
        #cv2.line(img1,(0,round(d)),(1,round(k+d)),(0,0,255),1)
    #cv2.imshow('img',img1)
    #cv2.waitKey(0)
    #print('length of lines:')
    #print(len(lines))

    #finding the closest and the nearest
    list = GridPoints.findingIntersection(path)
    # distance=[]
    # for point in list:
    #     distance.append(math.sqrt(math.pow(point[0]-midX,2)+math.pow(point[1]-midY,2)))
    # distance=np.array(distance)
    # biggest=heapq.nlargest(11,range(len(distance)),distance.take)
    # smallest=heapq.nsmallest(4,range(len(distance)),distance.take)
    # minX=list[smallest[0]][0]
    # #print('minX')
    # #print(minX)
    # minY=list[smallest[0]][1]
    # minX2=list[smallest[1]][0]
    # minY2=list[smallest[1]][1]
    # minX3=list[smallest[2]][0]
    # minY3=list[smallest[2]][1]
    # minX4=list[smallest[3]][0]
    # minY4=list[smallest[3]][1]
    # testX=list[biggest[10]][0]
    # testY=list[biggest[10]][1]

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

    # cv2.circle(img,(testX, testY),1,(255,0,0),-1, cv2.LINE_AA)
    # cv2.circle(img,(minX, minY),1,(255,0,0),-1,cv2.LINE_AA )
    # cv2.circle(img,(minX2, minY2),1,(0,255,0),-1,cv2.LINE_AA )
    # cv2.circle(img,(minX3, minY3),1,(0,255,0),-1,cv2.LINE_AA )
    # cv2.circle(img,(minX4, minY4),1,(0,255,0),-1,cv2.LINE_AA )
    # cv2.circle(img,(int(midX), int(midY)),1,(0,255,0),-1,cv2.LINE_AA )
    # cv2.circle(img, (0, 0), 3, (0, 255, 0), -1, cv2.LINE_AA)
    # cv2.circle(img,(0,401),3,(255,0,0),-1,cv2.LINE_AA)
    # cv2.circle(img, (301, 0), 3, (255, 0, 0), -1, cv2.LINE_AA)
    # cv2.imshow('img2',img)
    # cv2.waitKey(0)
    #
    # OneUmInPixelY=0.5*(math.sqrt(math.pow(minX-minX2,2)+math.pow(minY-minY2,2))+math.sqrt(math.pow(minX4-minX3,2)+math.pow(minY4-minY3,2)))
    # OneUmInPixelX=0.5*(math.sqrt(math.pow(minX4-minX,2)+math.pow(minY4-minY,2))+math.sqrt(math.pow(minX3-minX2,2)+math.pow(minY3-minY2,2)))
    # OneUmInPixel=0.5*(OneUmInPixelX+OneUmInPixelY)
    #OneUmInPixelX=OneUmInPixel
    #OneUmInPixelY=OneUmInPixel
    # if(abs(minY2-minY)<abs(minX2-minX)):
    #     r=OneUmInPixelX
    #     OneUmInPixelX=OneUmInPixelY
    #     OneUmInPixelY=r
    #print('OneUmInPixelX:')
    #print(OneUmInPixelX)
    #print('OneUmInPixelY:')
    #print(OneUmInPixelY)
    #calculate the scope
    posSum=0
    posNum=0
    negSum=0
    negNum=0
    xNewLine=[]
    x_d=[]
    yNewLine=[]
    y_d=[]
    maxD=0.8*OneUmInPixel
    #maxD2=
    lines=lines.tolist()
    for line in lines:
        x1,y1,x2,y2 = line[0]
        k=float(y1-y2)/float(x1-x2)
        d=float(y1)-k*float(x1)
        line[0].append(k)
        line[0].append(d)
        if k > 0 :
            ##print(k)
            found = False
            for i in x_d:
                if abs(float(i - d)*1/math.sqrt(1+math.pow(k,2))) < maxD:
                    found = True
                    break
            if found:
                continue
            x_d.append(d)
            posSum+=k
            posNum+=1
            xNewLine.append(line[0])
        else:
            #print('k:')
            #print(k)
            found=False
            for i in y_d:
                if abs(float(i - d)*1/math.sqrt(1.+math.pow(-k,2))) < maxD:
                    found = True
                    break
            for line2 in yNewLine:
                if abs(x1-line2[0])<5 or abs(x2-line2[3])<5:
                    found=True
                    break
            if found:
                continue
            y_d.append(d)
            negNum+=1
            negSum+=k
            yNewLine.append(line[0])
    xline=sorted(xNewLine,key=lambda s:s[5])
    yline=sorted(yNewLine,key=lambda s:s[5])
    #print('ylineNumber')
    #print(len(yline))
    posSlope=float(posSum)/float(posNum)
    negSlope=float(negSum)/float(negNum)
    xline2=xline.copy()
    yline2=yline.copy()
    minD=1.5*OneUmInPixel
    minD2=2.3*OneUmInPixel
    #print('start Adding')
    for i in range(len(xline)-1):
        p=abs(xline[i][5]-xline[i+1][5]*1)/math.sqrt(1+math.pow(xline[i][4],2))
        ##print(p)
        if(p>minD2):
            #print('ERR0R!')
            print('attention')
            print('OneGirdPixel:')
            print(OneUmInPixel)
            return []
        if (p>minD):
            ##print(p)
            #print('Adding')
            m=np.array(xline[i])+np.array(xline[i+1])
            m=np.rint(0.5*m)
            m=m.astype(int)
            m=m.tolist()
            ##print(m)
            xline2.append(m)
    #print(yline)
    for i in range(len(yline) - 1):
        p=abs((yline[i][0] - yline[i + 1][0] * 1)*negSlope-(yline[i][1]-yline[i+1][1]))/math.sqrt(1+pow(negSlope,2))
        ##print(p)
        if (p > minD2):
            #print('ERR0R!')
            print('attention')
            print('OneGirdPixel:')
            print(OneUmInPixel)
            return []

        if (p>minD):
            #print('Adding')
            #print(minD)
            #print(p)
            m=np.array(yline[i])+np.array(yline[i+1])
            m=np.rint(0.5*m)
            m=m.astype(int)
            m=m.tolist()
            ##print(m)
            yline2.append(m)
    ##print(xline2)
    img3=img.copy()
    for line in xline:
        x1, y1, x2, y2, k, d= line
        cv2.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 1)
    for line in yline:
        x1, y1, x2, y2, k, d= line
        cv2.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #cv2.imshow('img_line1',img3)
    #cv2.waitKey(0)
    #print('yline2')
    #print(len(yline2))
    for line in xline2:
        x1, y1, x2, y2, k, d= line
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #cv2.imshow('img_line2',img)
    #cv2.waitKey(0)

    for line in yline2:
        x1, y1, x2, y2, k, d= line
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        ##print('d')
        ##print(d)
        #print(line)
    #cv2.imshow('img_line2',img)
    #cv2.waitKey(0)


    '''
    #print(len(x_d)+len(y_d))
    posSlope=float(posSum)/float(posNum)
    negSlope=float(negSum)/float(negNum)
    #print('posSlope:')
    #print(posSlope)
    #print('negSlope:')
    #print(negSlope)
    xnormalCo=[1/(math.sqrt(1+math.pow(posSlope,2))),posSlope/(math.sqrt(1+math.pow(posSlope,2)))]
    ynormalCo=[1/(math.sqrt(1+math.pow(negSlope,2))),negSlope/(math.sqrt(1+math.pow(negSlope,2)))]
    BT=[float(maxX-minX),float(maxY-minY)]
    cBT=BT[0]*xnormalCo[0]+BT[1]*xnormalCo[1]
    rBT=BT[0]*ynormalCo[0]+BT[1]*ynormalCo[1]
    columNumber=math.ceil(abs(cBT/OneUmInPixel))*cBT/abs(cBT)
    rowNumber=math.ceil(abs(rBT/OneUmInPixel))*rBT/abs(rBT)
    #print('columNumber:')
    #print(columNumber)
    #print('rowNumber:')
    #print(rowNumber)
    '''
    xline2=sorted(xline2,key=lambda s:s[5])
    yline2=sorted(yline2,key=lambda s:s[5])
    minD3=0.5*OneUmInPixel
    #rx,ry=RelativeGridNumber.RGN([testX,testY], [minX,minY], xline, yline, minD3)
    pointSelected=[]
    #[[x_inImage(PIXEL),y_inImage(PIXEL),x_inReality(UM),y_shouldInReality(UM)]]
    #attention please!
    minMidDisX=((minX-midX)*1+(minY-midY)*posSlope)/math.sqrt(1+math.pow(posSlope,2))#PIXEL
    minMidDisY=((minX-midX)*1+(minY-midY)*negSlope)/math.sqrt(1+math.pow(negSlope,2))#PIXEL
    # print('minMidDisX:')
    # print(minMidDisX)
    # print('minMidDisY:')
    # print(minMidDisY)
    # print('imgSize:')
    # print(img.shape)
    for point in list:
        cv2.circle(img4, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1, cv2.LINE_AA)
        rx, ry = RelativeGridNumber.RGN(point, [minX, minY], xline2, yline2, minD3)

        if abs(rx)<1000:
            #m=[point[0]-midX,point[1]-midY,(rx+minMidDisX/OneUmInPixel)*aGrid,(ry+minMidDisY/OneUmInPixel)*aGrid]
            #m=[point[0]-midX,point[1]-midY,rx*OneUmInPixelX+minMidDisX,ry*OneUmInPixelY+minMidDisY]
            m = [point[0] - midX, point[1] - midY, rx * OneUmInPixelX + minMidDisX, ry * OneUmInPixelY + minMidDisY]
            #print(m)
            pointSelected.append(m)
    # for point2 in pointSelected:
    #     print(point2)
    #     cv2.circle(img4, (int(point2[0]+midX), int(point2[1]+midY)), 1, (255, 0, 0), -1, cv2.LINE_AA)
    #
    # #cv2.imshow('img4',img4)
    # #cv2.waitKey(0)
    # for point2 in pointSelected:
    #     cv2.circle(img4, (int(point2[2]+midX), int(point2[3]+midY)), 1, (0, 255, 0), -1, cv2.LINE_AA)
    # #cv2.imshow('img4', img4)
    # #cv2.waitKey(0)
    if(len(pointSelected)==0):
        print('attention')
        print('OneGirdPixel:')
        print(OneUmInPixel)
    #print('len pointSelcted')
    #print(len(pointSelected))
    return pointSelected
