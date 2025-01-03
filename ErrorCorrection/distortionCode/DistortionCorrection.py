import cv2
import SelectPoints
import math
import numpy as np
from scipy.optimize import curve_fit
import distortion
import undistort as undistort
import OneGridPixel

picNum=50
point_all=[]
para_list=[[],[],[],[],[],[]]
#add up to try to offset the noise
OnePixel,OnePixelX,OnePixelY,minX,minY=OneGridPixel.getOneGridPixel()
def get_para(path):
    #print(path)
    point_OnePic=SelectPoints.SelectPoints(path,OnePixel,OnePixelX,OnePixelY,minX,minY)
    # print(point_OnePic)
    # #point_aPic=np.array(point_OnePic)
    # print('point_apic')[
    point_aPic=point_OnePic
    # print(point_aPic)
    y_expected=np.zeros(len(point_aPic))
    if(len(point_aPic)==0):
        print('warning')
        print('pointaPic')
        print(point_aPic)
        print('pic')
        print(cv2.imread(path))
        return None
    #print('y_expected')
    #print(y_expected)
    para, eff=curve_fit(distortion.distortion,point_aPic,y_expected, maxfev = 50000)
    while(para[0]>=2*math.pi):
        para[0]=para[0]-2*math.pi
    while(para[0]<0):
        para[0]=para[0]+2*math.pi
    for i in range(0,6):
        para_list[i].append(para[i])
    # print('parameter:')
    # print(para)
    # print('effect:')
    # print(eff)
    auv=0
    i=0
    while(i<6):
        auv=auv+eff[i,i]
        i=i+1
    #print('addUpVariance:')
    #print(auv)
# Add figues
# for i in range(50):
#     #print('number:')
#     j=i+1
#     #print(j)
#     path='p#_'+str(j)+'.jpg'
#     if (j==1) : img_all=cv2.imread(path)
#     else: img_all=(img_all*float(i)+cv2.imread(path))/float(j)
#     #point_OnePic=SelectPoints.SelectPoints(path)
#     #print(point_OnePic)
#     #point_all+=point_OnePic
# #img_all=img_all/float(picNum)
# cv2.imwrite('addupImage.jpg',img_all)
# a=SelectPoints.SelectPoints('addupImage.jpg')
# get_para('addupImage.jpg')

#simulation on each figure
for i in range(50):
    #print('number:')
    j = i + 1
    # if(j==4 or j==5 or j==8 or j==9):continue
    #print(j)
    path = '../initialPic/p#_' + str(j) + '.jpg'
    #img=cv2.imread(path)
    #print(img)
    get_para(path)
get_para('addupImage.jpg')
para_mean=[]
for i in range(0,6):
    print(i)
    #arrayI=np.array(para_list[i])
    print('average:')
    print(np.mean(para_list[i]))
    para_mean.append(np.mean(para_list[i]))
    print('variance:')
    print(np.var(para_list[i]))
with open('../parameterNotebook.txt','w') as f:
    for i in range(0,6):
        f.write(str(np.var(para_list[i])))
        f.write('\n')
for i in range(1,51):
    #print(i)
    undistort.undistort('../initialPic/p#_'+str(i)+'.jpg','../parameterNotebook.txt','../undistorted/undistorted'+str(i)+'.jpg')

#
# #print('totalSelectedPoints:')
# #print(len(point_all))
# pos_inImage=point_all[0:1][:]
# #print(pos_inImage)
# pos_inReality=point_all[2:3][:]

# img=cv2.imread('p#_2.jpg')
# h,w=img.shape[:2]
# print('h')
# print(h)
# print('w')
# print(w)
# mtx=np.array([[1,0,0],[0,1,0],[0,0,1]])*1e-3
# dist=np.array([para_mean[1],para_mean[2],para_mean[4],para_mean[5],para_mean[3]])
# #dist=np.array([para_mean[1],para_mean[2],para_mean[4],para_mean[5]])
# print('dist')
# print(dist)
# #try distortion correction change to 4 digits:newCamMatrix very different
# #try to change mtx
# newcameramtx,roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h),True)
# #
# print('newcameramtx')
# print(newcameramtx)
# dst=cv2.undistort(img,mtx,dist,None,newcameramtx)
# print(dst)
# x,y,w,h=roi
# print('roi')
# print(roi)
# dst=dst[y:y+h,x:x+w]
# print('h')
# print(h)
# print('w')
# print(w)
# print(dst)
# cv2.imshow('dst',dst)
# cv2.waitkey(0)
# cv2.imwrite('undistorted2.jpg',dst)



#popt,pcov=curve_fit(distortion, pos_inImage,pos_inReality)
#print(popt)

