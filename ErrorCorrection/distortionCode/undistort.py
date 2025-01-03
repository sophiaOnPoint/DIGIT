import numpy as np
import cv2

def undistort(imgPath,paraPath,savePath):
    #还原真实图像的步骤——
    #用参数求出新点的坐标
    #read the parameter
    f=open(paraPath)
    para_mean=f.readlines()
    para_mean=list(map(float,para_mean))
    #拟合
    img=cv2.imread(imgPath)
    h,w=img.shape[:2]
    # print('h')
    # print(h)
    # print('w')
    # print(w)
    mtx=np.array([[1,0,0],[0,1,0],[0,0,1]])
    #dist=np.array([para_mean[1],para_mean[2],para_mean[4],para_mean[5],[para_mean[3]]])
    dist=np.array([para_mean[1],para_mean[2],para_mean[4],para_mean[5]])
    #print('dist')
    #print(dist)
    #尝试distortion correction 修改为4位:newCamMatrix 天差地别，完全不同
    #尝试修改mtx:不知道有没有用，目前没看出来有用
    #newcameramtx,roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h),True)
    #这里priniciple 是否要设置为true还要再考虑
    #print('newcameramtx')
    #print(newcameramtx)
    #dst=cv2.undistort(img,mtx,dist,None,newcameramtx)
    dst = cv2.undistort(img, mtx, dist, None)
    #print(dst)
    #x,y,w,h=roi
    #print('roi')
    #print(roi)
    #dst=dst[y:y+h,x:x+w]
    # print('h')
    # print(h)
    # print('w')
    # print(w)
    # print(dst)
    cv2.imshow('dst',dst)
    #cv2.waitkey(0)
    cv2.imwrite(savePath,dst)

#
#undistort('p#_2.jpg','parameterNotebook.txt')