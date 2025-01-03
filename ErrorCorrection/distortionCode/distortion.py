import math
import numpy as np
def distortion(pos,theta,k1,k2,p1,p2,k3):
    # print('pos:')
    # print(pos)
    x=pos[:,2]#position in real world
    y=pos[:,3]
    # print('x')
    # print(x)
    # print('y')
    # print(y)
    theta2=np.arctan2(y,x) + theta
    #x=np.cos(theta2)#position_in_real_world rotated
    #y=np.sin(theta2)
    r=np.sqrt(np.power(x,2)+np.power(y,2))
    x=r*np.cos(theta2)
    y=r*np.sin(theta2)
    #print('r')
    #print(r)
    # print(np.shape(r))
    # print('power')
    # print(np.shape(np.power(r,2)))
    coeff=k1*np.power(r,2)+k2*np.power(r,4)+k3*np.power(r,6)
    # print('coeff')
    # print(np.shape(coeff))
    #dx_r=x*(k1*np.power(r,2)+k2*np.power(r,4)+k3*np.power(r,6))
    dx_r=x*coeff
    #dy_r=y*(k1*np.power(r,2)+k2*np.power(r,4)+k3*np.power(r,6))
    dy_r=y*coeff
    # print('dx_r')
    # print(np.shape(dx_r))
    dx_t=(2*p1*x*y+p2*(np.power(r,2)+2*np.power(x,2)))
    dy_t=(p1*(np.power(r,2)+2*np.power(y,2))+2*p2*x*y)
    # print('dx_t')
    # print(np.shape(dx_t))
    x2=x+dx_r+dx_t
    y2=y+dy_r+dy_t
    xi=pos[:,0]#position in the picture;distorted;rotated
    yi=pos[:,1]
    # print('x2')
    # print(np.shape(x2))
    # print('xi')
    # #print(np.shape(xi))
    # print(xi)
    # print('x')
    # print(np.shape(x))
    # print('xi-x2')
    # print(xi-x2)
    # print('yi-y2')
    # print(yi-y2)
    delta=np.sqrt(np.power(xi-x2,2)+np.power(yi-y2,2))
    #print(delta)
    return delta