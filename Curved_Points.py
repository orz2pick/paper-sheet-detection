#-*- coding: utf-8 -*- 
# This code is to find the 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import random
from skimage import io
import collections

img =  io.imread('After.png')
w,h,t = img.shape # x is width, y is height, x<y

#img_s = cv2.resize(img,(w//4,h//4))
#img_s = img



img_s = img
#img_s = cv2.bilateralFilter(img,5,100,15)               #双向过滤
#img_s = cv2.GaussianBlur(img_s,(5,5),10,10)


gray_image = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
fil = np.array([[-1/16.,5/16.,-1/16.],
                [5/16.,-1,5/16.],
                [-1/16.,5/16.,-1/16.]])
res = cv2.filter2D(gray_image,-1,fil)

edged = cv2.Canny(gray_image,35,80,100,3,True)
ep = np.where(edged==255)
x=ep[0]
y=ep[1]
#plt.scatter(x,y,1)
#plt.imshow(np.transpose(edged))
#lt.imshow(edged)
#plt.title("Edged")
#plt.show()

edge_num = 0
Rx= [ 1,1,1,1,1, 2,2,2,2,2,2,2              ]+[3]*9+[4]*11
Ry= [ 2,1,0,-1,-2,2,1,0,-1,-2,3,-3          ]+[i for i in range(-4,5,1)]+[i for i in range(-5,6,1)]
Lx= [ -1,-1,-1,-1,-1, -2,-2,-2,-2,-2,-2,-2, ]+[-3]*9+[-4]*11

Rx= [ 1,2,3,3,2,1,3,2,1,3,2,3,2]
Ry= [ 0,0,0,-1,-1,-1,1,1,1,2,-2,2,-2]
Lx= [ -1,-2,-3,-3,-2,-1,-3,-2,-1,-3,-2,-3,-2]


def Right_Has(x0,y0):
    for ix in range(len(Rx)):
        x=x0+Rx[ix]
        y=y0+Ry[ix]
        if x<w and y>0 and y<h:
            if edged[x,y]==255:
                return [x,y]
    return [0]
@jit
def Left_Has(x0,y0):
    for ix in range(len(Rx)):
        x=x0+Lx[ix]
        y=y0+Ry[ix]
        if x>0 and y>0 and y<h:
            if edged[x,y]==255:
                return [x,y]
    return [0]
for i in range(w):
    for j in range(h):
        if edged[i,j]==255:

            dx = collections.deque([i])
            dy = collections.deque([j])
            #d.append([i,j])
            edged[i,j]=0
            x=i
            y=j
            xl=i
            yl=j
            while len(Right_Has(x,y))==2:
                rx,ry = Right_Has(x,y)
                dx.append(rx)
                dy.append(ry)
                #d.append([rx,ry])
                edged[rx,ry]=0
                x=rx
                y=ry
            while len(Left_Has(xl,yl))==2:
                lx,ly = Left_Has(xl,yl)
                dx.appendleft(lx)
                dy.appendleft(ly)
                edged[lx,ly]=0
                xl=lx
                yl=ly
            dx=list(dx)
            dy=list(dy)
            if len(dx)>500: # Find a proper edge curve
                plt.plot(dx,dy)
                edge_num+=1
                print(edge_num,len(dx))
            elif edge_num>100:
                break
plt.show()
plt.imshow(edged)
plt.show()
print(edge_num)
