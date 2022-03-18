
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import random
img =  cv2.imread('p8.jpg')
h,w,t = img.shape
#img_s = cv2.resize(img,(w//4,h//4))
#img_s = img
img_s = cv2.bilateralFilter(img,5,100,15)
gray_image = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)


edged = cv2.Canny(gray_image,35,80,100,3,False)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)

edge_con = sorted(contours[0],key =np.sum,reverse = False)
idd = []
p = cv2.arcLength(contours[0], True)  
def dis_ori(x,y):
    return np.abs(x-w/2)+np.abs(y-h/2)
cors= [] 
for i in range(len(contours[0])):
    q = cv2.arcLength(contours[0][0:i],False)
    x,y = contours[0][i][0]
    Y = dis_ori(x,y)
    x,y = contours[0][(i+2)% len(contours[0])][0]
    Y1 = dis_ori(x,y)
    x,y = contours[0][(i-2)% len(contours[0])][0]
    Y2 = dis_ori(x,y)
    C = int(q/p*255)
    if Y>Y1 and Y>Y2:
        cv2.circle(img_s,contours[0][i][0],50,(256-C,256-C,C),3)
        print(i,i*100/len(contours[0]),x,y)
        cors.append(i)
cur_tran = np.zeros((210*3,297*3,3))
#cors[0] and cors[3] is top one
cur_tran = np.uint8(cur_tran)

@jit
def line_full():
    for H in range(210*3):
        for a in range(297*3):
            kx = H/630.
            ky = a/891.
            di = int(kx*(cors[1]-cors[0]))
            j=cors[3]-di
            x,y = contours[0][di+cors[0]][0]
            x1,y1 = contours[0][j][0]
            x2 = int(ky*x+(1-ky)*x1)
            y2 = int(ky*y+(1-ky)*y1)
            cur_tran[H,a,:]=img[y2,x2,:]
#line_full()
print(cur_tran.shape)
'''
def line_full(x,y,x1,y1,H):
    for a in range(2970):
        k = np.float(a)/2970.0
        x2 = int(k*x+(1-k)*x1)
        y2 = int(k*y+(1-k)*y1)
        cur_tran[H//10,a//10,:]=img[y2,x2,:]
for i in range(cors[0],cors[1]+1):
    if i%30==0:
        print(i,cors[1])
    j=cors[3]-i+cors[0]
    x,y = contours[0][i][0]
    x1,y1 = contours[0][j][0]
    line_full(x,y,x1,y1,i-cors[0])
'''



rect = cv2.minAreaRect(contours[0]) 
box = np.int0(cv2.boxPoints(rect)) 
approx = np.float32(box)
pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])
 
M = cv2.getPerspectiveTransform(approx, pts2)
print(M,"M")
print(rect,box)
dst = cv2.warpPerspective(img, M, (800,800))


cv2.drawContours(img_s,[contours[0],],-1,(0,0,255),3)
#print(edge_con)
#cv2.circle(img_s,edge_con[0][0],10,(255,0,0),2)
cv2.drawContours(img_s,[box,],-1,(0,255,0),1)
#cv2.polylines(img_s, [box, ], True, (0, 0, 255), 2)  # red
cv2.namedWindow('box',cv2.WINDOW_FREERATIO)

cv2.imshow('box',img_s)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('WithLines.png',img_s)
plt.imshow(img_s)
plt.show()
#cv2.imwrite('RuledSurface.png',cur_tran) 
cv2.imwrite('EdgeAndRectangle.png',img_s)