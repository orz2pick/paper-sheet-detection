# this file is going to create a binary image of variance of rgb
from hashlib import new
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit
from skimage import io
import time
img = cv2.imread('p8.jpg')
h,w,t = img.shape
img_s = cv2.resize(img,(w//4,h//4))
img_s = cv2.GaussianBlur(img_s,(3,3),0)

#new_img = [[np.var(b) for b in a] for a in img_s]
#new_img = np.array(new_img)
@jit
def New_image_with_var(img):
    h,w,_ = img.shape
    new_img = np.zeros([h,w])
    g_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    for i in range(h):
        for j in range(w):
            new_img[i,j] = np.var(img[i,j])

    M = np.mean(new_img)        
    new_img =np.where(new_img<18,g_img,255-g_img) 
    return new_img

t1 = time.time()
new_img = New_image_with_var(img_s)
print(time.time()-t1,np.mean(new_img))
print(new_img.shape)
gray_image = np.uint8(new_img)
'''
thresh = cv2.Canny(gray_image, 10, 100)
#cv2.imshow('dst', dst)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
best_hull = contours[0]
for cnt in contours:
    if cv2.arcLength(cnt,False)>cv2.arcLength(best_hull,False):
        best_hull = cnt
#cv2.polylines(img, [best_hull, ], True, (0, 255, 0), 4) 
cv2.polylines(img, contours, True, (0, 0, 255), 2) 
print(len(contours))

'''
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#plt.imshow(thresh)
edged = cv2.Canny(gray_image,40,150)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)

edge_con = sorted(contours[0],key =np.sum,reverse = False)

rect = cv2.minAreaRect(contours[0]) 
box = np.int0(cv2.boxPoints(rect)) 
print(rect,box)
cv2.drawContours(img_s,[contours[0],],-1,(0,0,255),4)
#print(edge_con)
#cv2.circle(img_s,edge_con[0][0],10,(255,0,0),2)
cv2.drawContours(img_s,[box,],-1,(0,255,0),4)
#cv2.polylines(img_s, [box, ], True, (0, 0, 255), 2)  # red
cv2.imshow('box',img_s)
print(img_s.shape)

cv2.waitKey(0)
#plt.hist(new_img.flatten(),100)
#ax.plot_surface(x,y,new_img,cmap = cm.ocean)

#cv2.imshow('new',new_img)
#cv2.waitKey(0)