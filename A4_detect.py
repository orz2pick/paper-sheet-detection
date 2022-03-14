import cv2
import numpy as np
from itertools import product 
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
from scipy import stats
def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(cov[y,x],y,x)
def adjust_gamma(imgs, gamma=1.0):
    #assert (len(imgs.shape)==4)  #4D arrays
    
    #assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
    return new_imgs

i=0
cam = cv2.VideoCapture(1)
#fig = plt.figure(figsize=(8,7))
#ax = plt.axes(), plt.title("Histogram 3D")
#plt.ion()
while 1:
    #plt.cla()
    t1=time.time()
    ret,frame = cam.read()
    gama_f = adjust_gamma(frame,gamma=2.5)
    cov = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV_FULL) # 初始图像是RGB格式，转换成BGR即可正常显示了
    img = frame
    
    width,height = img.shape[0],img.shape[1]
    cv2.namedWindow('houghline', 0)
    cv2.resizeWindow('houghline', width,height//2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    h,s,v = cv2.split(cov)
    gaus = cv2.GaussianBlur(h,(3,3),0)
    img_bil = cv2.bilateralFilter(cov,15,50,1)
 
    edges = cv2.Canny(cov, 50, 150, apertureSize=3)

    cv2.imshow("houghline",edges)
    #for i in range(3):
    #    print(i,min(min(row) for row in cov[:,:,i]),max(max(row) for row in cov[:,:,i]))
    
    tran = cv2.cvtColor(cov,cv2.COLOR_HSV2RGB_FULL)
    r,g,b = cv2.split(frame)
    img1 = np.uint8(cv2.merge((h,s,v)))
    img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2BGR_FULL)


    lower=np.array([160,87,30])
    upper=np.array([170,140,101])
 
    mask=cv2.inRange(cov,lower,upper)
    res=cv2.bitwise_and(frame,frame,mask=mask)

    #cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
    #cv2.imshow("h",h)
    #img_bil = cv2.pyrMeanShiftFiltering(img,15,50,1)
    edges2 = cv2.Canny(img, 50, 150, apertureSize=3)
    cv2.imshow("edges2",edges2)
    counters,_ = cv2.findContours(edges2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_0 = np.ones(img.shape)
    cv2.drawContours(img_0,counters,-1,(0,0,255))
    cv2.imshow('Conters',img_0)

    best_c = [[0,0]]
    if len(counters)>0:
        best_c=counters[0]
    for c in counters:
        if cv2.arcLength(c,True)>cv2.arcLength(best_c,True):
            best_c = c
    cv2.drawContours(frame,best_c,-1,(0,100,255))
    
    cv2.imshow("mask",frame)
    cv2.setMouseCallback("res",getpos)


    #print('time:',time.time()-t1)
    if cv2.waitKey(100) & 0xff==ord('q'):
        break
    #plt.pause(0.01)
#plt.ioff()
#plt.show()
cam.release()
cv2.destroyAllWindows()