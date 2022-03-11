import cv2
import numpy as np
from itertools import product 
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
from scipy import stats

def show_i(img,s):
    w = img.shape[0]
    h = img.shape[1]
    cv2.namedWindow(s, 0)
#cv2.setWindowProperty('p1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(s, h//5,w//5)
    cv2.moveWindow(s,0,0)
    cv2.imshow(s,img)
    if cv2.waitKey(0) & 0xff==ord('q'):
        cv2.destroyAllWindows()
t1 = time.time()
img = cv2.imread("p2.jpg")
img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV_FULL)

print(img.shape)

w,h,_ = img.shape
img_flat = img.reshape((w*h,3))
img_flat = np.float32(img_flat)
CA = (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER,20,0.5)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(img_flat,2,None,CA,10,flags)
img_out = labels.reshape((w,h))
img_out.astype(np.int16)
print(type(img_out))
img_out.astype(np.uint8)
print(type(img_out))

plt.imshow(img_out,'gray')
plt.show()
## Cal the conters
#img = cv2.GaussianBlur(img,(3,3),0)
#img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV_FULL)
l = 'git'
edges = cv2.Canny(img_out,100, 200, apertureSize=3,L2gradient=True)
show_i(edges,'pP'+str(l))
counters,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,counters,-1,(0,0,255))
## Showing the img 
show_i(img,'p1')
show_i(edges,'p2')
print(time.time()-t1)
if cv2.waitKey(0) & 0xff==ord('q'):
    cv2.destroyAllWindows()