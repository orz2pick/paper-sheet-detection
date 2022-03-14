from skimage import io
from skimage import data,segmentation
from skimage.segmentation import mark_boundaries
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte
from skimage import measure,draw 
from skimage.filters import roberts, sobel, scharr, prewitt
import numpy as np

img = io.imread('p6.jpg')
h,w,t = img.shape
img_s = cv2.resize(img,(w//3,h//3))
gray_image = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
img_z = np.zeros(gray_image.shape)
w,h = img_z.shape
#print(type(img_z[0,0,0]),type(img_s[0,0,0]))
hsv = cv2.cvtColor(img_s,cv2.COLOR_RGB2HSV_FULL)
gray_image = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
#for i in range(w):
#    for j in range(h):
#        img_z[i,j] = np.var()
#T=142
thresholds = threshold_multiotsu(gray_image, classes=3)
binary = np.digitize(gray_image, bins=thresholds)

binary = img_as_ubyte(binary)
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
i=4
labels = segmentation.slic(img_s,n_segments = i)
labels = np.uint8(labels)
ro = roberts(gray_image)
#plt.imshow(gray_image)
#plt.title("2 parts")
#plt.show()
print(labels,type(labels[0,0]))
print(gray_image,type(gray_image[0,0]),'type')
thresh = cv2.Canny(gray_image, 0, 100)
print(thresh,type(thresh[0,0]))
plt.imshow(thresh,'gray')
plt.title('t')
plt.show()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
best_hull = contours[0]
for cnt in contours:
    if cv2.arcLength(cnt,True)>cv2.arcLength(best_hull,True):
        best_hull = cnt
cv2.polylines(img_s, [best_hull, ], True, (0, 0, 255), 2)  # red
v0 = best_hull[0][0]
v1 = best_hull[1][0]
s = []
def cal_theta(V0,V1,V2):
    A = np.dot(np.array(V1)-np.array(V0),np.array(V2)-np.array(V1))
    B = np.linalg.norm(np.array(V1)-np.array(V0))*np.linalg.norm(np.array(V2)-np.array(V1))
    return A/B
for i in range(2,len(best_hull)):
    x,y = best_hull[i][0]
    theta = cal_theta(v0,v1,(x,y))
    img_s[y,x,:]=(0,255,255)
    if abs(theta)<0.8:
        img_s[y,x,:]=(0,255,0)
        img_s[v0[1],v0[0],:]=(122,0,0)
        img_s[v1[1],v1[0],:]=(255,100,0)
    s=s+[theta]

    print('x,y,t',x,y,theta,i,np.arccos(theta))
    v0 = v1.copy()
    v1 = [x,y]

#cv2.imshow('gray',labels)
gray_image = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
gray_image = np.float32(gray_image)
corners = cv2.goodFeaturesToTrack(gray_image,6,0.2,10)
corners = np.int0(corners)
dst = cv2.cornerHarris(gray_image, 2, 9, 0.04)
dst = cv2.dilate(dst, None)
cv2.imshow('dst', dst)
#img_s[dst > 0.01 * dst.max()] = [0, 0, 255]
for i in corners:
    x,y = i.ravel()
    cv2.circle(img_s,(x,y),4,255,-1)
#print(best_hull,len(best_hull))
#hulls = [cv2.convexHull(cnt) for cnt in contours]
#cv2.polylines(img_s, hulls, True, (0, 0, 255), 2)  # red
#cv2.drawContours(img_s, contours, -1, (255, 0, 0), 2) 
    #contours = measure.find_contours(labels, 1)
#print(contours)
#for n, contour in enumerate(contours):
#    plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
#plt.imshow(labels)
#plt.imshow(mark_boundaries(img,labels))

plt.imshow(img_s)
plt.title('$p$'+str(len(contours)))
plt.show()
    
plt.axis('off')
plt.show()