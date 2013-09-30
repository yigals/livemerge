import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)

winName = "Movement Indicator"
diffWin = "Diff"
cv2.namedWindow(winName)
cv2.namedWindow(diffWin)

# Read and blur background image first:
time0 = time.time()
t_back = np.float32(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY))
while (time.time() < time0 + 0.25):
    print time.time()
    cv2.accumulateWeighted(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY), t_back, 0.04)
    t_back = cv2.GaussianBlur(t_back, (5, 5), 0)
t_back = cv2.convertScaleAbs(t_back)

imgRows, imgCols = np.shape(t_back)
cv2.moveWindow(winName, 0, 0)
cv2.moveWindow(diffWin, imgCols, 0)

while True:
    t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    time0 = time.time()
    diff = cv2.absdiff(t_back, t)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    cv2.imshow(diffWin, diff)
    ret, diff = cv2.threshold(diff, 117, 255, cv2.THRESH_BINARY)
    
    cv2.imshow(winName, diff)
    # cpdiff = diff.copy()
    # contours, hierarchy = cv2.findContours(cpdiff, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # cv2.drawContours(cpdiff, contours, -1, (0,255,0), 2)
    # cv2.imshow(winName, cpdiff)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
        cv2.destroyWindow(winName)
        break

print "Goodbye"
