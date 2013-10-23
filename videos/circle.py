import cv2, numpy as np

radius = 10
color = 100
height = 100
width = 500

video = cv2.VideoWriter(r"circle.avi", cv2.cv.CV_FOURCC('X','V','I','D'), 25, (width,height))
# video = cv2.VideoWriter(r"circle.avi", 0x58564944, 25, (width,height))
frames = []

for i in xrange(0, width + radius):
    a = np.zeros((height, width, 3), np.uint8)
    cv2.circle(a, (i, height/2), 10, (255, 255, 255), -1)
    frames.append(a)

for frame in frames: video.write(frame)

video.release()