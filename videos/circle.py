import cv2, numpy as np

radius = 10
color = 100
height = 100
width = 500

XVID_FOURCC = int("".join("%x" % ord(c) for c in 'XVID')
video = cv2.VideoWriter(r"circle.avi", XVID_FOURCC, 16), 25, (width,height))
# video = cv2.VideoWriter(r"circle.avi", 0x58564944, 25, (width,height))
frames = []

for i in xrange(0, width + radius):
    a = np.zeros((height, width, 3), np.uint8)
    cv2.circle(a, (i, height/2), 10, (255, 255, 255), -1)
    frames.append(a)

for frame in frames: video.write(frame)

video.release()