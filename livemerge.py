# Based spiritually on "Emerging Images" by Niloy J. Mitra, Hung-Kuo Chu,
#    Tong-Yee Lee, Lior Wolf, Hezy Yeshurun, Daniel Cohen-Or.
# Parts of the code are blatantly stolen from opencv sample code.
# You are free to do whatever you like with this code, but nothing dirty ;)

import cv2
import numpy as np
import time
import os
import random

class NoSplatsException(Exception):
    pass

class Splatter(object):
    PERCENTAGE = 5.0 / 100 # Size of splat/image
    splats = []
    def __init__(self, imgShape=(480, 640), splats_dir='.'):
        shape_per = (int(imgShape[0] * self.PERCENTAGE), int(imgShape[1] * self.PERCENTAGE))
        for f in os.listdir(splats_dir):
            splat = cv2.imread(os.path.join(splats_dir, f), 0)
            if splat is not None:
                splat = cv2.resize(splat, shape_per, interpolation=cv2.cv.CV_INTER_NN)
                self.splats.append(splat)
        if not self.splats:
            raise NoSplatsException(os.path.realpath(splats_dir))
        s = random.choice(self.splats)
                
    def _splat_once(self, t):
        s = random.choice(self.splats)
        tsx, tsy = t.shape[:2]  # frame shape
        ssx, ssy = s.shape[:2] # splat shape
        xd, yd = np.random.randint(-ssx + 1, tsx), np.random.randint(-ssy + 1, tsy)
        
        tx, txd = max(xd, 0), min(xd + ssx, tsx) # where the splat starts and ends, x axis
        ty, tyd = max(0, yd), min(yd + ssy, tsy) # where the splat starts and ends, y axis
        sx = max(0, -xd)
        sy = max(0, -yd)

        t[tx : txd, ty : tyd] |= s[sx : sx + min(ssx, txd - tx), sy : sy + min(ssy, tyd - ty)]
    
    def splat_mask(self, m):
        "Gets a mask and splats it"
        t = np.zeros(m.shape, np.uint8)
        for i in xrange(300):
            self._splat_once(t)
        
        return t & m


def main():
    cam = cv2.VideoCapture(0)

    winName = "Movement Indicator"
    diffWin = "Diff"
    cv2.namedWindow(winName)
    cv2.namedWindow(diffWin)

    imgHeight = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    imgWidth = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    cv2.moveWindow(winName, 0, 0)
    cv2.moveWindow(diffWin, imgWidth, 0)
    thresh = 20
    MHI_DURATION = 0.2
    MAX_TIME_DELTA = 0.1
    s = Splatter((imgHeight, imgWidth), 'splat_db')
    frames = []
    
    start_time = time0 = time.time()
    _, t0 = cam.read()
    motion_history = np.zeros((imgHeight, imgWidth), np.float32)
    while True:
        _, t = cam.read()
        if time.time() > time0 + 1:
            print len(frames)
            time0 = time.time()
        diff = cv2.absdiff(t, t0)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.medianBlur(gray_diff, 3)
        ret, mask = cv2.threshold(gray_diff, thresh, 255, cv2.THRESH_BINARY)
        vis = s.splat_mask(mask) | s.splat_mask(~mask)
        
        timestamp = cv2.getTickCount() / cv2.getTickFrequency() # unsure why can't use time.time()...
        cv2.updateMotionHistory(mask, motion_history, timestamp, MHI_DURATION)
        seg_mask, seg_bounds = cv2.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)
        
        for rect in list(seg_bounds):
            x, y, w, h = rect
            area = w * h
            if area < 64 ** 2:
                continue
            cv2.rectangle(diff, (x, y), (x+w, y+h), (0, 255, 0))
        
        cv2.imshow(diffWin, diff)
        cv2.imshow(winName, vis)
        frames.append(cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR))
        t0 = t
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('w'):
            thresh = min(255, thresh + 10)
            print thresh
        elif key == ord('e'):
            thresh = max(0, thresh - 10)
            print thresh
        elif key == ord('s'):
            cv2.waitKey()
            print thresh
    
    cam.release()
    
    # video = cv2.VideoWriter(r"result.avi", -1, len(frames) / (time.time() - start_time), (imgWidth, imgHeight))
    video = cv2.VideoWriter(r"result.avi", cv2.cv.CV_FOURCC('X','V','I','D'), len(frames) / (time.time() - start_time), (imgWidth, imgHeight))
    for frame in frames: video.write(frame)
    video.release()

if __name__ == '__main__':
    main()