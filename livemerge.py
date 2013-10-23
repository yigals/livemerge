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
    s = Splatter((imgHeight, imgWidth), 'splat_db')
    video = cv2.VideoWriter(r"result.avi", cv2.cv.CV_FOURCC('X','V','I','D'), 25, (imgWidth, imgHeight))
    frames = []
    
    time0 = time.time()
    t0 = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    while True:
        # print 0.040 - (time.time() - time0), time.time(), time0
        # time.sleep(max(0, 0.040 - (time.time() - time0)))
        # time0 = time.time()
        t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(t, t0)
        diff = cv2.medianBlur(diff, 3)
        cv2.imshow(diffWin, diff)
        ret, diff = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        diff = s.splat_mask(diff) | s.splat_mask(~diff)
        
        cv2.imshow(winName, diff)
        frames.append(cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR))
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

    for frame in frames: video.write(frame)

    video.release()
if __name__ == '__main__':
    main()