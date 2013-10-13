import cv2
import numpy as np
import time
import os

class NoSplatsException(Exception):
    pass

class Splatter(object):

    splats = []
    def __init__(self, splats_dir='.'):
        for f in os.listdir(splats_dir):
            splat = cv2.imread(os.path.join(splats_dir, f))
            if splat is not None:
                self.splats.append(splat)
        if not self.splats:
            raise NoSplatsException(os.path.realpath(splats_dir))
                
    def _splat_once(self, t):
        s = np.random.choice(self.splats)
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
        for i in xrange(30):
            self._splat_once(t)
        
        return t & m


    

def main():
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

if __name__ == '__main__':
    main()