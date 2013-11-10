# Based spiritually on "Emerging Images" by Niloy J. Mitra, Hung-Kuo Chu,
#    Tong-Yee Lee, Lior Wolf, Hezy Yeshurun, Daniel Cohen-Or.
# Parts of the code are blatantly stolen from opencv sample code.
# You are free to do whatever you like with this code, but nothing dirty ;)

import cv2
import numpy as np
import time
import os
import random


def rect_intersection(a, b):
    "returns intersection area of rectangles a, b"
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    rx = max(ax, bx)
    ry = max(ay, by)
    rw = min(ax+aw, bx+bw) - rx
    rh = min(ay+ah, by+bh) - ry
    return (rx, ry, rw, rh) if rw > 0 and rh > 0 else None


def safe_embed(t, f, target_pnt, mask=False):
    """
    Allows 'out of bounds' copies from f to t. Returns two rects - t and f.
    Assumes target_pnt makes sense - not too negative or large.
    Argument mask tells whether to mask t or overwrite it.
    """
    tsx, tsy = t.shape[:2]
    fsx, fsy = f.shape[:2]
    dx, dy = target_pnt
            
    tx, tdx = max(dx, 0), min(dx + fsx, tsx) # where the embedding starts and ends, x axis
    ty, tdy = max(0, dy), min(dy + fsy, tsy) # where the embedding starts and ends, y axis
    fx = max(0, -dx)
    fy = max(0, -dy)
    
    if mask:
        t[tx:tdx, ty:tdy] |= f[fx:fx + min(fsx, tdx - tx), fy:fy + min(fsy, tdy - ty)]
    else:
        t[tx:tdx, ty:tdy] = f[fx:fx + min(fsx, tdx - tx), fy:fy + min(fsy, tdy - ty)]
    

def safe_random_embed(t, f, mask=False):
    """
    Randomly embeds frame f in t. Uses safe_embed.
    """
    t_width, t_height = t.shape[:2] # frame shape
    f_width, f_height = f.shape[:2] # splat shape
    point = (np.random.randint(-f_width + 1, t_width), np.random.randint(-f_height + 1, t_height))
    safe_embed(t, f, point, mask)


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
        safe_random_embed(t, s, mask=True)

    
    def splat_mask(self, m):
        "Gets a mask and splats it"
        t = np.zeros(m.shape, np.uint8)
        for i in xrange(300):
            self._splat_once(t)
        
        return t & m


# for each contour in a frame:
#     if it lies in any rectangle:
#         for each in random [3-8]:
#             randomly erode/dilate
#             randomly rotate
#             randomly paste result
def c_p_p(frame, rects):
    "randomly copy-perturb-paste contour-contents of a rect"
    cpframe = frame.copy()
    all_contours, _ = cv2.findContours(cpframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_brects = []
    for c in all_contours:
        brect = cv2.boundingRect(c)
        for rect in rects:
            if rect_intersection(brect, rect):
                contours_brects.append((c, brect))
                break
    
    for c, brect in contours_brects:
        cpframe.fill(0)
        cv2.drawContours(cpframe, [c], 0, 255, -1)
        x, y, w, h = brect
        subframe = frame[x:x+w, y:y+h]
        # subframe = rand_e_d(subframe)
        # subframe = rand_rotate(subframe)
        safe_random_embed(frame, subframe, mask=True)
        
    return frame


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
        gray_diff = cv2.medianBlur(gray_diff, 5)
        ret, mask = cv2.threshold(gray_diff, thresh, 255, cv2.THRESH_BINARY)
        vis = s.splat_mask(mask) | s.splat_mask(~mask)
        
        timestamp = cv2.getTickCount() / cv2.getTickFrequency() # unsure why can't use time.time()...
        cv2.updateMotionHistory(mask, motion_history, timestamp, MHI_DURATION)
        seg_mask, seg_bounds = cv2.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)
        
        rects = [rect for rect in seg_bounds if rect[2] * rect[3] > 64 ** 2]
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(t, (x, y), (x+w, y+h), (0, 255, 0))
        vis |= c_p_p(mask, rects)
        cv2.imshow(diffWin, t)
        cv2.imshow(winName, vis)
        frames.append(cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR))
        t0 = t
        
        key = cv2.waitKey(10)
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