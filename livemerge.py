# Based spiritually on "Emerging Images" by Niloy J. Mitra, Hung-Kuo Chu,
#    Tong-Yee Lee, Lior Wolf, Hezy Yeshurun, Daniel Cohen-Or.
# Parts of the code are blatantly stolen from opencv sample code.
# You are free to do whatever you like with this code, but nothing dirty ;)

import cv2
import numpy as np
import time

from utils import do_nothing
from splatter import Splatter
import perturber


winName = 'Live Emergence'
captureInput = 'Capture Input'
controlTrackbars = 'Control Trackbars'
ThreshTrackbar = 'Threshold'
MhiDurationTrackbar = 'MHI Duration * 10'
MaxTimeDeltaTrackbar = 'Max Time Delta * 10'
MinRectAreaTrackbar = 'sqrt(Min Rect Area)' # this value ** 2 is the minum area motion recognition


def setup(imgWidth, imgHeight):
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(captureInput, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(controlTrackbars, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(ThreshTrackbar, controlTrackbars, 20, 255, do_nothing)
    cv2.createTrackbar(MhiDurationTrackbar, controlTrackbars, 4, 30, do_nothing)
    cv2.createTrackbar(MaxTimeDeltaTrackbar, controlTrackbars, 7, 30, do_nothing)
    cv2.createTrackbar(MinRectAreaTrackbar, controlTrackbars, 64, 200, do_nothing)
    cv2.moveWindow(winName, 0, 0)
    cv2.moveWindow(captureInput, imgWidth + 10, 0)
    cv2.moveWindow(controlTrackbars, 0, imgHeight + 30)


def main():
    cam = cv2.VideoCapture(0)
    imgHeight = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    imgWidth = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    
    setup(imgWidth, imgHeight)

    motion_history = np.zeros((imgHeight, imgWidth), np.float32)
    
    s = Splatter((imgHeight, imgWidth), 'splat_db')
    frames = []
    
    start_time = time.time()
    ret, t0 = cam.read() # First two frames
    ret, t = cam.read()
    while ret:
        thresh = cv2.getTrackbarPos(ThreshTrackbar, controlTrackbars)
        mhi_duration = cv2.getTrackbarPos(MhiDurationTrackbar, controlTrackbars) / 10.0
        max_time_delta = cv2.getTrackbarPos(MaxTimeDeltaTrackbar, controlTrackbars) / 10.0
        sqrt_rect_area = cv2.getTrackbarPos(MinRectAreaTrackbar, controlTrackbars)
        
        diff = cv2.absdiff(t, t0)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.medianBlur(gray_diff, 5)
        ret, mask = cv2.threshold(gray_diff, thresh, 255, cv2.THRESH_BINARY)
        vis = s.splat_mask(mask) | s.splat_mask(~mask)
        
        timestamp = time.clock()
        cv2.updateMotionHistory(mask, motion_history, timestamp, mhi_duration)
        seg_mask, seg_bounds = cv2.segmentMotion(motion_history, timestamp, max_time_delta)
        
        rects = [rect for rect in seg_bounds if rect[2] * rect[3] > sqrt_rect_area ** 2]
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(t, (x, y), (x+w, y+h), (0, 255, 0))
        vis |= perturber.c_p_p(mask, rects)
        cv2.imshow(captureInput, t)
        cv2.imshow(winName, vis)
        frames.append(cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR))
        t0 = t
        ret, t = cam.read()
        
        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('s'):
            if cv2.waitKey() == ord('p'):
                cv2.imwrite('res.png', mask)
    
    cam.release()
    
    # video = cv2.VideoWriter(r"result.avi", -1, len(frames) / (time.time() - start_time), (imgWidth, imgHeight))
    # video = cv2.VideoWriter(r"result.avi", cv2.cv.CV_FOURCC('X','V','I','D'), len(frames) / (time.time() - start_time), (imgWidth, imgHeight))
    # for frame in frames: video.write(frame)
    # video.release()

if __name__ == '__main__':
    main()