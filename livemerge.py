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
import utils

VERSION_FORMAT = '%(prog)s 1.0'

winName = 'Live Emergence'
controlTrackbars = 'Control Trackbars'
ThreshTrackbar = 'Threshold'
MhiDurationTrackbar = 'MHI Duration * 10'
MaxTimeDeltaTrackbar = 'Max Time Delta * 10'
MinSqrtRectAreaTrackbar = 'sqrt(Min Rect Area)' # this value ** 2 is the minum area motion recognition

DEFAULT_THRESH = 20
DEFAULT_MHI = 4
DEFAULT_TIME_DELTA = 7
DEFAULT_SQRT_RECT_AREA = 64


def setup(imgWidth, imgHeight):
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(controlTrackbars, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(ThreshTrackbar, controlTrackbars, DEFAULT_THRESH, 255, do_nothing)
    cv2.createTrackbar(MhiDurationTrackbar, controlTrackbars, DEFAULT_MHI, 30, do_nothing)
    cv2.createTrackbar(MaxTimeDeltaTrackbar, controlTrackbars, DEFAULT_TIME_DELTA, 30, do_nothing)
    cv2.createTrackbar(MinSqrtRectAreaTrackbar, controlTrackbars, DEFAULT_SQRT_RECT_AREA, 200, do_nothing)
    cv2.moveWindow(winName, 0, 0)
    cv2.moveWindow(controlTrackbars, 0, imgHeight + 30)


def main(args):
    if args.in_file:
        try: 
            inf = int(args.in_file)
        except ValueError: 
            inf = args.in_file
    else: 
        inf = 0
        
    cam = cv2.VideoCapture(inf)
    imgHeight = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    imgWidth = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    comb_vis = np.zeros((imgHeight, 2*imgWidth), np.uint8)
    
    setup(imgWidth, imgHeight)

    motion_history = np.zeros((imgHeight, imgWidth), np.float32)
    
    s = Splatter((imgHeight, imgWidth), 'splat_db')
    
    pause_time = 0
    start_time = time.time()
    ret, t0 = cam.read() # First two frames
    ret, t = cam.read()
    if args.flip:
        t0 = cv2.flip(t0, flipCode=1)
        t = cv2.flip(t, flipCode=1)

    if args.out_file:
        h, w = t.shape[:2]
        if not args.dont_show_capture: w *= 2
        video = cv2.VideoWriter(args.out_file, cv2.cv.CV_FOURCC('X','V','I','D'), 10, (w, h))
        # video = cv2.VideoWriter(args.out_file, -1, 10, (w, h))

    while ret:
        thresh = cv2.getTrackbarPos(ThreshTrackbar, controlTrackbars)
        mhi_duration = cv2.getTrackbarPos(MhiDurationTrackbar, controlTrackbars) / 10.0
        max_time_delta = cv2.getTrackbarPos(MaxTimeDeltaTrackbar, controlTrackbars) / 10.0
        sqrt_rect_area = cv2.getTrackbarPos(MinSqrtRectAreaTrackbar, controlTrackbars)
        
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
        
        if not args.dont_show_capture:
            vis = utils.combine_images(vis, t)
        
        cv2.imshow(winName, vis)
        if args.out_file:
            video.write(vis if len(vis.shape) == 3 else cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR))
        t0 = t
        ret, t = cam.read()
        if args.flip:
            t = cv2.flip(t, flipCode=1)
        key = cv2.waitKey(40)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('s'):
            if cv2.waitKey() == ord('p'):
                cv2.imwrite('res.png', mask)
    
    cam.release()

    if args.out_file:
        video.release()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LivEmerge\n'
        'Creating emerging images from video/webcam')
    parser.add_argument('-v', '--version', action='version', version=VERSION_FORMAT)
    parser.add_argument('-d', '--dont-show-capture', action="store_true", help="Show only algorithm output")
    parser.add_argument('-o', '--out-file', help="Write to output file")
    parser.add_argument('-f', '--flip', action="store_true", help="Flip images from camera (mine requires it...)")
    parser.add_argument('-i', '--in-file', help="If specified, input comes from video file and not cam")
    args = parser.parse_args()
    main(args)
