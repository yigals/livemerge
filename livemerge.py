# Based spiritually on "Emerging Images" by Niloy J. Mitra, Hung-Kuo Chu,
#    Tong-Yee Lee, Lior Wolf, Hezy Yeshurun, Daniel Cohen-Or.
# Parts of the code are blatantly stolen from opencv sample code.
# You are free to do whatever you like with this code, but nothing dirty ;)

import cv2
import numpy as np
import time

from splatter import Splatter
import perturber


def main():
    cam = cv2.VideoCapture(0)

    winName = "Live Emergence"
    captureInput = "Capture Input"
    cv2.namedWindow(winName)
    cv2.namedWindow(captureInput)
    imgHeight = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    imgWidth = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    cv2.moveWindow(winName, 0, 0)
    cv2.moveWindow(captureInput, imgWidth, 0)

    thresh = 20
    MHI_DURATION = 0.2
    MAX_TIME_DELTA = 0.1
    motion_history = np.zeros((imgHeight, imgWidth), np.float32)
    
    s = Splatter((imgHeight, imgWidth), 'splat_db')
    frames = []
    
    start_time = time.time()
    ret, t0 = cam.read() # First two frames
    ret, t = cam.read()
    while ret:
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
        elif key == ord('w'):
            thresh = min(255, thresh + 10)
        elif key == ord('e'):
            thresh = max(0, thresh - 10)
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