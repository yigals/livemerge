import numpy, cv2
import livemerge

m = cv2.imread(r'c:\projects\livemerge\samples\mask.png')
c = cv2.VideoCapture(r'c:/Projects/LivEmerge/videos/circle.avi')
winName = "blat"
cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)

s = livemerge.Splatter(r'c:\projects\livemerge\splat_db')

def t_splat_once():
    while True:
        s._splat_once(m)
        cv2.imshow(winName, m)
        if cv2.waitKey(500) == ord('q'):
            cv2.destroyAllWindows()
            break

def t_splat_mask():
    ret = s.splat_mask(m)
    cv2.imshow(winName, ret)
    cv2.waitKey(100)
    
def t_splat_video():
    ret, frame = c.read()
    while ret:
        im = s.splat_mask(frame) | s.splat_mask(~frame)
        cv2.imshow(winName, im)
        ret, frame = c.read()
        if (cv2.waitKey(20) == ord('s')): cv2.waitKey(-1)
        
if __name__ == "__main__":
    t_splat_video()
    cv2.waitKey(5000)