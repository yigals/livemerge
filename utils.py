import numpy as np
import cv2


def do_nothing(*arg, **kw):
    pass


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
    
def combine_images(a, b, dst=None):
    """
    Returns the two images side by side. There's an optional dst argument to
    avoid repeated mallocs.
    Assumes images are of same height.
    Returns color image.
    """
    if len(a.shape) < 3:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if len(b.shape) < 3:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        
    ah, aw = a.shape[:2]
    _, bw = b.shape[:2] # Assuming same height
    if not dst:
        dst = np.zeros((ah, aw + bw, 3), np.uint8)
        
    dst[:ah, :aw] = a[:,:]
    dst[:ah, aw:aw+bw] = b[:,:]
    return dst
