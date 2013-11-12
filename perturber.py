import cv2
import numpy as np
import random

import utils


def rand_rotate(image):
    angle = np.random.randint(360)
    h, w = image.shape[:2]
    center = (w/2, h/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
    

E_D_funcs = [cv2.erode, cv2.dilate]
ELEMENTS = [cv2.getStructuringElement(cv2.MORPH_RECT, (2,3)),
                     cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)),
                     cv2.getStructuringElement(cv2.MORPH_RECT, (4,3)),
                     cv2.getStructuringElement(cv2.MORPH_RECT, (3,4)),
                     cv2.getStructuringElement(cv2.MORPH_RECT, (4,4)),
                    ]
def  rand_e_d(image):
    func = random.choice(E_D_funcs)
    element = random.choice(ELEMENTS)
    return func(image, element)


# for each contour in a frame:
#     if it lies in any given movement-rectangle:
#         5 times:
#             if the contour's bounding rectangle is large:
#                 randomly erode/dilate
#                 randomly rotate
#             randomly paste result
def c_p_p(frame, rects):
    "randomly copy-perturb-paste contour-contents of a rect"

    cpframe = frame.copy()
    all_contours, _ = cv2.findContours(cpframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_brects = []
    for c in all_contours:
        brect = cv2.boundingRect(c)
        for rect in rects:
            if utils.rect_intersection(brect, rect):
                contours_brects.append((c, brect))
                break
    
    for c, brect in contours_brects:
        cpframe.fill(0)
        cv2.drawContours(cpframe, [c], 0, 255, -1)
        x, y, w, h = brect
        for i in xrange(5):
            subframe = cpframe[y:y+h, x:x+w]
            if w * h > 150:
                subframe = rand_rotate(subframe)
                subframe = rand_e_d(subframe)
            utils.safe_random_embed(frame, subframe, mask=True)
        
    return frame