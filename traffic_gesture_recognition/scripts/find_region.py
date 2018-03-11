#! /usr/bin/env python3

import argparse
import cv2 as cv
import numpy as np

refPt = []
curr_xy = (0,0)

def get_xy_onclick(event, x, y, flags, param):
    global refPt, curr_xy

    curr_xy = (x, y)

    if event == cv.EVENT_LBUTTONDOWN:
        refPt.append((x,y))

f_by_f = False
read_next = True
window_size = (640, 480)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="path of the file to be read")
    args = parser.parse_args()

    cap = cv.VideoCapture(args.filename)
    cv.namedWindow("frame_1")
    cv.setMouseCallback("frame_1", get_xy_onclick)

    while (cap.isOpened()):
        if read_next == True:
            ret, frame = cap.read()
            read_next = False

        height, width, channels = frame.shape
        image = np.copy(frame)
        image = cv.resize(image, window_size)
        for i, pt in enumerate(refPt):
            cv.circle(image, refPt[i], 5, (0, 255, 0), thickness=3)
            if i:
                cv.line(image, refPt[i-1], refPt[i], (255, 0, 0), thickness=2)
        if len(refPt):
            cv.line(image, refPt[-1], curr_xy, (0, 0, 0), thickness=1)
        cv.imshow('frame_1', image)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            refPt.pop()
        elif key == ord("n"):
            read_next = True
        elif key == ord("f"):
            f_by_f = True
        elif key == ord("c"):
            f_by_f = False

        if f_by_f == False:
            read_next = True

    cap.release()
    cv.destroyAllWindows()
    print([(x*1./window_size[0], y*1./window_size[1]) for (x,y) in refPt])
