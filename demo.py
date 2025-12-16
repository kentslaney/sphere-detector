import cv2
import numpy as np
from PIL import Image

import sys, pathlib
local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow('preview', frame)

    spin_key = cv2.waitKey(1) & 0xFF
    if spin_key == ord('q'):
        break
    elif spin_key == ord(' '):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Demo(Image.fromarray(frame_rgb))
        frame = cv2.cvtColor(np.array(im.cropped()), cv2.COLOR_RGB2BGR)

        _, bboxes = im.depth.binned().nominate()
        for bbox in bboxes.tolist():
            cv2.rectangle(frame, bbox[1::-1], bbox[3:1:-1], (0, 255, 0), 2)

        cv2.imshow('preview', frame)
        cv2.waitKey(0)

    try:
        cv2.getWindowProperty('preview', cv2.WND_PROP_VISIBLE)
    except cv2.error as e:
        print(e)
        break

cap.release()
cv2.destroyAllWindows()
