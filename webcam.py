import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from detect import *
sys.path.pop(0)

im1 = Demo.file(examples_dir / "BOX_0001.jpg", cache_dir / "box_depth1.npy")

def _main():
    import matplotlib.pyplot as plt
    for im in [im1]:
        im.draw_candidates(plt.subplots()[1])
        plt.show()

def main():
    import cv2
    import numpy as np
    for im in [im1]:
        frame = cv2.cvtColor(np.array(im.cropped()), cv2.COLOR_RGB2BGR)
        _, bboxes = im.depth.binned().nominate()
        for bbox in bboxes.tolist():
            cv2.rectangle(frame, bbox[1::-1], bbox[3:1:-1], (0, 255, 0), 2)
        cv2.imshow('preview', frame)

        while cv2.waitKey(1) & 0xFF != ord('q'):
            try:
                cv2.getWindowProperty('preview', cv2.WND_PROP_VISIBLE)
            except cv2.error as e:
                print(e)
                break
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
