import cv2
import torch
import threading
import queue
import time
import numpy as np
from PIL import Image

import sys, pathlib
local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

# TODO: pipeline batches up to an acceptable latency
input_queue = queue.Queue(maxsize=1)
output_queue = queue.Queue(maxsize=1)

class PyTorchWorker(threading.Thread):
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self.stop = threading.Event()

    def run(self):
        # shouldn't be any CPU bottlenecks
        torch.set_num_threads(1)

        while not stop_event.is_set():
            frame = input_queue.get()
            # no eager io
            results = self.model(frame.copy())

            # Clear old result
            if not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    pass
            output_queue.put(results)

def pollnt(*titles):
    try:
        for title in titles:
            if not cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE):
                return True
    except cv2.error as e:
        return True
    return False

def rect(im, bbox, color=(0, 255, 0), thickness=1, **kw):
    bbox = bbox.tolist() if hasattr(bbox, 'dtype') else bbox
    return cv2.rectangle(im, bbox[1::-1], bbox[3:1:-1], color, thickness, **kw)

async def preview_dispatch():
    pass

cap = None

def main(count_bboxes=3, live_bboxes=3):
    global cap
    if cap is not None:
        clean()
    cap = cv2.VideoCapture(0)

    queue = []
    def window(*titles):
        cv2.namedWindow("preview")
        queue.append(titles)
    window("preview")

    while queue:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('preview', frame)

        spin_key = cv2.waitKey(1) & 0xFF
        if spin_key == ord('q'):
            for popped in queue.pop():
                cv2.destroyWindow(popped)
        elif spin_key == ord(' '):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Demo(Image.fromarray(frame_rgb))
            smol = cv2.cvtColor(np.array(im.cropped()), cv2.COLOR_RGB2BGR)

            _, bboxes = im.depth.binned().nominate(16)
            for bbox in bboxes.tolist():
                rect(smol, bbox)

            counts = np.array(im.depth.binned().counts)
            counts = np.uint8(counts / np.max(counts) * 255)
            counts = cv2.applyColorMap(counts, cv2.COLORMAP_JET)

            if count_bboxes:
                for bbox in bboxes[:count_bboxes].tolist():
                    rect(counts, bbox)

            opening = {"bboxes": smol, "counts": counts, "capture": frame}
            window(*opening.keys())
            for title, showing in opening.items():
                cv2.imshow(title, showing)
            while pollnt(*opening.keys()):
                sys.stdout.flush()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        elif spin_key == ord('f'):
            pass

        it = enumerate(queue)
        for level, group in it:
            if pollnt(*group):
                for sibling in group:
                    cv2.destroyWindow(sibling)
                parent = level - 1
                for level, children in it:
                    for child in children:
                        cv2.destroyWindow(child)
                for i in range(level - parent):
                    queue.pop()

    clean()

def clean():
    global cap
    cap.release()
    cap = None
    clear()

def clear():
    cv2.destroyAllWindows()

main()
