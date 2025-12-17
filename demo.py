import cv2
import torch
import threading
from collections import deque
import time
import numpy as np
from PIL import Image

import sys, pathlib
local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

# TODO: pipeline batches up to an acceptable latency
input_queue = deque(maxlen=1)
output_queue = deque(maxlen=1)
daemon_stop = threading.Event()

class Demo(Demo):
    def uncrop(self, coords):
        if self.target is None:
            return coords
        size = jnp.array(self.spec)
        unscale = jnp.max(self.shape[::-1] / size)
        scaled = jnp.int32(unscale * size)
        origin = (scaled - self.shape[::-1]) // 2
        return jnp.tile(origin, [1, 2]) + coords * unscale

def demo_model(arr):
    im = Demo(arr)
    _, bboxes = im.depth.binned().nominate()
    return im.uncrop(bboxes)

class PyTorchWorker(threading.Thread):
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model

    def run(self):
        daemon_stop.clear()
        # shouldn't be any CPU bottlenecks
        torch.set_num_threads(1)

        while not daemon_stop.is_set():
            try:
                # TODO: is there a good reason deque doesn't have blocking pop?
                frame = input_queue.pop()
            except IndexError:
                time.sleep(0.01)
                continue
            frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            results = self.model(Image.fromarray(frame_rgb))

            print(results)
            output_queue.append(results)

def pollnt(*titles):
    try:
        for title in titles:
            if not cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE):
                return True
    except cv2.error as e:
        return True
    return False

def rect(im, *bboxes, color=(0, 255, 0), thickness=1, **kw):
    for bbox in bboxes:
        bbox = bbox.tolist() if hasattr(bbox, 'dtype') else bbox
        cv2.rectangle(im, bbox[1::-1], bbox[3:1:-1], color, thickness, **kw)

cap = None
worker = PyTorchWorker(demo_model)

def main(count_bboxes=3, live_bboxes=3):
    global cap, worker
    if cap is not None:
        clean()
    cap = cv2.VideoCapture(0)

    queue = []
    def window(*titles):
        cv2.namedWindow("preview")
        queue.append(titles)
    window("preview")
    preview_bboxes = False

    while queue:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

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
        elif spin_key == ord('g'):
            if not daemon_stop.is_set():
                worker.start()
            # else:
            #     daemon_stop.set()

        if preview_bboxes:
            try:
                bboxes = output_queue.pop()
                pending, frame = frame, frame.copy()
                rect(frame, *bboxes)
            except IndexError:
                pending = frame
            input_queue.append(pending)

        it = enumerate(queue)
        for level, group in it:
            if pollnt(*group):
                for sibling in group:
                    try:
                        cv2.destroyWindow(sibling)
                    except cv2.error:
                        return True
                parent = level - 1
                for level, children in it:
                    for child in children:
                        cv2.destroyWindow(child)
                for i in range(level - parent):
                    queue.pop()

        cv2.imshow('preview', frame)

    clean()

def clean():
    global cap
    daemon_stop.set()
    cap.release()
    cap = None
    clear()

def clear():
    cv2.destroyAllWindows()

main()
