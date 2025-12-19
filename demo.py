import cv2
import torch
import threading
import collections
import time
import numpy as np
from PIL import Image

import sys, pathlib
local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

class BlockingDeque:
    def __init__(self, maxlen):
        self.deque = collections.deque(maxlen=maxlen)
        self.condition = threading.Condition()

    def append(self, item):
        with self.condition:
            self.deque.append(item)
            self.condition.notify()

    def pop(self):
        with self.condition:
            while not self.deque:
                self.condition.wait()
            return self.deque.popleft()

    def pop_nowait(self):
        with self.condition:
            if not self.deque:
                raise IndexError()
            return self.deque.popleft()

# TODO: pipeline batches up to an acceptable latency
input_queue = BlockingDeque(maxlen=1)
output_queue = BlockingDeque(maxlen=1)

class Demo(Demo):
    def uncrop(self, coords):
        if self.target is None:
            return coords
        size = jnp.array(self.spec)
        unscale = jnp.max(self.shape[::-1] / size)
        scaled = jnp.int32(unscale * size)
        origin = (scaled - self.shape[::-1]) // 2
        return jnp.int32(jnp.tile(origin, [1, 2]) + coords * unscale)

def demo_model(arr):
    im = Demo(arr)
    _, bboxes = im.depth.binned().nominate()
    return im.uncrop(bboxes)

class PyTorchWorker(threading.Thread):
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model

    def run(self):
        # shouldn't be any CPU bottlenecks
        torch.set_num_threads(1)

        while True:
            frame = input_queue.pop()
            if frame is None:
                return
            frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            results = self.model(Image.fromarray(frame_rgb))

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

    worker.start()

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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        elif spin_key == ord('g'):
            preview_bboxes = not preview_bboxes

        try:
            bboxes = output_queue.pop_nowait()
            pending, frame = frame, frame.copy()
            rect(frame, *bboxes)
        except IndexError:
            pending = frame
        if preview_bboxes:
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
    input_queue.append(None)
    cap.release()
    cap = None
    clear()

def clear():
    cv2.destroyAllWindows()

main()
