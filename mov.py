import pathlib, sys
import numpy as np

from PIL import Image
local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from static import Raster
sys.path.pop(0)

# Raster(PIL.Image).density()

import cv2
import matplotlib.pyplot as plt

mov = local / "assets" / "examples" / "videos" / "IMG_0009.MOV"
cap = cv2.VideoCapture(str(mov))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set a fixed figure size to ensure consistent output frame dimensions.
dpi = 100
fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)

out = local / "cache" / "IMG_0009.mp4"
fourcc = cv2.VideoWriter_fourcc(*'X264') 
video = None 

idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    print(idx)
    
    # Convert numpy array from OpenCV to PIL Image
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.imshow(Raster(pil_frame).depth.density())
    fig.canvas.draw()

    buf, shape = fig.canvas.print_to_buffer()

    if video is None:
        video = cv2.VideoWriter(str(out), fourcc, 30, (shape[1], shape[0])) # Use a more standard framerate

    img = np.frombuffer(buf, dtype=np.uint8)
    img  = img.reshape(shape + (-1,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    video.write(img)

cap.release()
video.release()