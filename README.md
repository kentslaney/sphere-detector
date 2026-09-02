# Sphere Detector (Depth Test)
Not quite fast enough for real-time, full-resolution inference on an edge
device, but good at localization. Bounding boxes are expected to cover occluded
areas.

<img width="1280" height="640" alt="debug_frame_00000_ball_0" src="https://github.com/user-attachments/assets/36281dac-b940-490f-b110-159841f804ca" />

## Install
The submodule population can be skipped if the repo is cloned recursively.

```bash
git submodule update --init
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For execution in CUDA environments:

```bash
pip install -e ".[cuda]"
```

## Snippets
The main CLI will run on the reference images in
[assets/examples](./assets/examples) that were used for development. The demo
will use the default CV2 video input and, on MacOS, Continuity Camera will allow
previewing the behavior for an iPhone camera. Exporting is currently for Apple
Silicon deployment targets, but only depth data onwards is licensed under CC0,
in order to comply with the submodules' licenses.

With the demo in the foreground, use the G key to toggle real time bounding box
estimation, space to capture the current frame and display visualizations, or Q
to close either the current group of windows or the preview window.

```bash
python -m src.sphere_detector
python -m src.sphere_detector.demo
python -m src.sphere_detector.export
```

```bash
( grep \[T\]ODO -r src && grep \[T\]ODO -A 99 README.md | tail -n +2 ) | cat -n
```

## TODOs
- Interface internals for [tracking](https://github.com/kentslaney/h264events)
  (starting as closed source, ideally a separate repo)
- Verify behavior for a basketball net
- Support for a ball in a pitcher's hand (outline problems for ray casts)
- Support for airless tennis balls (depth drop-off problems)
- Move early NMS to uint1 in MIL and possibly uint16 in StableHLO
