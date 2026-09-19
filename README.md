# Sphere Detector (Depth Test)
Not quite fast enough for real-time, full-resolution inference on an edge
device, but good at localization. Bounding boxes are expected to cover occluded
areas.

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
- Consider uint4 on StableHLO for early NMS
    - uint16 loses some of the memory benefits because of stride 2
    - consider letting it be a compiler's problem
    - justify the performance gains, which requires data quality to start with
- Debug `Bounds.metric` for full resolution
    - choose a sifting pass in the middle
    - cut out 75% of the scatters for a lower count at the same granularity
    - adjust the bounds stats to match
    - seems suspiciously close to looking at different level granularity?
- Use a more appropriate loss metric than DA2 [e07309d](https://github.com/kentslaney/sphere-lab/commit/e07309d029f32eb4b47bfd0c05951c638a0d35df)
    - (possibly prerequisite to the previous one)
    - Think about general-purpose CV methods if I'm going that direction
    - Depth at least has a well-defined answer
    - Don't encourage speculative data annotation
    - Depth is already pushing towards hardware-at-runtime
    - Don't reduce the portion of this pipeline that I can actually debug
    - Some of the accuracy is already going towards a scale factor
    - Limited scope requirements
    - Time is already removed, and a camera pan helps FoV calibration
    - Don't add state I can't debug either
    - Am I avoiding hardware differences I can't recreate too?
- Error bars on derived values
