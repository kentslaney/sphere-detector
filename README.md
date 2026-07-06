# Sphere Detector (Depth Test)
Not quite fast enough for real-time, full-resolution inference on an edge
device, but good at localization. Bounding boxes are expected to cover occluded
areas.

## Install
The coremltools compilation needs conda and zsh, but is only needed for the
Apple Silicon export pipeline. Otherwise, it can be skipped and removed from the
`requirements.txt`. Once `coremltools` releases a version newer than 9.0 to
PyPI, the submodule can be switched back to a prebuild wheel.

For execution in CUDA environments, `jax` has to be switched to `jax[cuda13]`.
At some point, this process might end up streamlined via a `pyproject.toml`.

The submodule population can be skipped if the repo is cloned recursively.

```bash
git submodule update --init
sh assets/coremltools/scripts/build.sh --python=3.12
python -m venv env
env/bin/pip install -r requirements.txt
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
python -m src.sphere-detector
python -m src.sphere-detector.demo
python -m src.sphere-detector.export
```

```bash
( grep \[T\]ODO -r src && grep \[T\]ODO -A 99 README.md | tail -n +2 ) | cat -n
```

## TODOs
Move early NMS to uint1 in MIL and possibly uint16 in StableHLO
