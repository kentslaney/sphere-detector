# Sphere Detector (Depth Test)
## Install
```bash
git submodule update --init
sh assets/coremltools/scripts/build.sh --python=3.12
python -m venv env
env/bin/pip install -r requirements.txt
```

## Snippets
```bash
python -m src.sphere-detector
python -m src.sphere-detector.demo
python -m src.sphere-detector.export
```

```bash
( grep \[T\]ODO -r src && grep \[T\]ODO -A 99 README.md | tail -n +2 ) | cat -n
```

## TODOs
Move early NMS to uint8 to avoid size 2 strides with padding bits for bit tricks
