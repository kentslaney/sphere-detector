# Spheroid depth test
## Install
```bash
git submodule update --init
sh assets/coremltools/scripts/build.sh --python=3.12
python -m venv env
env/bin/pip install -r requirements.txt
```

## Snippets
```bash
python -m src.depth-test
python -m src.depth-test.demo
python -m src.depth-test.export
```

```bash
( grep \[T\]ODO -r src && grep \[T\]ODO -A 99 README.md | tail -n +2 ) | cat -n
```

## TODOs
- `float16` inputs to avoid data format changes outside CoreML
- Figure out a better repo name
