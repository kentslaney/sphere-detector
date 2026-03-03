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

## TODOs
- coremltools: allow folding while_loop to const
- stablehlo_coreml: special case reduce_window for max_pool and sum (conv 1s)
- Figure out a better repo name

```bash
grep TODO -r src | cat -n
```
