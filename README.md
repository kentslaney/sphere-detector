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
- Figure out a better repo name
- End of pipeline NMS via MIL op
- Live debugging tools
- Thresholding (possibly in tracking instead of detection)

```bash
grep TODO -r src | cat -n
```

## ImageNet-1K (130GB, bboxes seperate) 0.9% of 1.28M
- 429 baseball
- 430 basketball
- 522 croquet ball
- 574 golf ball
- 722 ping-pong ball
- 805 soccer ball
- 852 tennis ball
- 890 volleyball
