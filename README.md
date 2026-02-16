# Spheroid depth test
## Install
```bash
git submodule update --init
sh assets/coremltools/scripts/build.sh --python=3.12
python -m venv env
env/bin/pip install -r requirements.txt
```

## TODOs
```bash
grep TODO *.py | cat -n
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
