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
armpit -m src.sphere_detector.examples
python -m src.sphere_detector
python -m src.sphere_detector.demo
python -m src.sphere_detector.export
```

The default output of the first one is

```
name      i          x         y          r    n       edge    surface
------  ---  ---------  --------  ---------  ---  ---------  ---------  -----------
im4       1  265.794    187.528    15.8237    28   0.423339   1.11896   0.351783
          0  265.559    187.332    14.783     40   0.575819   1.13328   0.327653
          2  265.589    187.597    14.4725    41   0.709955   1.95171   0.225239
          4   37.6461   158.224   136.196     15   0.732362   4.21761   0.0795811
          3    7.81173   69.1768  229.695     15   1.39071    9.11673   0.0094753
          7   61.7366   404.946   114.066     20   2.36757   13.419     0.00112128
------  ---  ---------  --------  ---------  ---  ---------  ---------  -----------
im5       3  365.031    143.67     23.1789    51   2.7424     2.48964   0.0260912
          4  462.092    281.607    46.2809    33   2.24278    5.76529   0.0147754
          7  468.185    289.315    53.744     29   4.66265    7.07316   0.00085838
          2  502.579    289.395    29.565     25   6.57098    4.73281   0.000245842
          5  469.855    291.16     55.3061    30   6.0396     7.20144   0.000210358
          6  477.524    292.293    48.3479    26   6.65717    6.79513   0.000122991
          1  484.94     287.349    43.447     28  10.685      4.82205   4.04768e-06
          0  484.679    284.192    45.2754    32  12.4186     5.51695   6.0158e-07
------  ---  ---------  --------  ---------  ---  ---------  ---------  -----------
im7       3  267.798    223.113     9.42318   33   0.582021   0.842551  0.340544
          6  264.563    224.72     13.3825    24   0.563233   2.26377   0.206863
          4  491.5       78.0544    8.08513   49   1.72713    1.2647    0.103332
          0  253.87     195.277    18.4964    42   2.69773    2.04842   0.0301151
          5  112.269    100.593    37.9136    13   4.45508    2.91108   0.00262206
          7  251.25     199.888    24.3823    15   3.65106    9.88523   0.000784895
------  ---  ---------  --------  ---------  ---  ---------  ---------  -----------
im8       2  242.872    214.135    10.7732    36   0.233249   0.580265  0.532831
          0  242.784    214.136    10.8814    30   0.257668   0.55459   0.501219
          1  242.898    214.638    10.7503    50   0.555544   0.491923  0.421843
          3  242.913    214.236    11.036      9   0.172407   0.936114  0.261288
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
