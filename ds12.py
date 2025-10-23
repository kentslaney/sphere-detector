# imagenet-1k bounding boxes
import pathlib, sys
import tensorflow as tf
import tensorflow_datasets as tfds
import xml.etree.ElementTree as ET

local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from wn12 import balls
sys.path.pop(0)

ds12, info12 = tfds.load("imagenet2012", with_info=True)

ids = tf.constant(next(zip(*balls)), dtype=tf.int64)
data_base = pathlib.Path(info12.data_dir).parents[0]
bboxes = data_base / "ILSVRC2012_bbox_train_v2"
if not bboxes.is_dir():
    raise FileNotFoundError(f"missing extracted bounding boxes at {bboxes}")

parse = lambda path: ET.parse(path).getroot()
tagname = lambda tag, root: filter(lambda x: x.tag == tag, root)
bbox = ("xmin", "ymin", "xmax", "ymax")

def objects(root):
    out = [None] * 4
    for obj in tagname("object", root):
        for i, (tag, el) in enumerate(zip(bbox, next(tagname("bndbox", obj)))):
            assert el.tag == tag
            out[i] = int(el.text)
        yield tuple(out)

def packed(root):
    for a, b, c, d in objects(root):
        yield (a << 48) + (b << 32) + (c << 16) + d

# $ find ~/tensorflow_datasets/imagenet2012/ILSVRC2012_bbox_train_v2/ -type f |\
#       xargs grep -c '/object' | cut -d':' -f2 | sort -n | tail -1
# 17
# $ find ~/tensorflow_datasets/imagenet2012/ILSVRC2012_bbox_train_v2/ -type f |\
#       wc -l
# 544546

full_shape = (544546, 17, 4)
ball_samples = sum(len(list((bboxes / i).iterdir())) for _, i, _ in balls)
ball_shape = (ball_samples,) + full_shape[1:]

feature_description = {
    'filename': tf.io.FixedLenFeature([], tf.string),
    'bbox': tf.io.RaggedFeature(tf.int64),
}

def write_example(writer, file):
    root = parse(file)
    feature = {
        'filename': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[file.name.split('.', 1)[0].encode()])),
        'bbox': tf.train.Feature(
            int64_list=tf.train.Int64List(value=packed(root)))}
    example = tf.train.Example(features=tf.train.Features(
        feature=feature))
    writer.write(example.SerializeToString())

all_records = data_base / "bndbox.tfrecord"
ball_records = data_base / "balls.tfrecord"

def write_all():
    with tf.io.TFRecordWriter(str(all_records)) as writer:
        for synset in bboxes.iterdir():
            for file in synset.iterdir():
                write_example(writer, file)

def write_balls():
    with tf.io.TFRecordWriter(str(ball_records)) as writer:
        for _, synset, _ in balls:
            for file in (bboxes / synset).iterdir():
                write_example(writer, file)

if not ball_records.exists():
    print("writing bounding box records")
    write_balls()

def read_tfrecord(records):
    ds = tf.data.TFRecordDataset(str(records))
    def _parse_function(x):
        res = tf.io.parse_single_example(x, feature_description)
        return {
            'filename': res['filename'],
            'bbox': tf.stack((
                tf.cast(tf.bitwise.right_shift(res['bbox'], 48), tf.int32),
                tf.cast(tf.bitwise.bitwise_and(
                    tf.bitwise.right_shift(res['bbox'], 32), 0xFFFF), tf.int32),
                tf.cast(tf.bitwise.bitwise_and(
                    tf.bitwise.right_shift(res['bbox'], 16), 0xFFFF), tf.int32),
                tf.cast(tf.bitwise.bitwise_and(res['bbox'], 0xFFFF), tf.int32)),
                axis=-1)
        }
    return ds.map(_parse_function)

def read_table(records, shape):
    ds = read_tfrecord(records)
    def _parse_filename(x):
        return x['filename']
    def _padded_bbox(x):
        pad = tf.constant([[0, 1], [0, 0]])
        return tf.pad(
                x['bbox'], pad * (shape[1] - tf.shape(x['bbox'])[0]),
                "CONSTANT", -1)
    keys = next(iter(ds.map(_parse_filename).batch(shape[0]).take(1)))
    vals = next(iter(ds.map(_padded_bbox).batch(shape[0]).take(1)))
    idx = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, tf.range(shape[0])), -1)
    return idx, vals

read_all = lambda: read_table(all_records, full_shape)
read_balls = lambda: read_table(ball_records, ball_shape)

def balls12():
    idx, vals = read_balls()
    def _mapping(ex):
        fname = tf.strings.split(ex['file_name'], tf.cast(b'.', tf.string))[0]
        i = idx[fname]
        return {**ex, 'bbox': tf.cond(
            i < 0, lambda: -tf.ones(full_shape[1:], tf.int32), lambda: vals[i])}
    return ds12['train'].map(_mapping)

def filtered12():
    def _filter(x):
        return tf.math.reduce_any(x['label'] == ids)
    return balls12().filter(_filter)
