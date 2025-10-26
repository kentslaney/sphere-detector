# imagenet-1k bounding boxes
import pathlib, sys
import tensorflow as tf
import tensorflow_datasets as tfds
import xml.etree.ElementTree as ET

local = pathlib.Path(__file__).parents[0]
sys.path.insert(0, str(local))
from wn12 import balls
from oop import Da2
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

dsfs = local / "assets" / "datasets"
dsfs.mkdir(parents=True, exist_ok=True)
ball_records = dsfs / "bndbox_balls.tfrecord"

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

model = Da2('vits')

balls_depth_records = dsfs / "depth_balls.tfrecord"
ballnt_depth_records = dsfs / "depth_ballnt.tfrecord"

def write_balls_depth_records():
    ds = filtered12()

    def _get_depth(image_tensor):
        # The model expects a numpy array.
        image_numpy = image_tensor.numpy()
        depth_numpy = model(image_numpy)
        return depth_numpy.astype('float32')

    def _process_with_depth(example):
        depth_tensor = tf.py_function(
            _get_depth, [example['image']], tf.float32)
        # The model output shape is not static, so we don't set it here.
        return {**example, 'depth': depth_tensor}

    ds_with_depth = ds.map(_process_with_depth)

    with tf.io.TFRecordWriter(str(balls_depth_records)) as writer:
        for example in ds_with_depth:
            feature = {
                'file_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['file_name'].numpy()])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example['image']).numpy()])),
                'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example['depth']).numpy()])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example['label']).numpy()])),
                'bbox': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(example['bbox'], [-1]).numpy()))
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())

def read_balls_depth_records():
    depth_feature_description = {
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([17 * 4], tf.int64)
    }

    ds = tf.data.TFRecordDataset(str(balls_depth_records))

    def _parse_depth_function(x):
        res = tf.io.parse_single_example(x, depth_feature_description)
        return {
            'file_name': res['file_name'],
            'image': tf.io.parse_tensor(res['image'], out_type=tf.uint8),
            'depth': tf.io.parse_tensor(res['depth'], out_type=tf.float32),
            'label': tf.io.parse_tensor(res['label'], out_type=tf.int64),
            'bbox': tf.reshape(tf.cast(res['bbox'], tf.int32), (17, 4))
        }

    return ds.map(_parse_depth_function)

# >>> sum(1 for _ in read_depth_records())
# 10382

def read_ballnt_depth_records():
    ballnt_depth_feature_description = {
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }

    ds = tf.data.TFRecordDataset(str(ballnt_depth_records))

    def _parse_ballnt_depth_function(x):
        res = tf.io.parse_single_example(x, ballnt_depth_feature_description)
        return {
            'file_name': res['file_name'],
            'image': tf.io.parse_tensor(res['image'], out_type=tf.uint8),
            'depth': tf.io.parse_tensor(res['depth'], out_type=tf.float32),
            'label': tf.io.parse_tensor(res['label'], out_type=tf.int64),
        }

    return ds.map(_parse_ballnt_depth_function)

def write_ballnt_depth_records():
    """
    Filters for images not in the 'balls' synsets, takes 10382 of them,
    calculates their depth maps, and writes them to ballnt_depth.tfrecord.
    """
    def _filter_not_balls(x):
        return tf.math.logical_not(tf.math.reduce_any(x['label'] == ids))

    ds = ds12['train'].filter(_filter_not_balls).take(10382)

    def _get_depth(image_tensor):
        # The model expects a numpy array.
        image_numpy = image_tensor.numpy()
        depth_numpy = model(image_numpy)
        return depth_numpy.astype('float32')

    def _process_with_depth(example):
        depth_tensor = tf.py_function(
            _get_depth, [example['image']], tf.float32)
        return {**example, 'depth': depth_tensor}

    ds_with_depth = ds.map(_process_with_depth)

    with tf.io.TFRecordWriter(str(ballnt_depth_records)) as writer:
        for i, example in enumerate(ds_with_depth):
            if (i + 1) % 1000 == 0:
                print(f"Writing record {i + 1}/10382 for ballnt_depth.tfrecord")
            feature = {
                'file_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['file_name'].numpy()])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example['image']).numpy()])),
                'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example['depth']).numpy()])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example['label']).numpy()])),
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())