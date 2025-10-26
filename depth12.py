import pathlib
import tensorflow as tf

local = pathlib.Path(__file__).parents[0]
dsfs = local / "assets" / "datasets"
balls_depth_records = dsfs / "depth_balls.tfrecord"
ballnt_depth_records = dsfs / "depth_ballnt.tfrecord"

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