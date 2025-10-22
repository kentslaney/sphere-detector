# COCO segmentation masks
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds

def serialize_tensor_features(values):
    res = {}
    for k, value in values.items():
        serialized_nonscalar = tf.io.serialize_tensor(value)
        res[k] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized_nonscalar.numpy()]))
    return tf.train.Example(features=tf.train.Features(feature=res))

output_signature={
        "image": tf.TensorSpec((None, None, 3), dtype=tf.uint8),
        "mask": tf.TensorSpec(
            (None, None, None, 3), dtype=tf.uint8),
        "label": tf.TensorSpec((None,), dtype=tf.int64)}
parse_sig = {
        **output_signature, "mask": tf.TensorSpec(
            (None, None), dtype=tf.int32)}

sig = tf.function(input_signature=[tf.data.DatasetSpec(parse_sig)])

@sig
def tupled(data):
    def _tupled(x):
        return (x["image"], x["mask"])
    return data.map(_tupled)

@sig
def _identity(x):
    return x

def ref_coco(data_dir, as_supervised=False, split=None, **kw):
    def decode_fn(record_bytes):
        res = tf.io.parse_single_example(record_bytes, {
            k: tf.io.FixedLenFeature([], tf.string)
            for k in output_signature.keys()})
        return {
                k: tf.io.parse_tensor(v, parse_sig[k].dtype)
                for k, v in res.items()}
    def from_tfrecord(split, info):
        fname = f"ref_coco-{split}.tfrecord*"
        file_path = str(tf.io.gfile.join(info.data_dir, fname))
        file_path = tf.io.gfile.glob(file_path)
        if file_path:
            return tf.data.TFRecordDataset(file_path).take(
                    info.splits[split].num_examples).map(decode_fn)
    def split_tfrecord(split, iterator, info):
        total, shards = info.splits[split].num_examples, 10
        iterator = iter(iterator.as_numpy_iterator())
        def subset(i):
            if i == shards - 1:
                yield from iterator
                return
            for _ in range(total // shards * i, total // shards * (i + 1)):
                yield next(iterator)
        for i in tqdm(range(shards)):
            fname = f"ref_coco-{split}.tfrecord-{i}-of-{shards}"
            write_tfrecord(subset(i), fname, info)
        return from_tfrecord(split, info)
    def write_tfrecord(iterator, fname, info):
        file_path = tf.io.gfile.join(info.data_dir, fname)
        with tf.io.TFRecordWriter(str(file_path)) as fp:
            for kw in iterator:
                fp.write(serialize_tensor_features(kw).SerializeToString())
    def to_tfrecord(split, iterator, info):
        if split == "train":
            return split_tfrecord("train", iterator, info)
        write_tfrecord(iterator, f"ref_coco-{split}.tfrecord", info)
        return from_tfrecord(split, info)
    def body(data_source):
        def _generator():
            return (
                    {
                        "image": i["image"], "mask": i["objects"]["mask"],
                        "label": i["objects"]["label"]}
                    for i in data_source)
        info = data_source.dataset_info
        data = tf.data.Dataset.from_generator(
                _generator,
                output_signature=output_signature)
        def _mapping(x):
            return {**x, "mask": tf.transpose(
                    tf.keras.ops.any(x['mask'], -1), (1, 2, 0))}
        data = data.map(_mapping)
        return data, info
    data_source = tfds.data_source("ref_coco", split=split, data_dir=data_dir)
    mapping = tupled if as_supervised else _identity
    if isinstance(split, str):
        info = data_source.dataset_info
        res = from_tfrecord(split, info)
        if res is not None:
            return mapping(res), info
        return mapping(to_tfrecord(split, *body(data_source))), info
    else:
        res = {
                k: from_tfrecord(k, v.dataset_info)
                for k, v in data_source.items()}
        info = next(iter(data_source.values())).dataset_info
        if all(i is not None for i in res.values()):
            return {k: mapping(v) for k, v in res.items()}, info
        data = {k: body(v) for k, v in data_source.items()}
        data = {k: mapping(to_tfrecord(k, *v)) for k, v in data.items()}
        return data, info

data_dir = pathlib.Path("~/tensorflow_datasets").expanduser()
ds14, info = ref_coco(data_dir, split="train")
sports_ball = 33

def filtered12():
    def _filter(x):
        return tf.math.reduce_any(x['label'] == sports_ball)
    return ds14.filter(_filter)
