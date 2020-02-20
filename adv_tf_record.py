import tensorflow as tf
import numpy as np
import os
import horovod.tensorflow as hvd
from easydict import EasyDict as edict


IMAGE_SCALE = 2.0 / 255


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not (isinstance(value, list) or isinstance(value, np.ndarray)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not (isinstance(value, list) or isinstance(value, np.ndarray)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_num_records(filenames):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count

    print(filenames)
    nfile = len(filenames)
    return (count_records(filenames[0]) * (nfile - 1) +
            count_records(filenames[-1]))


def deserialize_att_record(record):
    feature_map = {
        'image/orig': tf.FixedLenFeature([], tf.string, ''),
        'image/float': tf.FixedLenFeature([], tf.string, ''),
        'image/floor': tf.FixedLenFeature([], tf.string, ''),
        'image/ceil': tf.FixedLenFeature([], tf.string, ''),
        'image/round': tf.FixedLenFeature([], tf.string, ''),
        'image/shape': tf.FixedLenFeature([3], tf.int64, [-1, -1, -1]),
        'diff/l0': tf.FixedLenFeature([1], tf.float32, 0),
        'diff/l1': tf.FixedLenFeature([1], tf.float32, 0),
        'diff/l2': tf.FixedLenFeature([1], tf.float32, 0),
        'diff/l_inf': tf.FixedLenFeature([1], tf.float32, 0),
        'label/adv': tf.FixedLenFeature([1], tf.int64, -1),
        'label/orig': tf.FixedLenFeature([1], tf.int64, -1),
        #'image/idx': tf.FixedLenFeature([1], tf.int64, -1)
        "label/pred": tf.FixedLenFeature([1000], tf.float32, np.zeros([1000], dtype=np.float32))
    }

    with tf.name_scope('deserialize_att_record'):
        obj = tf.parse_single_example(record, feature_map)

        # image_orig = tf.image.decode_png(obj['image/orig'], channels=3)
        # image_floor = tf.image.decode_png(obj['image/floor'], channels=3)
        # image_ceil = tf.image.decode_png(obj['image/ceil'], channels=3)
        # image_round = tf.image.decode_png(obj['image/round'], channels=3)

        # image_float = tf.reshape(tf.decode_raw(obj['image/float'], out_type=tf.float32), obj['image/shape'])

        image_orig = obj['image/orig']
        image_floor = obj['image/floor']
        image_ceil = obj['image/ceil']
        image_round = obj['image/round']
        image_float = obj['image/float']

        image_shape = obj['image/shape']

        # set image shape
        # image_orig.set_shape([image_size, image_size, None])
        # image_floor.set_shape([image_size, image_size, None])
        # image_ceil.set_shape([image_size, image_size, None])
        # image_round.set_shape([image_size, image_size, None])
        # image_float.set_shape([image_size, image_size, None])

        diff_l0 = tf.squeeze(obj['diff/l0'])
        diff_l1 = tf.squeeze(obj['diff/l1'])
        diff_l2 = tf.squeeze(obj['diff/l2'])
        diff_l_inf = tf.squeeze(obj['diff/l_inf'])

        label_adv = tf.squeeze(tf.cast(obj['label/adv'], tf.int32))
        label_orig = tf.squeeze(tf.cast(obj['label/orig'], tf.int32))
        image_pred = tf.squeeze(tf.cast(obj['label/pred'], tf.float32))

        return image_orig, image_float, image_floor, image_ceil, image_round, image_shape, \
            diff_l0, diff_l1, diff_l2, diff_l_inf, label_adv, label_orig, image_pred


def save_para_data_obj(self, trainer=None, image_size=64, save_dir=None):
    img_raw_data = tf.placeholder(tf.uint8, (image_size, image_size, 3))
    data_obj = edict({
        "self": self,
        "trainer": trainer,
        "image_size": image_size,
        "save_dir": save_dir,
        "op_img_raw_data": img_raw_data,
        "op_img_compressed": tf.image.encode_png(img_raw_data, compression=9)
    })
    return data_obj


def save_adv(image_orig, image_adv, label, target_label, logits, data_obj):
    print("********---------", data_obj.self.step)

    session = data_obj.trainer.sess
    print(hvd.rank(), hvd.local_rank())
    output_filename = 'rank-%.5d-%.5d' % (hvd.rank(), data_obj.self.step)
    output_file = os.path.join(data_obj.save_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    data_obj.self.step += 1
    count = len(label)

    out_image_orig = (np.transpose(image_orig, [0, 2, 3, 1]) + 1.0) / IMAGE_SCALE
    out_image_adv = (np.transpose(image_adv, [0, 2, 3, 1]) + 1.0) / IMAGE_SCALE
    out_image_orig = np.clip(out_image_orig, 0, 255).round()
    out_image_adv = np.clip(out_image_adv, 0, 255)

    out_image_adv_float = out_image_adv
    out_image_adv_floor = np.floor(out_image_adv)
    out_image_adv_ceil = np.ceil(out_image_adv)
    out_image_adv_round = np.round(out_image_adv)

    diff_data = (out_image_adv_round - out_image_orig) / 255
    diff_data = diff_data.reshape([count, -1])
    dist_l0 = np.linalg.norm(diff_data, 0, axis=1)
    dist_l1 = np.linalg.norm(diff_data, 1, axis=1)
    dist_l2 = np.linalg.norm(diff_data, 2, axis=1)
    dist_l_inf = np.linalg.norm(diff_data, np.inf, axis=1)

    # convert image to uint8 type
    out_image_adv_floor = out_image_adv_floor.astype(np.uint8)
    out_image_adv_ceil = out_image_adv_ceil.astype(np.uint8)
    out_image_adv_round = out_image_adv_round.astype(np.uint8)

    _img_compressed = data_obj.op_img_compressed
    _img_raw_data = data_obj.op_img_raw_data
    _image_size = data_obj.image_size

    print(np.shape(logits), logits.dtype, type(logits), type(logits[0]))
    # print(logits[0])
    for i in range(0, count):
        new_feature_map = {
            "image/orig": _bytes_feature(session.run(_img_compressed, feed_dict={_img_raw_data: out_image_orig[i]})),
            "image/float": _bytes_feature(out_image_adv_float[i].tobytes()),
            "image/floor": _bytes_feature(
                session.run(_img_compressed, feed_dict={_img_raw_data: out_image_adv_floor[i]})),
            "image/ceil": _bytes_feature(
                session.run(_img_compressed, feed_dict={_img_raw_data: out_image_adv_ceil[i]})),
            "image/round": _bytes_feature(
                session.run(_img_compressed, feed_dict={_img_raw_data: out_image_adv_round[i]})),
            "image/shape": _int64_feature([_image_size, _image_size, 3]),
            "diff/l0": _float_feature(dist_l0[i]),
            "diff/l1": _float_feature(dist_l1[i]),
            "diff/l2": _float_feature(dist_l2[i]),
            "diff/l_inf": _float_feature(dist_l_inf[i]),
            "label/adv": _int64_feature(target_label[i]),
            "label/orig": _int64_feature(label[i]),
            # "image/idx": _int64_feature(cur_adv_img_idx)
            "label/pred": _float_feature(logits[i])
        }

        example = tf.train.Example(features=tf.train.Features(feature=new_feature_map))
        writer.write(example.SerializeToString())

    writer.close()
    return image_orig, image_adv, label, target_label, logits
