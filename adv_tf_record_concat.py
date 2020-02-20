import os
import argparse
import tensorflow as tf
from adv_tf_record import _bytes_feature, _float_feature, _int64_feature, get_num_records, deserialize_att_record

parser = argparse.ArgumentParser(description='concat generated adv examples')
parser.add_argument('--dir', help='accuracy save dir, required', type=str, default=None)
parser.add_argument('--rank', help='output filename, required', type=int, default=None)
# parser.add_argument('--image_size', help='image dimension: default 224', type=int, default=224)
parser.add_argument('--batch_size', help='image batch size: default 256', type=int, default=256)

args = parser.parse_args()
# image_size = args.image_size
data_dir = args.dir
batch_size = args.batch_size
rank = args.rank

filenames = []
num_samples = 0
for rank_i in range(rank):
    cur_filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, 'rank-%.5d-*' % rank_i)))
    cur_num_samples = get_num_records(cur_filenames)
    filenames.append(cur_filenames)
    num_samples += cur_num_samples

ds = tf.data.Dataset.from_tensor_slices(filenames)

ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=1, block_length=1)
# preproc_func = lambda record: deserialize_att_record(record, image_size)
ds = ds.map(deserialize_att_record, num_parallel_calls=1)

ds = ds.batch(batch_size)

iterator = ds.make_initializable_iterator()
next_element = iterator.get_next()
sess = tf.Session() #tf.keras.backend.get_session()
sess.run(iterator.initializer)

writer = None
batch_count = 0
out_idx = 0
for i in range(0, num_samples, batch_size):
    image_orig, image_float, image_floor, image_ceil, image_round, image_shape, \
        diff_l0, diff_l1, diff_l2, diff_l_inf, label_adv, label_orig, image_pred = sess.run(next_element)

    len(image_orig)
    if writer is None:
        writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, "tf-%.5d" % out_idx))
        out_idx += 1
    for image_i in range(len(image_orig)):
        # print(image_shape, image_shape.dtype)
        new_feature_map = {
            "image/orig": _bytes_feature(image_orig[image_i]),
            "image/float": _bytes_feature(image_float[image_i]),
            "image/floor": _bytes_feature(image_floor[image_i]),
                # session.run(_img_compressed, feed_dict={_img_raw_data: out_image_adv_floor[i]})),
            "image/ceil": _bytes_feature(image_ceil[image_i]),
                # session.run(_img_compressed, feed_dict={_img_raw_data: out_image_adv_ceil[i]})),
            "image/round": _bytes_feature(image_round[image_i]),
                # session.run(_img_compressed, feed_dict={_img_raw_data: out_image_adv_round[i]})),
            "image/shape": _int64_feature(image_shape[image_i]), # [image_size, image_size, 3]),
            "diff/l0": _float_feature(diff_l0[image_i]),
            "diff/l1": _float_feature(diff_l1[image_i]),
            "diff/l2": _float_feature(diff_l2[image_i]),
            "diff/l_inf": _float_feature(diff_l_inf[image_i]),
            "label/adv": _int64_feature(label_adv[image_i]),
            "label/orig": _int64_feature(label_orig[image_i]),
            # "image/idx": _int64_feature(cur_adv_img_idx)
            "label/pred": _float_feature(image_pred[image_i])
        }

        example = tf.train.Example(features=tf.train.Features(feature=new_feature_map))
        writer.write(example.SerializeToString())

    batch_count += 1

    if batch_count == 4:
        batch_count = 0
        writer.close()
        writer = None

if writer is not None:
    writer.close()
