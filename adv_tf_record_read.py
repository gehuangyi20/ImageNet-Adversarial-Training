import os
import argparse
import tensorflow as tf
from adv_tf_record import get_num_records, deserialize_att_record

parser = argparse.ArgumentParser(description='concat generated adv examples')
parser.add_argument('--dir', help='accuracy save dir, required', type=str, default=None)
parser.add_argument('--batch_size', help='image batch size: default 256', type=int, default=256)

args = parser.parse_args()
# image_size = args.image_size
data_dir = args.dir
batch_size = args.batch_size

filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, 'tf-*')))
num_samples = get_num_records(filenames)

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

image_orig, image_float, image_floor, image_ceil, image_round, image_shape, \
        diff_l0, diff_l1, diff_l2, diff_l_inf, label_adv, label_orig, image_pred = sess.run(next_element)

for image_i in range(len(image_orig)):
    print("l0: %10.5f l1: %10.5f l2: %10.5f l_inf: %10.5f " %
          (diff_l0[image_i], diff_l1[image_i], diff_l2[image_i], diff_l_inf[image_i]))
