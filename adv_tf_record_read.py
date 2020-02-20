import os
import argparse
import tensorflow as tf
import numpy as np
from scipy import misc
from adv_tf_record import get_num_records, preprocess_att_record, stat_pred

os.environ['CUDA_VISIBLE_DEVICES'] = ''

parser = argparse.ArgumentParser(description='concat generated adv examples')
parser.add_argument('--dir', help='accuracy save dir, required', type=str, default=None)
parser.add_argument('--image_size', help='image dimension: default 224', type=int, default=64)
parser.add_argument('--batch_size', help='image batch size: default 256', type=int, default=256)
parser.add_argument('--save', help='whether to save the image', type=str, default='no')

args = parser.parse_args()
image_size = args.image_size
data_dir = args.dir
batch_size = args.batch_size
save = args.save == 'yes'

filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, 'tf-*')))
num_samples = get_num_records(filenames)

ds = tf.data.Dataset.from_tensor_slices(filenames)
ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=1, block_length=1)
preproc_func = lambda record: preprocess_att_record(record, image_size)
ds = ds.map(preproc_func, num_parallel_calls=1)

ds = ds.batch(batch_size)

iterator = ds.make_initializable_iterator()
next_element = iterator.get_next()
sess = tf.Session()
sess.run(iterator.initializer)

image_orig, image_float, image_floor, image_ceil, image_round, image_shape, \
        diff_l0, diff_l1, diff_l2, diff_l_inf, label_adv, label_orig, image_pred = sess.run(next_element)

image_pred_top5 = image_pred.argsort(axis=1)[:, -5:]
image_pred_top1 = image_pred.argsort(axis=1)[:, -1:]
att_success, top1_err, top5_err = stat_pred(label_adv, label_orig, image_pred)

if save:
    image_diff = np.abs(image_orig - image_round)
    image_compare = np.concatenate((image_orig, image_round, image_diff), axis=2)
else:
    image_compare = None

count = len(image_orig)
for image_i in range(count):
    print("l0: %10.5f l1: %10.5f l2: %10.5f l_inf: %10.5f label_adv; %4.d label_orig; %4.d " %
          (diff_l0[image_i], diff_l1[image_i], diff_l2[image_i], diff_l_inf[image_i],
           label_adv[image_i], label_orig[image_i]), image_pred_top5[image_i],
          image_pred_top1[image_i], top5_err[image_i])

    if save:
        output_img = misc.toimage(image_compare[image_i])
        misc.imsave(os.path.join(data_dir, 'image_%.5d.png' % image_i), output_img)

print("att_success: ", np.sum(att_success) / count)
print("top1_err: ", np.sum(top1_err) / count)
print("top5_err: ", np.sum(top5_err) / count)
