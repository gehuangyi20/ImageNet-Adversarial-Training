import os
import argparse
import tensorflow as tf
import numpy as np
from adv_tf_record import get_num_records, deserialize_att_record, stat_pred

os.environ['CUDA_VISIBLE_DEVICES'] = ''

parser = argparse.ArgumentParser(description='concat generated adv examples')
parser.add_argument('--dir', help='accuracy save dir, required', type=str, default=None)
parser.add_argument('--output', help='output filename', type=str, default=None)
parser.add_argument('--batch_size', help='image batch size: default 256', type=int, default=256)

args = parser.parse_args()
data_dir = args.dir
batch_size = args.batch_size
output = args.output

filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, 'tf-*')))
num_samples = get_num_records(filenames)

ds = tf.data.Dataset.from_tensor_slices(filenames)
ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=1, block_length=1)
ds = ds.map(deserialize_att_record, num_parallel_calls=1)

ds = ds.batch(batch_size)

iterator = ds.make_initializable_iterator()
next_element = iterator.get_next()
sess = tf.Session()
sess.run(iterator.initializer)

att_success = 0
top1_err = 0
top5_err = 0
count = 0

diff_l0 = []
diff_l1 = []
diff_l2 = []
diff_l_inf = []

for i in range(0, num_samples, batch_size):
    image_orig, image_float, image_floor, image_ceil, image_round, image_shape, \
        cur_diff_l0, cur_diff_l1, cur_diff_l2, cur_diff_l_inf, label_adv, label_orig, image_pred = sess.run(next_element)

    cur_att_success, cur_top1_err, cur_top5_err = stat_pred(label_adv, label_orig, image_pred)
    att_success += np.sum(cur_att_success)
    top1_err += np.sum(cur_top1_err)
    top5_err += np.sum(cur_top5_err)
    count += len(image_orig)

    diff_l0.extend(cur_diff_l0)
    diff_l1.extend(cur_diff_l1)
    diff_l2.extend(cur_diff_l2)
    diff_l_inf.extend(cur_diff_l_inf)

att_success = att_success / count
top1_err = top1_err / count
top5_err = top5_err / count
print("att_success: ", att_success)
print("top1_err: ", top1_err)
print("top5_err: ", top5_err)
print("count:", count)

diff_l0_mean = np.mean(diff_l0)
diff_l1_mean = np.mean(diff_l1)
diff_l2_mean = np.mean(diff_l2)
diff_l_inf_mean = np.mean(diff_l_inf)

diff_l0_std = np.std(diff_l0)
diff_l1_std = np.std(diff_l1)
diff_l2_std = np.std(diff_l2)
diff_l_inf_std = np.std(diff_l_inf)

print("diff_l0: %10.4f±%10.4f" % (diff_l0_mean, diff_l0_std))
print("diff_l1: %10.4f±%10.4f" % (diff_l1_mean, diff_l1_std))
print("diff_l2: %10.4f±%10.4f" % (diff_l2_mean, diff_l2_std))
print("diff_l_inf: %10.4f±%10.4f" % (diff_l_inf_mean, diff_l_inf_std))

if not os.path.exists(os.path.dirname(output)):
    os.makedirs(os.path.dirname(output))

fp = open(output, 'wb')
fp.write("name\tatt_success\ttop1_err\ttop5_err\tcount\tdiff_l0_mean\tdiff_l0_std"
         "\tdiff_l1_mean\tdiff_l1_std\tdiff_l2_mean\tdiff_l2_std"
         "\tdiff_l_inf_mean\tdiff_l_inf_std\n".encode())
fp.write(("%s\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f"
          "\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" %
          (data_dir, att_success, top1_err, top5_err, count,
           diff_l0_mean, diff_l0_std, diff_l1_mean, diff_l1_std,
           diff_l2_mean, diff_l2_std, diff_l_inf_mean, diff_l_inf_std)).encode())
fp.close()
