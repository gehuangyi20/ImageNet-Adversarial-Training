#!/usr/bin/python3
import os
import json
import csv
import argparse


def load_json(path):
    if not os.path.isfile(path):
        return None
    json_fp = open(path, "rb")
    json_str = json_fp.read()
    json_fp.close()
    # do not use edict since array type json is not a dictionary
    config = json.loads(json_str.decode())
    return config


parser = argparse.ArgumentParser(description='summarize model accuracy for different method')
parser.add_argument('--dir', help='accuracy save dir, required', type=str, default=None)
parser.add_argument('--output', help='output filename, required', type=str, default=None)
parser.add_argument('--config', help='config file, required', type=str, default="config.json")

args = parser.parse_args()

_dir = args.dir
output_file = args.output
config_filename = args.config

config = load_json(os.path.join(_dir, config_filename))

out_fp = open(os.path.join(_dir, output_file), 'wb')

out_fp.write("name\tatt_success\ttop1_err\ttop5_err\tcount\tdiff_l0_mean\tdiff_l0_std"
             "\tdiff_l1_mean\tdiff_l1_std\tdiff_l2_mean\tdiff_l2_std"
             "\tdiff_l_inf_mean\tdiff_l_inf_std\n".encode())

for cur_set in config:

    cur_fp = open(os.path.join(_dir, cur_set), 'r')
    cur_reader = csv.DictReader(cur_fp, dialect='excel-tab')
    fieldnames = cur_reader.fieldnames
    for row in cur_reader:
        line = '\t'.join([row[_key] for _key in fieldnames]) + '\n'
        out_fp.write(line.encode())

out_fp.close()
