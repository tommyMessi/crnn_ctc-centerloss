"""
Helper functions to train/test on dataset provided by https://github.com/senlinuc/caffe_ocr
Convert labels format and image name
"""
import argparse
import shutil
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from libs.utils import check_dir_exist


def load_chars(filepath):
    if not os.path.exists(filepath):
        print("Chars file not exists. %s" % filepath)
        exit(1)

    ret = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x[0] for x in lines]
        ret = ''.join(lines)
    return ret


def decode(chars, encoded_label):
    out = ''
    for c in encoded_label:
        out += chars[c]
    return out


def main(args):
    with open(args.labels_file, 'r') as f:
        lines = f.readlines()
        print('Total image count %d' % len(lines))
        chars = load_chars(args.chars_file)
        print('Chars %d' % len(chars))
        labels = []
        count = 0
        for line in lines:
            name = line[:line.index(' ')]
            label = line[line.index(' '):]
            label = [int(x) for x in label.strip().split(' ')]

            src_path = os.path.join(args.img_dir, name)
            dst_path = os.path.join(args.output_dir, '%08d.jpg' % count)
            try:
                shutil.copy(src_path, dst_path)
                labels.append(decode(chars, label))
                count += 1
                if count % 1000 == 0:
                    print(count)
            except:
                continue

        out_labels_path = os.path.join(args.output_dir, 'labels.txt')
        with open(out_labels_path, 'w') as f:
            for label in labels:
                f.write('%s\n' % label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', default='/home/cwq/data/virtualbox/shared/caffe_ocr/caffe_ocr/images')
    parser.add_argument('--chars_file', default='/home/cwq/data/virtualbox/shared/caffe_ocr/chars.txt')
    parser.add_argument('--labels_file', default='/home/cwq/data/virtualbox/shared/caffe_ocr/train.txt')
    parser.add_argument('--output_dir', default='/home/cwq/data/virtualbox/shared/caffe_ocr/train')

    args, _ = parser.parse_known_args()

    check_dir_exist(args.output_dir)

    main(args)
