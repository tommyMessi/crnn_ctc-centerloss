import time
import os
import math

import numpy as np

from libs import utils
from libs.img_dataset import ImgDataset
from nets.crnn import CRNN
import shutil


def calculate_accuracy(predicts, labels):
    """
    :param predicts: encoded predict result
    :param labels: ground true label
    :return: accuracy
    """
    assert len(predicts) == len(labels)

    correct_count = 0
    for i, p_label in enumerate(predicts):
        if p_label == labels[i]:
            correct_count += 1

    acc = correct_count / len(predicts)
    return acc, correct_count


def calculate_edit_distance_mean(edit_distences):
    data = np.array(edit_distences)
    data = data[data != 0]
    if len(data) == 0:
        return 0
    return np.mean(data)

def round_up(n):
    print(n * 10 % 10)
    k = n * 10 % 10
    if k < 5:
        return int(n)
    else:
        return int(n) + 1
def validation(sess, feeds, fetches, dataset: ImgDataset, converter, result_dir, name,
               step=None, print_batch_info=False, copy_failed=False):
    """
    Save file name: {acc}_{step}.txt
    :param sess: tensorflow session
    :param model: crnn network
    :param result_dir:
    :param name: val, test, infer
    :return:
    """
    sess.run(dataset.init_op)

    img_paths = []
    predicts = []
    labels = []
    edit_distances = []
    total_batch_time = 0

    for batch in range(dataset.num_batches):
        # img_batch, label_batch, batch_labels, batch_img_paths = dataset.get_next_batch(sess)
        img_batch, label_batch, batch_labels, positions, batch_img_paths = dataset.get_next_batch(sess)
        image_batch_shape = img_batch.shape
        w = round_up(image_batch_shape[2] / 4)
        batch_start_time = time.time()

        positions_list = []
        for position_str in positions:
            list2 = [6941 for i in range(w)]
            position_list = str(position_str, encoding = "utf8").split(',')
            num_list_new = [int(x) for x in position_list]
            list2[:len(num_list_new)] = num_list_new

            positions_list.append(list2)
        con_labels_batch = np.array(positions_list)
        con_labels = con_labels_batch.reshape((-1))


        feed = {feeds['inputs']: img_batch,
                feeds['labels']: label_batch,
                feeds['con_labels']:con_labels,
                feeds['len_labels']:w,
                feeds['is_training']: False}

        log_prob, batch_predicts, edit_distance, batch_edit_distances, logits, decoded = sess.run(fetches, feed)

        batch_predicts = [converter.decode(p, CRNN.CTC_INVALID_INDEX) for p in batch_predicts]

        img_paths.extend(batch_img_paths)
        predicts.extend(batch_predicts)
        labels.extend(batch_labels)
        edit_distances.extend(batch_edit_distances)
        print('batch_predicts:',batch_predicts)
        print('batch_predicts:', batch_labels)


        acc, correct_count = calculate_accuracy(batch_predicts, batch_labels)
        batch_time = time.time() - batch_start_time
        total_batch_time += batch_time
        if print_batch_info:
            print("Batch [{}/{}] {:.03f}s accuracy: {:.03f} ({}/{}), edit_distance: {:.03f}"
                  .format(batch, dataset.num_batches, batch_time, acc, correct_count, dataset.batch_size,
                          edit_distance))

    acc, correct_count = calculate_accuracy(predicts, labels)
    edit_distance_mean = calculate_edit_distance_mean(edit_distances)
    acc_str = "Accuracy: {:.03f} ({}/{}), Average edit distance: {:.03f}, Average batch time: {:.03f}" \
        .format(acc, correct_count, dataset.size, edit_distance_mean, total_batch_time / dataset.num_batches)

    print(acc_str)

    save_dir = os.path.join(result_dir, name)
    utils.check_dir_exist(save_dir)
    if step is not None:
        file_path = os.path.join(save_dir, '%.3f_%d.txt' % (acc, step))
    else:
        file_path = os.path.join(save_dir, '%.3f.txt' % acc)

    print("Write result to %s" % file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, p_label in enumerate(predicts):
            t_label = labels[i]
            f.write("{:08d}\n".format(i))
            f.write("input:   {:17s} length: {}\n".format(t_label, len(t_label)))
            f.write("predict: {:17s} length: {}\n".format(p_label, len(p_label)))
            f.write("all match:  {}\n".format(1 if t_label == p_label else 0))
            f.write("edit distance:  {}\n".format(edit_distances[i]))
            f.write('-' * 30 + '\n')
        f.write(acc_str + "\n")

    # Copy image not all match to a dir
    if copy_failed:
        failed_infer_img_dir = file_path[:-4] + "_failed"
        if os.path.exists(failed_infer_img_dir) and os.path.isdir(failed_infer_img_dir):
            shutil.rmtree(failed_infer_img_dir)

        utils.check_dir_exist(failed_infer_img_dir)

        failed_image_indices = []
        for i, val in enumerate(edit_distances):
            if val != 0:
                failed_image_indices.append(i)

        for i in failed_image_indices:
            img_path = img_paths[i]
            img_name = img_path.split("/")[-1]
            dst_path = os.path.join(failed_infer_img_dir, img_name)
            shutil.copyfile(img_path, dst_path)

        failed_infer_result_file_path = os.path.join(failed_infer_img_dir, "result.txt")
        with open(failed_infer_result_file_path, 'w', encoding='utf-8') as f:
            for i in failed_image_indices:
                p_label = predicts[i]
                t_label = labels[i]
                f.write("{:08d}\n".format(i))
                f.write("input:   {:17s} length: {}\n".format(t_label, len(t_label)))
                f.write("predict: {:17s} length: {}\n".format(p_label, len(p_label)))
                f.write("edit distance:  {}\n".format(edit_distances[i]))
                f.write('-' * 30 + '\n')

    return acc, edit_distance_mean
