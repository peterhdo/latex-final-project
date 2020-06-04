#!/usr/bin/env python
"""
Generates a subset of the dataset based on the number of subclasses you want.
Subclasses are generated based on the amount of data they have. 
So running this with --n 5 would give subsets with the top 5 most data.
"""
import argparse
import os
from distutils.dir_util import copy_tree
from multiprocessing import Pool, Value

processes_finished = Value('i', 0)
total_processes = 0


def get_largest_classes(n):
    file_path = './datasets/train/'
    class_sizes = {}
    for dir_path, _, file_names in os.walk(file_path):
        symbol = dir_path.split('/')[-1]
        class_sizes[symbol] = len(file_names)
    return [k for k, v in sorted(class_sizes.items(), key=lambda item: item[1], reverse=False)][3:n+3]


def copy_tree_process(args):
    global processes_finished

    dataset_type, c, new_class_subdir = args
    print('[{}] Start copying {}'.format(dataset_type, c))
    copy_tree('datasets/{}/{}'.format(dataset_type, c), new_class_subdir)
    with processes_finished.get_lock():
        processes_finished.value += 1
    print('[{}] Finished copying {}.  {}/{} done'.format(dataset_type, c, processes_finished.value, total_processes))


def create_datasets(num_classes):
    global total_processes
    total_processes = num_classes * 3

    p = Pool(100)
    classes = get_largest_classes(num_classes)
    new_datasets_dir = 'datasets{}'.format(str(num_classes))
    os.mkdir(new_datasets_dir)
    args = []
    for dataset_type in ['train', 'dev', 'test']:
        os.mkdir('{}/{}'.format(new_datasets_dir, dataset_type))
        for c in classes:
            new_class_subdir = '{}/{}/{}'.format(new_datasets_dir, dataset_type, c)
            os.mkdir(new_class_subdir)
            args.append((dataset_type, c, new_class_subdir))
    p.map(copy_tree_process, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    args = parser.parse_args()
    num_classes = args.n
    create_datasets(num_classes)


if __name__ == '__main__':
    main()
