#!/usr/bin/env python
"""
Generates a subset of the dataset based on the number of subclasses you want.
Subclasses are generated based on the amount of data they have. 
So running this with --n 5 would give subsets with the top 5 most data.
"""
import argparse
import os
from distutils.dir_util import copy_tree


def get_largest_classes(n):
    file_path = './datasets/train/'
    class_sizes = {}
    for dir_path, _, file_names in os.walk(file_path):
        symbol = dir_path.split('/')[-1]
        class_sizes[symbol] = len(file_names)
    return [k for k, v in sorted(class_sizes.items(), key=lambda item: item[1], reverse=True)][:n]


def create_datasets(num_classes):
    classes = get_largest_classes(num_classes)
    new_datasets_dir = 'datasets{}'.format(str(num_classes))
    os.mkdir(new_datasets_dir)
    for dataset_type in ['train', 'dev', 'test']:
        os.mkdir('{}/{}'.format(new_datasets_dir, dataset_type))
        for c in classes:
            new_class_subdir = '{}/{}/{}'.format(new_datasets_dir, dataset_type, c)
            os.mkdir(new_class_subdir)
            copy_tree('datasets/{}/{}'.format(dataset_type, c), new_class_subdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    args = parser.parse_args()
    num_classes = args.n
    create_datasets(num_classes)


if __name__ == '__main__':
    main()
