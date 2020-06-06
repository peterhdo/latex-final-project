import argparse
import os
import random
from shutil import copyfile


def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def get_images_in_dir(dir):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(dir) for f in filenames
            if os.path.splitext(f)[1] == '.png']


def get_unused_images(image_files, num_images_per_sequence, num_classes_folder_path):
    random.shuffle(image_files)
    used_images = set([f'{x.split("/")[-2]}/{x.split("/")[-1]}' for x in get_images_in_dir(num_classes_folder_path)])
    unused_images = []
    i = 0
    while len(unused_images) < num_images_per_sequence:
        image_path = image_files[i]
        symbol_image = f'{image_path.split("/")[-2]}/{image_path.split("/")[-1]}'
        if symbol_image not in used_images:
            unused_images.append(image_path)
        i += 1
    return unused_images


def create_new_sequence_name(num_classes_folder_path):
    subdirs = get_immediate_subdirectories(num_classes_folder_path)
    existing_seq_names = [x.split('/')[-1] for x in subdirs]
    max_seq_num = max([int(x.split('_')[-1]) for x in existing_seq_names]) if existing_seq_names else 0
    return f'seq_{str(max_seq_num + 1)}'


def create_new_sequence_folder(num_classes_folder_path, sequence_images):
    seq_name = create_new_sequence_name(num_classes_folder_path)
    sequence_folder = f'{num_classes_folder_path}/{seq_name}'
    for sequence_image in sequence_images:
        sequence_image_parts = sequence_image.split('/')
        symbol = sequence_image_parts[-2]
        png = sequence_image_parts[-1]
        make_folder(f'{sequence_folder}/{symbol}')
        copyfile(sequence_image, f'{sequence_folder}/{symbol}/{png}')


def generate_sequence(num_classes, num_classes_folder_path, images_per_sequence):
    datasets_dir_suffix = '' if num_classes == 959 else str(num_classes)
    datasets_dir = 'datasets{}/test/'.format(datasets_dir_suffix)
    image_files = get_images_in_dir(datasets_dir)
    sequence_images = get_unused_images(image_files, images_per_sequence, num_classes_folder_path)
    create_new_sequence_folder(num_classes_folder_path, sequence_images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--num_classes', type=int, required=True)
    parser.add_argument('-i', '--num_images_per_sequence', type=int, default=10)
    parser.add_argument('-s', '--num_sequences', type=int, default=1)
    args = parser.parse_args()
    make_folder('sequence_datasets')
    num_classes_folder_path = f'sequence_datasets/classes_{args.num_classes}'
    make_folder(num_classes_folder_path)
    for _ in range(args.num_sequences):
        generate_sequence(args.num_classes, num_classes_folder_path, args.num_images_per_sequence)


if __name__ == '__main__':
    main()
