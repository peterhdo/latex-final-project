# Our CS230 Project: Multi-Symbol LaTeX Conversion
This is our repo.

to generate particular dataset subsets use the `gen_dataset_subsets.py` script.

Our vgg model is in the vgg folder. Our resnet model is in the resnet folder.

## Sequence generator script usage
This script will generate image "sequences" (using the term loosely; these are really just lists of images)
```
python gen_sequence_dataset.py -c {num_classes} -i {num_images_per_sequence} -s {num_sequences_to_be_created}
```
Defaults for `-i` and `-s` are 10, and 1, respectively.

Example: `python gen_sequence_dataset.py -c 3 -i 5 -s 2`. This will create 2 new sequences (-s) using 3 classes (-c) with 5 images per sequence (-i). The sequences will be created under `sequence_datasets/classes_{c}` each named `seq_{x}` (`x` will be from 1 on, in order)
