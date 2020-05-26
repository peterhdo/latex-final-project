import matplotlib.pyplot as plt
import numpy as np

train_epoch_loss = []
val_accuracy = []
train_accuracy = []

# Update these constants for generating the plots with other graphs.
MODEL_NAME = 'ResNet152'
INPUT_FILE = './resnet152_2.txt'
OUTPUT_IMAGE_FILES = 'milestone2_ResNet152_'

with open(INPUT_FILE) as fp:
    lines = fp.readlines()
    for line in lines:
        if line.startswith("Train Epoch"):
            end = line.split("Loss", 1)[1]
            loss = float(end.split(' ', 1)[1].split('\n')[0])
            train_epoch_loss.append(loss)
        if line.startswith("Dev set"):
            end = line.split("Accuracy", 1)[1]
            vals = end.split(' ', 2)[1].split('/')
            pct = int(vals[0]) / int(vals[1])
            val_accuracy.append(pct)
        if line.startswith("Train set"):
            end = line.split("Accuracy", 1)[1]
            vals = end.split(' ', 2)[1].split('/')
            pct = int(vals[0]) / int(vals[1])
            train_accuracy.append(pct)

val_acc_x = np.linspace(0, len(val_accuracy), len(val_accuracy))

val_accuracy = [v * 100 for v in val_accuracy]

plt.plot(val_acc_x, val_accuracy)
plt.title('{} Validation Accuracy'.format(MODEL_NAME))
plt.xlabel('Every 100 Batches')
plt.ylabel('Accuracy (Percent)')
plt.savefig(OUTPUT_IMAGE_FILES + 'validation_accuracy.png')
plt.show()

train_acc_x = np.linspace(0, len(train_accuracy), len(train_accuracy))

train_accuracy = [v * 100 for v in train_accuracy]

plt.plot(train_acc_x, train_accuracy)
plt.title('{} Training Accuracy'.format(MODEL_NAME))
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (Percent)')
plt.savefig(OUTPUT_IMAGE_FILES + 'train_accuracy.png')
plt.show()


# with outliers
# n_loss = len(train_epoch_loss)
#
# train_epoch_x = np.linspace(0, n_loss, n_loss)
#
# plt.plot(train_epoch_x, train_epoch_loss)
# plt.show()

# https://www.kite.com/python/answers/how-to-remove-outliers-from-a-numpy-array-in-python

train_epoch_loss = np.array(train_epoch_loss)

mean = np.mean(train_epoch_loss)
std = np.std(train_epoch_loss)
distance_from_mean = abs(train_epoch_loss - mean)
max_deviations = 2
no_outliers = train_epoch_loss[distance_from_mean < max_deviations * std]

n_loss = len(no_outliers)

train_epoch_x = np.linspace(0, n_loss, n_loss)

plt.plot(train_epoch_x, no_outliers)
plt.title('{} Training Loss'.format(MODEL_NAME))
plt.xlabel('Number of Batches')
plt.ylabel('Cross-Entropy Loss (per batch)')
plt.savefig(OUTPUT_IMAGE_FILES + 'train_loss.png')
plt.show()
