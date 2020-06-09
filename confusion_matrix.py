from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import time

ce_loss = torch.nn.CrossEntropyLoss(size_average=False)

# referencing https://github.com/KBaichoo/cs221/blob/master/scripts/test_and_cm.py
def cm(args, model, device, test_loader):
    class_to_idx = test_loader.dataset.class_to_idx
    labels = [None] * 50
    for (c, i) in class_to_idx.items():
        labels[i] = c

    model.eval()
    test_loss = 0
    correct = 0

    true_values = []
    predicted_values = []

    with torch.no_grad():
        for data, target in test_loader:
            # imshow(torchvision.utils.make_grid(data))
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += ce_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            _, predicted = torch.max(output, 1)

            true_values.append(labels[target[0]])
            predicted_values.append(labels[predicted[0]])

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print(true_values)

    print()

    print(predicted_values)

    cm = confusion_matrix(true_values, predicted_values, labels=labels)
    cm_pd = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(20, 18))

    sn.heatmap(cm_pd, annot=True)# font size
    plt.title('ResNet50 Confusion Matrix for 50 classes', fontsize=20)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    # plt.tight_layout()
    plt.savefig('resnet_50_confusion_matrix.png')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ResNet')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-eval-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging dev status during training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./datasets50/test/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((256,256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False).to(device)
    model.load_state_dict(torch.load('./resnet50_final_experiments/latex_resnet50_50.pt'))
    model.eval()

    cm(args, model, device, test_loader)

if __name__ == '__main__':
    main()
