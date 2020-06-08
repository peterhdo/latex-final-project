from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

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

total_correct = 0
total_examples = 0 

def evaluate_sequence(model, device, test_loader, type="Test"):
    global total_correct, total_examples
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += ce_loss(output, target)  # sum up batch loss
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    total_examples += len(test_loader.dataset)
    total_correct += correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch VGG')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('Using gpu: {}'.format(use_cuda))

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           'vgg16_bn', pretrained=True).to(device)
    
    # Newly created modules have require_grad=True by default
    num_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    # Add our layer with NUM_OUTPUT_CLASSES, or the specified class amount.
    features.extend([nn.Linear(num_features, 200)])
    features[-1].to(device)
    model.classifier = nn.Sequential(*features)  # Replace the model classifier
    
    model.load_state_dict(torch.load('./latex_vgg16bn_200.pt'))

    for i in range(1, 101):
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('../sequence_datasets/classes_200/seq_{}/'.format(i),
                                 transform=transforms.Compose([
                                     transforms.Grayscale(
                                         num_output_channels=3),
                                     transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], std=[
                                             0.229, 0.224, 0.225])
                                 ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)


        evaluate_sequence(model, device, test_loader, type="Test")
    print('Total correct: {}, Total: {}'.format(total_correct, total_examples))


if __name__ == '__main__':
    main()
