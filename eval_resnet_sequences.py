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

def evaluate_sequence(model, device, test_loader, type="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += ce_loss(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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

    for i in range(1,100):
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('./sequence_datasets/classes_200/seq_{}/'.format(i),
                                 transform=transforms.Compose([
                                     transforms.Grayscale(num_output_channels=3),
                                     transforms.Resize((256,256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False).to(device)
        model.load_state_dict(torch.load('./resnet50_final_experiments/latex_resnet50_200.pt'))
        model.eval()

        evaluate_sequence(model, device, test_loader, type="Test")

if __name__ == '__main__':
    main()
