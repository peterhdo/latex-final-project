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

# https://github.com/pytorch/examples/blob/d91adc972cef0083231d22bcc75b7aaa30961863/imagenet/main.py#L407
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# https://github.com/pytorch/examples/blob/d91adc972cef0083231d22bcc75b7aaa30961863/imagenet/main.py#L359
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def evaluate_topk(model, device, test_loader, type="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    top5 = AverageMeter('Acc@5', ':6.2f')
    top4 = AverageMeter('Acc@4', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += ce_loss(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            acc1 = accuracy(output, target, topk=(1,1))[1]
            acc2 = accuracy(output, target, topk=(1,2))[1]
            acc3 = accuracy(output, target, topk=(1,3))[1]
            acc4 = accuracy(output, target, topk=(1,4))[1]
            acc5 = accuracy(output, target, topk=(1,5))[1]

            top5.update(acc5[0], data.size(0))
            top4.update(acc4[0], data.size(0))
            top3.update(acc3[0], data.size(0))
            top2.update(acc2[0], data.size(0))
            top1.update(acc1[0], data.size(0))

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print(' * Acc@5 {top5.avg:.3f}'
          .format(top5=top5))

    print(' * Acc@4 {top4.avg:.3f}'
          .format(top4=top4))

    print(' * Acc@3 {top3.avg:.3f}'
          .format(top3=top3))

    print(' * Acc@2 {top2.avg:.3f}'
          .format(top2=top2))

    print(' * Acc@1 {top1.avg:.3f}'
          .format(top1=top1))

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

    for i in [50, 100, 150, 200, 250, 500, '']:
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('./datasets{}/test/'.format(i),
                                 transform=transforms.Compose([
                                     transforms.Grayscale(num_output_channels=3),
                                     transforms.Resize((256,256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        if i == '':
            i = 959

        model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False).to(device)
        model.load_state_dict(torch.load('./resnet50_final_experiments/latex_resnet50_{}.pt'.format(i)))
        model.eval()

        evaluate_topk(model, device, test_loader, type="Test")

if __name__ == '__main__':
    main()
