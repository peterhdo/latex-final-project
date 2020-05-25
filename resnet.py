from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

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

def train(args, model, device, train_loader, optimizer, epoch, dev_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = ce_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        if batch_idx % args.log_eval_interval == 0:
            evaluate(model, device, dev_loader, "Dev")


def evaluate(model, device, test_loader, type="Dev"):
    model.eval()
    test_loss = 0
    correct = 0
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += ce_loss(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            acc5 = accuracy(output, target, topk=(1,5))[1]
            top5.update(acc5[0], data.size(0))

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    import pdb
    pdb.set_trace()

    print(' * Acc@5 {top5.avg:.3f}'
          .format(top5=top5))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ResNet')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
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

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./datasets/train/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((256,256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./datasets/dev/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((256,256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./datasets/test/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((256,256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet152', pretrained=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, dev_loader)
        evaluate(model, device, train_loader, "Train")
        evaluate(model, device, dev_loader, "Dev")

    if args.save_model:
        torch.save(model.state_dict(), "latex_resnet.pt")

    evaluate(model, device, test_loader, "Test")


if __name__ == '__main__':
    main()
