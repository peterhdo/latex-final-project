#!/usr/bin/env python3
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

ce_loss = torch.nn.CrossEntropyLoss(size_average=False)

# Constants to change if we changing the number of classes, dataset path, etc.
DATASET_BASE_PATH = '../datasets50'
NUM_OUTPUT_CLASSES = 50
MODEL_FILE = 'latex_vgg16bn.pt'


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


def evaluate(model, device, test_loader, type="Dev", top_k=0):
    """
    Evaluates the model on a given device using the specified test loader.
    You can specify top_k as some positive number if you're interested in
    accuracy based on the top_k ouputs.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += ce_loss(output, target)  # sum up batch loss
            # get the index of the max log-probability
            if top_k:
                # Add the number correct from this batch.
                topk_acc = accuracy(output, target, topk=[top_k])[0]
                pred_acc = topk_acc.item()
                curr_batch_size = data.size()[0]
                batch_correct = (curr_batch_size * pred_acc) / 100
                correct += batch_correct
            else:
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values
    of k
    """
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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ResNet')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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
                        help='how many batches to wait before logging '
                        'training status')
    parser.add_argument('--log-eval-interval', type=int, default=100,
                        metavar='N',
                        help='how many batches to wait before logging dev '
                        'status during training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model in the default '
                        'path')
    # To make our script more CLI friendly
    parser.add_argument('--num-classes', type=int,
                        metavar='N',
                        help='How many output classes to expect. If not '
                        'specified uses the constant NUM_OUTPUT_CLASSES.')
    parser.add_argument('--dataset-path',
                        help='Path to the dataset to use. If not specified'
                        ' uses the constant DATASET_BASE_PATH.')
    # Support testing top-N accuracy
    parser.add_argument('--test-model', action='store_true', default=False,
                        help='If we want to just test the model using top-k')
    parser.add_argument('--top_k', type=int, default=5, metavar='N',
                        help='How many of the top outputs in the softmax to'
                        ' consider for accuracy.')
    parser.add_argument('--load-model-file', 
                        help='Path to load the model file from.')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # Determine device, output classes and dataset path
    device = torch.device("cuda" if use_cuda else "cpu")
    out_classes = args.num_classes if args.num_classes else NUM_OUTPUT_CLASSES
    data_path = DATASET_BASE_PATH

    if args.dataset_path:
        data_path = args.dataset_path
        # Remove trailing slashes
        if data_path.endswith('/'):
            data_path = data_path[:-1]

    print('Using gpu: {}'.format(use_cuda))
    print('Number of output classes:{}'.format(out_classes))
    print('Using dataset path {}'.format(data_path))

    # You can use more num_workers if there's more memory on the machine
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('{}/train/'.format(data_path),
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406], std=[
                                         0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('{}/dev/'.format(data_path),
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406], std=[
                                         0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('{}/test/'.format(data_path),
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406], std=[
                                         0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = torch.hub.load('pytorch/vision:v0.6.0',
                           'vgg16_bn', pretrained=True).to(device)

    # Freeze training for all layers (since we're just appending a final layer)
    for param in model.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    # Add our layer with NUM_OUTPUT_CLASSES, or the specified class amount.
    features.extend([nn.Linear(num_features, out_classes)])
    features[-1].to(device)
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.test_model:
        # Just test the model assuming the model output file.
        # Use top-k in testing accuracy.
        # Load the weights
        model_file_path = MODEL_FILE
        if args.load_model_file:
            model_file_path = args.load_model_file
        model.load_state_dict(torch.load(model_file_path))
        evaluate(model, device, test_loader, "Test", top_k=args.top_k)
    else:
        # Train the model from scratch
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader,
                  optimizer, epoch, dev_loader)
            evaluate(model, device, train_loader, "Train")
            evaluate(model, device, dev_loader, "Dev")

        if args.save_model:
            torch.save(model.state_dict(), MODEL_FILE)

        evaluate(model, device, test_loader, "Test")


if __name__ == '__main__':
    main()
