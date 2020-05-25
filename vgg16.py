from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

ce_loss = torch.nn.CrossEntropyLoss(size_average=False)

# Constants to change if we changing the number of classes, dataset path, etc.
DATASET_BASE_PATH = './datasets50'
NUM_OUTPUT_CLASSES = 50

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
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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

    # TODO(kbaichoo): use more num_workers -- likely have memory on machines
    # to raise it up for 2,4... might make mini-batch processing better
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('{}/train/'.format(DATASET_BASE_PATH),
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('{}/dev/'.format(DATASET_BASE_PATH),
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('{}/test/'.format(DATASET_BASE_PATH),
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # TODO(kbaichoo): modify the model (so it's vgg + vgg we want)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True).to(device)
    
    # Freeze training for all layers (since we're just appending a final layer)
    for param in model.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    #num_features = model.classifier[6].in_features
    num_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    # Add our layer with NUM_OUTPUT_CLASSES 
    features.extend([nn.Linear(num_features, NUM_OUTPUT_CLASSES)])
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, dev_loader)
        evaluate(model, device, train_loader, "Train")
        evaluate(model, device, dev_loader, "Dev")

    if args.save_model:
        torch.save(model.state_dict(), "latex_vgg16bn.pt")

    evaluate(model, device, test_loader, "Test")


if __name__ == '__main__':
    main()
