import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters. You can not change
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--data-dir', type=str, default='./data', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='./prune_out/')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--mask-file', type=str, required='./mask_out/mask_values.txt', help='The text file containing the mask values')

# Hyper-parameters you can change
parser.add_argument('--threshold', type=float, default=0.25, help='the threshold for pruning')

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    # Step 1: create poisoned / clean test set
    trigger_info = torch.load('./trigger_info_foranp.th', map_location=device)

    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

    # Step 2: load model checkpoints and trigger info
    net = getattr(models, 'resnet18')(num_classes=10)
    checkpoint = 'badnetsmodel_foranp.th'
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Step 3: pruning
    mask_values = read_data(args.mask_file)
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    print('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    results = evaluate_by_threshold(
        net, mask_values, criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader
    )
    file_name = os.path.join(args.output_dir, 'pruning_by_{}.txt'.format(args.threshold))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        f.writelines(results)


def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


def evaluate_by_threshold(model, mask_values, criterion, clean_loader, poison_loader):
    results = []
    start = 0
    idx = start
    for idx in range(start, len(mask_values)):
        if float(mask_values[idx][2]) <= args.threshold:
            pruning(model, mask_values[idx])
            start += 1
        else:
            break
    layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
    cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
    po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
    print('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
        start, layer_name, neuron_idx, args.threshold, po_loss, po_acc, cl_loss, cl_acc))
    results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
        start, layer_name, neuron_idx, args.threshold, po_loss, po_acc, cl_loss, cl_acc))
    return results


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            if len(labels.shape) == 2:
                labels = labels.squeeze(1)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    main()

