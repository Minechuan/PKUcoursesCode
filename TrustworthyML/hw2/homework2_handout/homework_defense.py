from __future__ import print_function
import os
import argparse
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from models.resnet import *
import random
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--alp', default=0.8, type=float, metavar='alpha',
                    help='epoch 1: alpha*CE(x,y)+(1-alpha)*CE(adv,y)')

parser.add_argument('--lam', default=1.0, type=float, metavar='lambda',
                    help='epoch2: lambda')

args = parser.parse_args()


seed = args.seed

# alpha=0.70 lambda=0.20 natural_accuracy=0.5855 robustness=0.3609 score=0.6537
lam=0.60
alpha=0.70

random.seed(seed)                          # Python 原生的 random                      # NumPy
torch.manual_seed(seed)                    # CPU
torch.cuda.manual_seed(seed)               # 当前 GPU
torch.cuda.manual_seed_all(seed)           # 所有可见 GPU
np.random.seed(seed)                       # NumPy

torch.backends.cudnn.benchmark = False     # 关闭自动寻找最优算法
torch.backends.cudnn.deterministic = True  # 强制使用确定性算法

# settings

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("my_device is:",device)
# setup data loader
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,worker_init_fn=worker_init_fn, **kwargs)



testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,worker_init_fn=worker_init_fn, **kwargs)


def PGD(model,
            x_natural,
            y,
            optimizer,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10,
            device= torch.device("cuda")):
    # define KL-loss
    criterion = nn.CrossEntropyLoss(size_average=False)
    model.eval()
    x_adv = x_natural+0.001 * torch.randn(x_natural.shape).to(device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = criterion(model(x_adv),y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train() # change the model to train state 
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    adv_prob=F.softmax(logits)

    '''boosted cross entropy : BCE=-log(x'right)-log(1-max (P(x)|y!=label))'''

    true_class_prob = adv_prob.gather(1, y.unsqueeze(1)).squeeze()
    mask = torch.ones_like(adv_prob).scatter_(1, y.unsqueeze(1), 0) 
    other_class_prob = adv_prob * mask
    max_other_prob, _ = other_class_prob.max(dim=1)
    BCE_loss = -torch.log(true_class_prob) - torch.log(1 - max_other_prob)
    BCE_loss = BCE_loss.mean()  # Batch num
    return BCE_loss,adv_prob,logits
    ''' 
    ori
    loss = F.cross_entropy(logits, y)
    return loss
    '''
    


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        
        # calculate robust loss
        loss_PGD,adv_probs,adv_logits = PGD(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           device = device)
        
        logits = model(data)
        
        
        # print(loss_PGD.shape,loss_KL.shape)

        if epoch==1:
            loss=alpha*F.cross_entropy(logits,target)+(1-alpha)*F.cross_entropy(adv_logits,target)
        else:
            probs=F.softmax(logits)
            kl_div = F.kl_div(
                input=torch.log(adv_probs),  # log Q
                target=probs,                # P
                reduction='none',            # shape=(batch_size, num_classes)
                log_target=False
            ).sum(dim=1)  # shape=(batch_size,)

            true_class_prob = probs.gather(1, target.unsqueeze(1)).squeeze()
            weighted_kl = kl_div * (1 - true_class_prob)  # shape=(batch_size,)
            loss_KL = weighted_kl.mean()  # 最终取 batch 平均

            loss=loss_PGD+lam*loss_KL
        
        
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    import time
    start_time = time.time()
    model = ResNet18().to(device)


    def count_parameters(model, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    param = count_parameters(model)
    print(param/1000000)




    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')
        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
    end_time = time.time()

    ###################################
    # You should not change those codes
    ###################################

    def _pgd_whitebox(model,
                      X,
                      y,
                      epsilon=0.031,
                      num_steps=3,
                      step_size=0.0157):
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
        print('err pgd (white-box): ', err_pgd)
        return err, err_pgd

    def eval_adv_test_whitebox(model, device, test_loader):
        """
        evaluate model by white-box attack
        """
        model.eval()
        robust_err_total = 0
        natural_err_total = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = _pgd_whitebox(model, X, y)
            robust_err_total += err_robust
            natural_err_total += err_natural
        robust_acc = (len(testset)-robust_err_total)/len(testset)
        clean_acc = (len(testset)-natural_err_total)/len(testset)
        print('natural_accuracy: ', clean_acc)
        print('robustness: ', robust_acc)
    eval_adv_test_whitebox(model, device, test_loader)
    print(end_time - start_time)

    ###################################
    # You should not change those codes
    ###################################



if __name__ == '__main__':
    main()