import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        # ----------TODO------------
        # define a network 
        # ----------TODO------------
        # input size: 3*32*32
        sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output size: 16*32*32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output size: 32*32*32
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output size: 32*16*16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output size: 64*16*16
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output size: 128*16*16
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output size: 128*8*8
            nn.Flatten(),
            # output size: 128*8*8=8192
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )
        self.sequential = sequential


    def forward(self, x):

        # ----------TODO------------
        # network forwarding 
        # ----------TODO------------
        x = self.sequential(x)
        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
