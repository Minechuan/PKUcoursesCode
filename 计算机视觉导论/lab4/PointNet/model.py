from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    '''
        The feature extractor in PointNet, corresponding to the left MLP in the pipeline figure.
        Args:
        d: the dimension of the global feature, default is 1024.
        segmentation: whether to perform segmentation, default is True.
    '''
    def __init__(self, segmentation = False, d=1024):
        super(PointNetfeat, self).__init__()
        self.segmentation = segmentation
        self.d = d
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, d, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d)

    def forward(self, x):
        '''
            If segmentation == True
                return the concatenated global feature and local feature. # (B, d+64, N)
            If segmentation == False
                return the global feature, and the per point feature for cruciality visualization in question b). # (B, d), (B, N, d)
            Here, B is the batch size, N is the number of points, d is the dimension of the global feature.
        '''
        # x: (B, N, 3) -> (B, 3, N) for shared MLP by 1x1 conv
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        local_feat = x  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B, d, N)

        point_feat = x.transpose(2, 1)  # (B, N, d), for cruciality visualization
        global_feat = torch.max(x, 2, keepdim=False)[0]  # (B, d)

        if self.segmentation:
            n_pts = local_feat.size(2)
            global_expanded = global_feat.unsqueeze(2).repeat(1, 1, n_pts)  # (B, d, N)
            return torch.cat([local_feat, global_expanded], 1)  # (B, d+64, N)

        return global_feat, point_feat


class PointNetCls1024D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the middle right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()
        self.feat = PointNetfeat(segmentation=False, d=1024) # (B, d), (B, N, d)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=1024)
        '''
        x, point_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), point_feat

class PointNetCls256D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the upper right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()
        self.feat = PointNetfeat(segmentation=False, d=256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=256)
        '''
        x, point_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), point_feat


class PointNetSeg(nn.Module):
    '''
        The segmentation head in PointNet, corresponding to the lower right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k = 2):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.feat = PointNetfeat(segmentation=True, d=1024)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        '''
            Input:
                x: the input point cloud. # (B, N, 3)
            Output:
                the log softmax of the segmentation result. # (B, N, k)
        '''
        x = self.feat(x)  # (B, 1088, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # (B, k, N)
        x = x.transpose(2, 1).contiguous()  # (B, N, k)
        return F.log_softmax(x, dim=-1)
