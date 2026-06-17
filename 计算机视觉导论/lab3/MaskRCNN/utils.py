import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torch.utils.tensorboard import SummaryWriter


def get_instance_segmentation_model(num_classes):

    backbone = torchvision.models.mobilenet_v2().features[:14]
 
    backbone.out_channels = 96
    anchor_generator = AnchorGenerator(sizes=((24, 32, 48, 64, 80),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))


    model = MaskRCNN(backbone,num_classes,
        rpn_anchor_generator=anchor_generator
        )

    return model


def plot_save_output(path, imgs, output):

    boxes = output["boxes"].detach().cpu().numpy()      # [N, 4]
    labels = output["labels"].detach().cpu().numpy()    # [N]
    scores = output["scores"].detach().cpu().numpy()    # [N]
    masks = output["masks"].detach().cpu().numpy()      # [N, 1, H, W]

    # 按 score 降序
    order = np.argsort(-scores)

    boxes = boxes[order]
    labels = labels[order]
    scores = scores[order]
    masks = masks[order]

    # 如果你后面还想用原来那种索引方式再转置
    masks = np.transpose(masks, (1, 2, 3, 0))  # [1, H, W, N]



    

    if (scores.shape[0]==0):
        print("no prediction, continue")
        return
    visualize_num = min(1, scores.shape[0])
    for i in range(visualize_num):
        plt.figure(figsize=(14*2, 14))    

        plt.subplot(1 ,2 ,1)
        plt.title("Input img")
        plt.axis("off")
        plt.imshow(imgs.permute(1,2,0).detach().numpy().astype(np.uint8))

        plt.subplot(1 ,2 ,2)
        plt.title("Mask "+str(scores[i]))
        plt.axis("off")
        plt.imshow(masks[0,:,:,i])

        plt.savefig(path.split(".")[0]+"_"+str(i)+".png")
        plt.close()


def plot_save_dataset(path, imgs, output):
    boxes = output["boxes"] 
    labels = output["labels"]
    masks = output["masks"].permute(1,2,0).detach().numpy()

    cols = 5
    num_objs = boxes.shape[0]

    plt.figure(figsize=(3*cols, 3))    

    plt.subplot(1 ,cols ,1)
    plt.title("Input img")
    plt.axis("off")
    plt.imshow(imgs.permute(1,2,0).detach().numpy().astype(np.uint8))

    for i in range(num_objs):
        plt.subplot(1, cols, i+2)
        plt.title("Mask"+str(i))
        plt.axis("off")
        plt.imshow(masks[:,:,i])


    plt.savefig(path)


class log_writer:
    def __init__(self, path, log_name ) -> None:

        output_path = os.path.join(path, log_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.writer = SummaryWriter(log_dir=output_path)


    def add_train_scalar(self, name, data, n):
        self.writer.add_scalar('train/' + name, data, n)


    def add_test_scalar(self,name, data, n):
        self.writer.add_scalar('test' + name, data, n)


def collate_fn(batch):
    return tuple(zip(*batch))