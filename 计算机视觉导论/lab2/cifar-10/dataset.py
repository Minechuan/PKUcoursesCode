import os
import os.path
import numpy as np
import pickle
import random
import torch 
import torchvision.transforms as tfs
import torchvision.transforms.functional as TF
from PIL import Image


def _random_crop_with_reflect_padding(img, max_crop=5):
    """Random crop with max pixel number {max_crop} reflect padding in [0, max_crop].
    The image size is unchanged after this transformation.
    """
    padding = random.randint(0, max_crop)
    img = tfs.Pad(padding, padding_mode='reflect')(img)
    left = random.randint(0, 2*padding)
    top = random.randint(0, 2*padding)
    img = TF.crop(img, top, left, img.size[1]-2*padding, img.size[0]-2*padding)
    return img


def _random_hflip(img):
    return TF.hflip(img)


def _random_rotate(img, max_degree=30):
    angle = random.uniform(-max_degree, max_degree)
    return TF.rotate(img, angle)


def _random_brightness(img):
    factor = random.uniform(0.6, 1.4)
    return TF.adjust_brightness(img, factor)


def _random_contrast(img):
    factor = random.uniform(0.6, 1.4)
    return TF.adjust_contrast(img, factor)


def _random_saturation(img):
    factor = random.uniform(0.6, 1.4)
    return TF.adjust_saturation(img, factor)


def _random_hue(img):
    factor = random.uniform(-0.2, 0.2)
    return TF.adjust_hue(img, factor)

class CIFAR10(torch.utils.data.Dataset):
    """
        modified from `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """
    def __init__(self, train=True):
        super(CIFAR10, self).__init__()

        self.base_folder = '../datasets/cifar-10-batches-py'
        self.train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4','data_batch_5']
        self.test_list = ['test_batch']

        self.meta = {
            'filename': 'batches.meta',
            'key': 'label_names'
        }

        self.train = train  # training set or test set
        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in file_list:
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        # ------------TODO--------------
        # data augmentation
        # ------------TODO--------------
        # trick 1: position augmentation
        # each image is 32*32
        # random crop (0-5 pixels, repeatedly padding), random horizontal flip, random rotation(30 degrees))

        if self.train:
            type = random.randint(0, 7)
            
            # debug = random.randint(0, 2)
            # if debug == 0:
            #     type = 1
            # elif debug == 1:
            #     type = 6
            # else:
            #     type = 7


            if type == 0: # random crop with reflect padding
                img = _random_crop_with_reflect_padding(img.copy(),max_crop=5)
            if type == 1: # horizontal flip
                img = _random_hflip(img.copy())
            if type == 2: # random rotation
                img = _random_rotate(img.copy(), max_degree=30)

        # trick 2: color augmentation
        # random change the brightness, contrast, saturation and hue of the image.
            
            if type == 3: # random brightness
                img = _random_brightness(img.copy())
            if type == 4: # random contrast
                img = _random_contrast(img.copy())
            if type == 5: # random saturation
                img = _random_saturation(img.copy())
            if type == 6: # random hue
                img = _random_hue(img.copy())
            else: # no augmentation
                pass

            # keep output shape unchanged for training; sample one augmentation each call

        img = np.asarray(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)


        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # --------------------------------
    # The resolution of CIFAR-10 is tooooo low
    # You can use Lenna.png as an example to visualize and check your code.
    # Submit the origin image "Lenna.png" as well as at least two augmented images of Lenna named "Lenna_aug1.png", "Lenna_aug2.png" ...
    # --------------------------------

    # # Visualize CIFAR-10. For someone who are intersted.
    # train_dataset = CIFAR10()
    # i = 0
    # for imgs, labels in train_dataset:
    #     imgs = imgs.transpose(1,2,0)
    #     cv2.imwrite(f'aug1_{i}.png', imgs)
    #     i += 1
    #     if i == 10:
    #         break 

    # Visualize and save for submission
    img = Image.open('Lenna.png')
    img.save('../results/Lenna.png')

    # --------------TODO------------------
    # Copy the first kind of your augmentation code here
    # --------------TODO------------------
    aug1 = _random_hflip(img.copy())
    aug1.save(f'../results/Lenna_aug1.png')

    # --------------TODO------------------
    # Copy the second kind of your augmentation code here
    # --------------TODO------------------
    aug2 = _random_hue(img.copy())
    aug2.save(f'../results/Lenna_aug2.png')

    
