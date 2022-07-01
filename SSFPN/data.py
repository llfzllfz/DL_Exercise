import matplotlib.pyplot as plt
import config
import os
import numpy as np
from tqdm import tqdm
import util
import torch

class DATASET(torch.utils.data.Dataset):
    def __init__(self):
        self.TRAIN_DATA_PATH = config.TRAIN_DATA_PATH
        self.LABEL_DATA_PATH = config.LABEL_DATA_PATH
        self.filename_list = util.get_filename_list()
        self.classifier_dict = util.get_classifier(self.filename_list, c = 1)
        pass

    def __len__(self) -> int:
        return len(self.filename_list)

    def __getitem__(self, idx: int) -> dict:
        image = util.read_image(os.path.join(config.TRAIN_DATA_PATH, self.filename_list[idx] + '.jpg'), use_resize=True)
        label = util.read_image(os.path.join(config.LABEL_DATA_PATH, self.filename_list[idx] + '-profile.jpg'), use_resize=True)
        image = image / 255.0
        image = image.transpose(2,0,1)
        # label = label / 255
        # label = np.uint8(label)
        # for x in range(label.shape[0]):
        #     for y in range(label.shape[1]):
        #         if label[x][y] != 0:
        #             label[x][y] = 1
                # label[x][y] = self.classifier_dict[label[x][y]]
        image = torch.FloatTensor(image)
        label = torch.LongTensor(label)
        label[label != 0] = 1
        # label_palette = torch.LongTensor(label_palette)
        data = dict(
            image=image,
            label=label
        )
        return data

def create_dataloader():
    dataset = DATASET()
    size = len(dataset)
    val_size = int(size * config.val_radio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = config.batch_size)
    return train_dataloader, val_dataloader