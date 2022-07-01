import torch
import util
import os
import config
from PIL import Image
import numpy as np

class DATASET(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        image = util.read_image(os.path.join(config.TRAIN_DATA_PATH, config.predict_pic + '.jpg'), use_resize=True, resize=(224, 224))
        image = image / 255.0
        image = image.transpose(2,0,1)
        image = torch.FloatTensor(image)
        data = dict(
            image=image
        )
        return data

def create_dataloader():
    dataset = DATASET()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    return dataloader

def inference(dataloader):
    model = torch.load(config.model_path)
    model.eval()
    for batch in dataloader:
        output = model(batch['image'])
    output = output.view(output.shape[1], output.shape[2], output.shape[3])
    output = output.argmax(dim=0)
    return output.detach().cpu().numpy()

if __name__ == '__main__':
    dataloader = create_dataloader()
    pre = inference(dataloader)
    image = util.read_image(os.path.join(config.TRAIN_DATA_PATH, config.predict_pic + '.jpg'), use_resize=True, resize=(224, 224))
    util.show_image(image, pre)
    img = Image.fromarray(np.uint8(pre) * 255)
    img.save(config.predict_pic + '.png')


