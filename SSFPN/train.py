import torch
from data import create_dataloader
import SSFPN
import S2_FPN
import config
import util
from tqdm import tqdm
import numpy as np
from PIL import Image

def val(val_loader, model):
    descs = f'val los: {0} all_los: {0}'
    los = 0
    all_los = 0
    model.eval()
    loss = torch.nn.CrossEntropyLoss()
    with tqdm(val_loader, desc = descs) as t:
        for batch in t:
            pre = model(batch['image'])
            los = loss(pre, batch['label'].cuda())
            all_los += los.item()
            descs = f'val los: {los:.6f} all_los: {all_los:.6f}'
            t.set_description(desc=descs)
    return all_los



def train():
    model = SSFPN.SSFPN('resnet18', classes=config.classes)
    # model = SSFPN.SSFPN('resnet34', classes=2)
    model = torch.nn.parallel.DataParallel(model.cuda())
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=config.lr)
    train_loader, val_loader = create_dataloader()
    min_los = 999999999
    cls_to_label = util.get_label(util.get_classifier([], c=1))
    # print(cls_to_label)
    for epoch in range(config.epochs):
        all_los = 0
        step = 0
        descs = f'step: {epoch} los: {0} all_los: {0}'
        with tqdm(train_loader, desc = descs) as t:
            for batch in t:
                opt.zero_grad()
                model.train()
                if step == 0 and epoch == 0:
                    save_image = batch['image'][0].permute(1,2,0).detach().cpu().numpy() * 255
                    save_label = batch['label'][0].clone().detach().cpu().numpy() * 255
                    # for x in range(save_label.shape[0]):
                    #     for y in range(save_label.shape[1]):
                    #         save_label[x][y] = cls_to_label[save_label[x][y]]
                    #         # if save_label[x][y] != 0 and save_label[x][y] != 255:
                    #         #     print(save_label[x][y])
                    # # print(save_image.shape)
                    save_image = Image.fromarray(np.uint8(save_image))
                    save_label = Image.fromarray(np.uint8(save_label))
                    # save_label.putpalette(batch['label_palette'][0])
                    save_image.save('origin_image__.png')
                    save_label.save('origin_label__.png')
                    # util.show_image(batch['image'][0].permute(1,2,0), batch['label'][0])
                pre, sup5, sup4, sup3, sup2 = model(batch['image'])
                # los = loss(pre, batch['label'].cuda()) + loss(sup5, batch['label'].cuda()) + loss(sup4, batch['label'].cuda()) + loss(sup3, batch['label'].cuda()) + loss(sup2, batch['label'].cuda())
                los = loss(pre, batch['label'].cuda())
                if step == 0:
                    # pre = torch.sum(pre, 1)
                    pre = pre.argmax(dim=1)
                    pre = pre[0].detach().cpu().numpy()
                    # for x in range(pre.shape[0]):
                    #     for y in range(pre.shape[1]):
                    #         pre[x][y] = cls_to_label[pre[x][y]]
                    pre = Image.fromarray(np.uint8(pre * 255))
                    # pre.putpalette(batch['label_palette'][0])
                    pre.save(str(epoch) + '__.png')
                    # util.show_image(batch['image'][0].permute(1,2,0), pre[0].detach().cpu().numpy())
                los.backward()
                opt.step()
                all_los += los.item()
                descs = f'epoch: {epoch} los: {los.item():.6f} all_los: {all_los:.6f}'
                t.set_description(desc=descs)
                step += 1
        
        val_los = val(val_loader=val_loader, model=model)
        if val_los < min_los:
            min_los = val_los
            torch.save(model, config.model_path)
            print('have save model with {:.6f}'.format(min_los))


if __name__ == '__main__':
    train()