import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./dataset/testing_lr_images/')
    parser.add_argument('--model')
    parser.add_argument('--output')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    model.eval()

    img_names = os.listdir(args.input)

    for name in img_names:
        img = transforms.ToTensor()(Image.open(os.path.join(args.input, name)).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            pred = model((img- 0.5) / 0.5)[0]
        name = name.split(".")[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).cpu()
        transforms.ToPILImage()(pred).save(os.path.join(args.output, name)+"_pred.png")