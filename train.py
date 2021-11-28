import argparse
import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    ToTensor,
    Normalize
)
from model import init_model
from dataset import Imagenette
from cutmix import CutMix
from maxup_loss import MaxupCrossEntropyLoss
from config import *
import faulthandler
faulthandler.enable()

# 字符串转二进制
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 运行参数
parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=1,
                    help="Number of augmentations for element in maxup")
parser.add_argument("--cutmix", type=str2bool, default=False,
                    help="Whether to use cutmix or not")
parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument("--result_path", type=str, default='./result/Base_exp')
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--pretrained_weights", type=str, default='')
# 输出路径
RESULT_PATH = './result'
os.makedirs(RESULT_PATH, exist_ok=True)


def adjust_learning_rate(optimizer, epoch, lr):
    """将学习速率设置为每30个历元衰减10次的初始LR"""
    lr = lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validation(lossf, valid_loader, model, device):
    model.eval()
    v_loss, acc = 0, 0
    vlen = len(valid_loader)
    all_output = []
    all_gt = []
    with torch.no_grad():
        for img, labels in tqdm(valid_loader):
            img = img.to(device).float()
            labels = labels.to(device)
            output = model(img)
            loss = lossf(output, labels, valid=True)
            v_loss += loss.item()
            all_gt.extend(labels.max(1)[1].cpu().numpy().tolist())
            all_output.extend(output.max(1)[1].cpu().numpy().tolist())
    accuracy = accuracy_score(all_gt, all_output)
    v_loss = v_loss / (vlen+1)
    return accuracy, v_loss

# 模型训练：
# cutmix - 使用cutmix;
# m - Maxup的Cutmix放大倍数;
# device - 使用CPU或GPU;
# epochs - 检查点数;
# pretrained_weights - 预制模型的天平路径.
def train(args):
    os.makedirs(args.result_path, exist_ok=True)
    transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = Imagenette(DATAPATH,
                               mode='train',
                               size=IMG_SIZE,
                               transform=transform_train)
    if args.cutmix is True:
        train_dataset = CutMix(train_dataset, args.m)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True)
    transform_valid = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''数据集来源'''
    valid_dataset = Imagenette(DATAPATH,
                               mode='train',
                               size=IMG_SIZE,
                               valid=True,
                               transform=transform_valid)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=1,
                                               pin_memory=True)
    '''模型构建'''
    model = init_model()
    if args.pretrained_weights:
        print('Load pretrained weights ', args.pretrained_weights)
        state_dict = torch.load(args.pretrained_weights, map_location=args.device)
        model.load_state_dict(state_dict)
    model.to(args.device)
    '''使用maxup'''
    criterion = MaxupCrossEntropyLoss(args.m)
    optimizer = torch.optim.SGD(model.parameters(), LR,
                                momentum=MOMENTUM,
                                weight_decay=1e-4,
                                nesterov=True)
    len_loader = len(train_loader)
    min_loss = 100000000
    iter = 0
    no_loss_improve = 0
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, LR)
        t_loss = 0
        for i, (imgs, labels) in tqdm(enumerate(train_loader)):
            model.train()
            imgs = imgs.reshape((imgs.shape[0] * args.m, 3, IMG_SIZE, IMG_SIZE))
            labels = labels.to(args.device)
            imgs = imgs.to(args.device).float()
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels, True) if args.m == 1 else criterion(output, labels)
            # loss.backward()
            t_loss += loss.item()
            optimizer.step()
            iter += 1
        val_acc, val_loss = validation(criterion, valid_loader, model, args.device)
        print("\nEpoch {} -> Train Loss: {:.4f}".format(epoch, t_loss / len_loader))
        # print("\nEpoch {} -> Valid Loss: {:.4f}\n Valid Accuracy: {:.4f}".format(epoch, val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(args.result_path, 'weights.pth'))
            min_loss = val_loss
            no_loss_improve = 0
        else:
            no_loss_improve += 1
        if no_loss_improve == EARLY_STOPPNIG_NUM:
            break

# main函数启动
if __name__ == "__main__":
    args = parser.parse_args()
    if args.m <= 0:
        raise argparse.ArgumentTypeError('m > 0 expected')
    train(args)
