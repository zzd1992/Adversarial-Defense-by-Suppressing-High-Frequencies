import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
import os, argparse, copy, time
from dataset import DatasetFile
from torch.utils.data import DataLoader
from resnet import ResNet18
from core import Trades
import cfg
import eval

parse = argparse.ArgumentParser()
parse.add_argument('train_file', help='file of training images')
parse.add_argument('valid_file', help='file of validation images')
parse.add_argument('-pth', default=None, help='pre-trained model file (from train_base.py)')
parse.add_argument("-seed", default=0, type=int, help='random seed')
parse.add_argument('-workers', default=4, type=int, help='workers for data load')
parse.add_argument('-bs', default=64, type=int, help='batch size for training')
parse.add_argument('-epoch', default=40, type=int, help='training epoch')
parse.add_argument('-lr', default=0.1, type=float, help='learning rate')
parse.add_argument('-wd', default=1e-4, type=float, help='weight decay')
parse.add_argument('-init_channels', default=64, type=int, help='channels of the first block of ResNet-18')
parse.add_argument('-r', default=12, type=int, help='radius of high frequency suppression module')
parse.add_argument('-epsilon', default=0.05, type=float, help='distance constraint for adversarial perturbations')
parse.add_argument('-step_size', default=0.0075, type=float, help='step size for adversarial sample generation')
parse.add_argument('-perturb_steps', default=3, type=int, help='iterations for adversarial sample generation')
parse.add_argument('-beta', default=1.0, type=float, help='weight of adversarial loss')
args = parse.parse_args()
print(args)

if not os.path.isdir("checkpoints"): os.mkdir("checkpoints")

transform_train = transforms.Compose([
    transforms.RandomCrop(cfg.crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(cfg.crop_size),
    transforms.ToTensor(),
])

torch.backends.cudnn.benchmark = True

trainset = DatasetFile(cfg.root, args.train_file, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers)

testset = DatasetFile(cfg.root, args.valid_file, transform=transform_test)
testloader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.workers)

net = ResNet18(cfg.nb_class)
if args.pth is not None:
    net.load_state_dict(torch.load(args.pth))
net.cuda()
net.train()

opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
criterion = Trades(args.step_size, args.epsilon, args.perturb_steps, args.beta)
epoch = args.epoch
best_acc = 0.0

for i in range(epoch):
    if i + 1 in [int(epoch * 0.4), int(epoch * 0.8)]:
        for param_group in opt.param_groups:
            param_group['lr'] /= 10
    error, t = 0.0, time.time()
    net.train()
    for num, (img, label) in enumerate(trainloader):
        opt.zero_grad()
        img = img.cuda()
        label = label.cuda()

        preds = net(img)
        preds_detach = preds.detach()
        img_adv = criterion.PGD_L2(net, img, preds_detach)
        loss = criterion.loss(net, preds, img_adv, label, opt)
        loss.backward()
        opt.step()
        error += loss.item()

    print("{}th epoch: {:.5f}\t{:.1f}s".format(i, error/(num+1), time.time()-t))

    if i%2 == 0:
        accs = eval.clean(net, testloader)
        accs /= len(testset)
        print("metric: {:.4f}".format(accs))
        if accs > best_acc:
            best_acc = copy.deepcopy(accs)
            if i >= int(epoch * 0.65):
                torch.save(net.state_dict(), "checkpoints/adv_r{}_epoch{}.pth".format(args.r, i))

torch.save(net.state_dict(), "checkpoints/adv_r{}_epoch{}.pth".format(args.r, i))
