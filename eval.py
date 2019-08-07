import torch

def clean(net, loader):
    net.eval()
    with torch.no_grad():
        positive = 0.0
        for num, (img, label) in enumerate(loader):
            img = img.cuda()
            label = label.cuda()
            preds = net(img)
            preds = torch.argmax(preds, -1)
            acc = preds.eq(label)
            acc = acc.float().sum()
            positive += acc.item()

        return positive

def white_box(net, loader, criterion):
    net.eval()
    with torch.no_grad():
        positive = 0.0
        for num, (img, label) in enumerate(loader):
            img = img.cuda()
            label = label.cuda()

            preds = net(img)
            img_adv = criterion.PGD_L2(net, img, preds)

            preds_adv = net(img_adv)
            preds_adv = torch.argmax(preds_adv, -1)
            acc = preds_adv.eq(label)
            acc = acc.float().sum()
            positive += acc.item()

        return positive

def black_box(net_defense, net_attack, loader, criterion):
    net_defense.eval()
    net_attack.eval()
    with torch.no_grad():
        positive = 0.0
        for num, (img, label) in enumerate(loader):
            img = img.cuda()
            label = label.cuda()

            preds = net_attack(img)
            img_adv = criterion.PGD_L2(net_attack, img, preds)

            preds_adv = net_defense(img_adv)
            preds_adv = torch.argmax(preds_adv, -1)
            acc = preds_adv.eq(label)
            acc = acc.float().sum()
            positive += acc.item()

        return positive