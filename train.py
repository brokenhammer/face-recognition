import torch
from models.arcface_model import ArcSoftmax, Classifier
import os
from torchvision import transforms
from config import cfg
from data import get_loader, prepare_split
from datetime import datetime
import torch.optim as optim
import numpy as np
import shutil
import argparse
from models.resnet import resnet_face18
from models.metrics import ArcMarginProduct
from models.focal_loss import FocalLoss


def main(args):
    torch.cuda.set_device(0)
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)

    transform = transforms.Compose([
        transforms.CenterCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # only val once
    prepare_split(args.split_fpath, args.identity_fpath, args.save_dir)
    train_loader, val_loader = get_loader(args.train_img_dir, args.val_img_dir, args.save_dir,
                                          transform, args.batch_size)
    iter_size = len(train_loader)
    val_size = len(val_loader)
    model = Classifier(train_loader.dataset.class_num)
    #model = resnet_face18(use_se=False)
    #metric_fc = ArcMarginProduct(512,13938)
    #metric_fc.cuda()
    model.cuda()
    start_epoch = 0
    best_epoch = 0
    if args.start_epoch > 0:
        model.load_state_dict(torch.load(args.pretrained))
        start_epoch = args.start_epoch
        #save_dir = os.path.split(args.pretrained)
        #fc_path = os.path.join(save_dir, 'fc_{}'.format(start_epoch))
        #metric_fc.load_state_dict(torch.load(fc_path))
    best_acc = 0.0
    acc_hist = []

    criterion = ArcSoftmax(gamma=2)#FocalLoss(gamma=2)
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        stime = datetime.now()
        if epoch <= 8:
            lr = 0.1
        elif epoch <= 14:
            lr = 0.01
        elif epoch <= 20:
            lr = 0.001
        else:
            lr = 0.0001
        print('Epoch {}, learning rate = {:.3e}'.format(epoch, lr))
        model.train()
        #params = model.parameters()
        optimizer = optim.SGD(model.parameters(), lr, weight_decay=5e-4)
        train_loss = 0
        np.random.shuffle(train_loader.dataset.entry)
        total = 0
        correct = 0
        for i, data in enumerate(train_loader):
            print("\rTRAIN_ITER:{}/{} ".format(i, iter_size), end="")
            img_inputs, id, labels = data
            img_inputs = img_inputs.cuda()
            labels = torch.LongTensor(labels).cuda()
            output = model(img_inputs)
            cos = output[0]
            _, pred_labels = torch.max(cos.data, 1)
            crt_num = labels.shape[0]
            crt_correct = (pred_labels == labels).cpu().sum()
            total += crt_num
            correct += crt_correct

            loss = criterion(output, labels)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('loss:{:.4f}, correct:{}/{}'.format(loss, crt_correct, crt_num) + " "*10,end="")

        etime = datetime.now()
        acc = correct / total
        print("\nepoch {}, mean loss"
                ":{:.4f}, time:{}, correct/total: {}/{} acc:{:.4f}\n".format(epoch,
                                                    train_loss / iter_size,
                                                    str(etime - stime),
                                                    correct, total,
                                                    acc))
        val_loss = 0.0
        torch.cuda.empty_cache()

        if epoch % args.save_freq == args.save_freq - 1:
            fullpath = os.path.join(args.weight_dir, 'epoch_{}'.format(epoch))
            #fc_path = os.path.join(args.weight_dir, 'fc_{}'.format(epoch))
            torch.save(model.state_dict(), fullpath)
            #torch.save(metric_fc.state_dict(), fc_path)

        # model.eval()
        # with torch.no_grad():
        #     for i, data in enumerate(val_loader):
        #         model.zero_grad()
        #         print("\rVAL_ITER:{}/{} ".format(i, val_size), end="")
        #         img_inputs, labels = data
        #         img_inputs = img_inputs.cuda()
        #         labels = torch.LongTensor(labels).cuda()
        #         output = model(img_inputs)
        #
        #         cos = output[0]
        #         _, pred_labels = torch.max(cos.data, 1)
        #         crt_num = labels.shape[0]
        #         crt_correct = (pred_labels == labels).cpu().sum()
        #         total += crt_num
        #         correct += crt_correct
        #
        #         loss = criterion(output, labels)
        #         train_loss += loss.item()
        #
        #         print('loss:{:.4f}, correct:{}/{}'.format(loss, crt_correct, crt_num) + " "*10, end="")
        #
        # print("\n val_loss: {}, acc: {}\n".format(val_loss / val_size, correct/total))
        #
        # torch.cuda.empty_cache()
        # val_acc = correct / total
        # if val_acc > best_acc:
        #     best_epoch = epoch
        #     best_acc = val_acc
        #
        # if len(acc_hist) > 5:
        #     last_6 = np.array(acc_hist)
        #     if max(last_6) < best_acc:
        #         print(' No improvement with validation loss ... Early stopping triggerd')
        #         print(' Model of best epoch #: {} with accuarcy {:.3e}'.format(best_epoch, best_acc))
        #         shutil.copy(os.path.join(args.weight_dir, 'epoch_{}'.format(best_epoch)),
        #                     os.path.join(args.weight_dir, 'best'))
        #     break
    else:
        torch.save(model.state_dict(), os.path.join(args.weight_dir, 'best'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_dir", type=str, default="./weights")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_img_dir", type=str,
                        default="/data/public/weixishuo/recognition/img_align_celeba")
    parser.add_argument("--val_img_dir", type=str,
                        default="/data/public/weixishuo/recognition/img_align_celeba")
    parser.add_argument("--split_fpath",
                        default="/data/public/weixishuo/recognition/annotations/list_eval_partition.txt")
    parser.add_argument("--identity_fpath",
                        default="/data/public/weixishuo/recognition/annotations/identity_CelebA.txt")
    parser.add_argument("--save_dir",
                        default="/data/public/weixishuo/recognition/annotations/")

    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=40)

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--pretrained", default="")

    args = parser.parse_args()
    main(args)
