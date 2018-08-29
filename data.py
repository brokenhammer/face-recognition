import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
import torch
import os
import pickle
from config import cfg


class TrainValSet(data.Dataset):
    def __init__(self, img_dir, ann_path, transform):
        super(TrainValSet, self).__init__()
        self.img_dir = img_dir
        self.ann_path = ann_path
        self.transform = transform
        self.entry = []
        self._create_entry()


    def __getitem__(self, index):
        img_info = self.entry[index]
        img_name = img_info['img_name']
        img_path = os.path.join(self.img_dir, img_name)
        id = img_info['identity']
        img_raw = Image.open(img_path).convert('RGB')
        #longer = max(img_raw.size)
        #dw = longer - img_raw.size[0]
        #dh = longer - img_raw.size[1]
        #img_raw = ImageOps.expand(img_raw, (dw//2, dh//2, dw//2, dh//2), fill=0)
        img_data = self.transform(img_raw)

        return img_data, id, self.id2label[id]

    def _create_entry(self):
        self.id2label = {}
        id_set = set()
        with open(self.ann_path, 'rb') as f:
            self.entry = pickle.load(f)
        label = 0
        for item in self.entry:
            if not item["identity"] in id_set:
                self.id2label[item["identity"]] = label
                id_set.add(item["identity"])
                label += 1
        self.class_num = len(id_set)
        print("identity:{}".format(self.class_num))

    def __len__(self):
        return len(self.entry)


def get_loader(train_img_dir, val_img_dir, ann_save_dir,
               transform, batch_size):
    train_ann_path = os.path.join(ann_save_dir, 'train.pickle')
    train_set = TrainValSet(train_img_dir, train_ann_path, transform)
    train_loader = data.DataLoader(train_set,batch_size=batch_size,
                                   shuffle=True, num_workers=4)
    print("train identity:{}".format(train_set.class_num))
    val_ann_path = os.path.join(ann_save_dir, 'val.pickle')
    val_set = TrainValSet(val_img_dir, val_ann_path, transform)
    val_loader = data.DataLoader(val_set,batch_size=batch_size,
                                   shuffle=False, num_workers=4)
    print("val identity:{}".format(val_set.class_num))

    return train_loader, val_loader

def prepare_split(split_fpath, identity_fpath, save_dir, override=False):

    write_train = override or not os.path.exists(os.path.join(save_dir, "train.pickle"))
    write_val = override or not os.path.exists(os.path.join(save_dir, "val.pickle"))
    write_test = override or not os.path.exists(os.path.join(save_dir, "test.pickle"))
    if not (write_test or write_train or write_val):
        print("all data files exists, return ...")
        return
    train_set = []
    val_set = []
    test_set = []
    with open(split_fpath, 'r') as f:
        lines = f.read().split('\n')[:-1]

    with open(identity_fpath, 'r') as f2:
        lines2 = f2.read().split('\n')[:-1]
    for i, ln in enumerate(lines):
        img_name, usage = ln.strip().split()
        img_name2, identity = lines2[i].strip().split()
        img_name = img_name.strip()
        img_name2 = img_name2.strip()
        assert img_name == img_name2
        usage = usage.strip()
        identity = int(identity.strip())

        if usage == "0":
            train_set.append({"img_name": img_name, "identity": identity})
        elif usage == "1":
            val_set.append({"img_name": img_name, "identity": identity})
        elif usage == "2":
            test_set.append({"img_name": img_name, "identity": identity})
    print("finished parsing, train images:{}, val images:{}, test images:{}".format(len(train_set), len(val_set), len(test_set)))

    with open(os.path.join(save_dir, "train.pickle"), 'wb') as ftrain:
        pickle.dump(train_set, ftrain)

    with open(os.path.join(save_dir, "val.pickle"), 'wb') as fval:
        pickle.dump(val_set, fval)

    with open(os.path.join(save_dir, "test.pickle"), 'wb') as ftest:
        pickle.dump(test_set, ftest)


