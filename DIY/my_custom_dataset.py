from PIL import Image
from os.path import join
from torch.utils.data import Dataset

def get_class_dicts( root='/home/nemo/VRDL2021/HW1_Fine_Grained/datasets/CUB'):
    class2int = dict()
    int2class = dict()
    with open(join(root, 'classes.txt')) as f :
        lines = f.readlines()
        for l in lines:
            index , class_name = int(l.split('.')[0].strip()), l.strip()
            class2int[class_name] = index
            int2class[index] = class_name
        return class2int, int2class

class trainset(Dataset):
    def __init__(self, transform=None, root='/home/nemo/NYCU_VRDL_HW1/datasets/CUB'):
        self.root = root
        self.transform=transform
        self.class2int, self.int2class = get_class_dicts(root)
        self.labels = dict()
        self.files = []
        with open(join(root,'training_labels.txt')) as f:
            for l in f.readlines():
                file , label = l.split(' ')[0].strip(), int(l.split(' ')[1].split('.')[0].strip())
                self.labels[file] = label
                self.files.append(file)
    def __len__(self):
        return int(len(self.files)*0.9)
    def __getitem__(self, idx):
        label = self.labels[self.files[idx]]
        img = Image.open(join(self.root,'training_images', self.files[idx]))
        if self.transform:
            img = self.transform(img)
        return img, label


class testset(Dataset):
    def __init__(self, transform=None, root='/home/nemo/NYCU_VRDL_HW1/datasets/CUB'):
        self.root = root
        self.transform=transform
        self.class2int, self.int2class = get_class_dicts(root)
        self.labels = dict()
        self.files = []
        with open(join(root,'training_labels.txt')) as f:
            for l in f.readlines():
                file , label = l.split(' ')[0].strip(), int(l.split(' ')[1].split('.')[0].strip())
                self.labels[file] = label
                self.files.append(file)
    def __len__(self):
        # this line is different from class trainset
        return len(self.files) - int(len(self.files)*0.9)
    def __getitem__(self, idx):
        # this line is different from class trainset
        idx = idx + int(len(self.files)*0.9)
        label = self.labels[self.files[idx]]
        img = Image.open(join(self.root,'training_images', self.files[idx]))
        if self.transform:
            img = self.transform(img)
        return img, label

class evalset(Dataset):
    def __init__(self, transform=None, root='/home/nemo/NYCU_VRDL_HW1/datasets/CUB'):
        self.root = root
        self.transform = transform
        self.class2int, self.int2class = get_class_dicts(root)
        self.files = []
        with open(join(root,'testing_img_order.txt')) as f:
            for file in f.readlines():
                self.files.append(file.strip())
    def __len__(self):
            return len(self.files)
    def __getitem__(self, index):
            img = Image.open(join(self.root,'testing_images',self.files[index]))
            if self.transform:
                img = self.transform(img)
            return self.files[index], img
