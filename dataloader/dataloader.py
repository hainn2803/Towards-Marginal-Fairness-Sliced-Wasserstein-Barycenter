from pytorch_balanced_sampler import *
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from dataloader.imbalanced_dataset import IMBALANCEMNIST, IMBALANCECIFAR10, IMBALANCECIFAR100


class BaseDataLoader:
    def __init__(self, data_dir="data/", train_batch_size=128, test_batch_size=64, num_classes=0, dataset_name="mnist"):
        self.test_loader = None
        self.train_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.dataset_name = dataset_name
        self.create_dataset()

    def create_dataset(self):
        pass

    def create_dataloader(self):

        if self.dataset_name == "stl10":
            train_target = self.train_dataset.labels
        else:
            train_target = self.train_dataset.targets

        instances_indices = torch.arange(len(train_target))
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[torch.tensor(train_target) == i].tolist()
            all_classes_indices.append(class_index)

        batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.train_batch_size,
            n_batches=len(train_target) // self.train_batch_size,
            alpha=0,
            kind='random'
        )

        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)

        return self.train_loader, self.test_loader

class MNISTDataLoader(BaseDataLoader):
    def __init__(self, data_dir="data/", train_batch_size=250, test_batch_size=250):
        super(MNISTDataLoader, self).__init__(data_dir=data_dir,
                                              train_batch_size=train_batch_size,
                                              test_batch_size=test_batch_size,
                                              num_classes=10,
                                              dataset_name="mnist")

    def create_dataset(self):
        train_set = datasets.MNIST(root=self.data_dir,
                                   train=True,
                                   download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
        test_set = datasets.MNIST(root=self.data_dir,
                                  train=False,
                                  download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))
        self.train_dataset = train_set
        self.test_dataset = test_set

class CIFAR10DataLoader(BaseDataLoader):
    
    def __init__(self, data_dir="data/", train_batch_size=80, test_batch_size=80):
        super(CIFAR10DataLoader, self).__init__(data_dir=data_dir,
                                                  train_batch_size=train_batch_size,
                                                  test_batch_size=test_batch_size,
                                                  num_classes=10,
                                                  dataset_name="cifar10")

    def create_dataset(self):
        train_set = datasets.CIFAR10(root=self.data_dir,
                                     train=True,
                                     download=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))

        test_set = datasets.CIFAR10(self.data_dir,
                                    train=False, 
                                    download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))

        self.train_dataset = train_set
        self.test_dataset = test_set

class STL10DataLoader(BaseDataLoader):
    
    def __init__(self, data_dir="data/", train_batch_size=80, test_batch_size=80, image_size=64):
        self.image_size = image_size
        super(STL10DataLoader, self).__init__(data_dir=data_dir,
                                                  train_batch_size=train_batch_size,
                                                  test_batch_size=test_batch_size,
                                                  num_classes=10,
                                                  dataset_name="stl10")

    def create_dataset(self):
        train_set = datasets.STL10(root=self.data_dir,
                                     split="train",
                                     download=False,
                                     transform=transforms.Compose([
                                                transforms.Resize(self.image_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                # transforms.Normalize((0.4384, 0.4314, 0.3989), (0.2647, 0.2609, 0.2741))
                                     ])
                                     )

        test_set = datasets.STL10(root=self.data_dir,
                                    split="test", 
                                    download=False,
                                     transform=transforms.Compose([
                                                transforms.Resize(self.image_size),
                                                transforms.ToTensor(),
                                                # transforms.Normalize((0.4384, 0.4314, 0.3989), (0.2647, 0.2609, 0.2741))
                                     ])
                                    )

        self.train_dataset = train_set
        self.test_dataset = test_set

class CIFAR100DataLoader(BaseDataLoader):
    def __init__(self, data_dir="data/", train_batch_size=80, test_batch_size=80):
        super(CIFAR100DataLoader, self).__init__(data_dir=data_dir,
                                                   train_batch_size=train_batch_size,
                                                   test_batch_size=test_batch_size,
                                                   num_classes=100,
                                                   dataset_name="cifar100")

    def create_dataset(self):
        train_set = datasets.CIFAR100(root=self.data_dir,
                                      train=True, 
                                      download=False,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                      ]))

        test_set = datasets.CIFAR100(self.data_dir,
                                     train=False, 
                                     download=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))

        self.train_dataset = train_set
        self.test_dataset = test_set

class CelebADataLoader():
    def __init__(self, data_dir="data/", train_batch_size=80, test_batch_size=80, num_classes=2):
        super(CelebADataLoader, self).__init__()
        
        self.test_loader = None
        self.train_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.create_dataset()

    def create_dataloader(self):
        instances_indices = torch.arange(len(self.train_dataset.attr))
        print(instances_indices.shape)
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[torch.tensor(self.train_dataset.attr[:, 20]) == i].tolist()
            all_classes_indices.append(class_index)

        batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.train_batch_size,
            n_batches=len(self.train_dataset.attr) // self.train_batch_size,
            alpha=0,
            kind='random'
        )

        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)

        return self.train_loader, self.test_loader

    def create_dataset(self):
        train_set = datasets.CelebA(root=self.data_dir, 
                                    split='train', 
                                    target_type='attr', 
                                    transform=transforms.Compose([
                                            transforms.Resize((64, 64)),
                                            transforms.ToTensor()
                                        ]), 
                                    download=False)

        test_set = datasets.CelebA(root=self.data_dir, 
                                    split='test', 
                                    target_type='attr', 
                                    transform=transforms.Compose([
                                            transforms.Resize((64, 64)),
                                            transforms.ToTensor()
                                        ]), 
                                    download=False)

        self.train_dataset = train_set
        self.test_dataset = test_set

class CIFAR10LTDataLoader(BaseDataLoader):
    def __init__(self, data_dir="data/", train_batch_size=80, test_batch_size=80):
        self.test_loader = None
        self.train_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = 10

        self.create_dataset()

    def create_dataset(self):

        train_set = IMBALANCECIFAR10(root=self.data_dir + "train/",
                                     imb_type='exp', imb_factor=0.01,
                                     train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))

        test_set = IMBALANCECIFAR10(self.data_dir + "test/",
                                    imb_type='exp', imb_factor=1,
                                    train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))

        self.train_dataset = train_set
        self.test_dataset = test_set


class MNISTLTDataLoader(BaseDataLoader):
    def __init__(self, data_dir="data", train_batch_size=80, test_batch_size=80):
        super(MNISTLTDataLoader, self).__init__(data_dir=data_dir,
                                              train_batch_size=train_batch_size,
                                              test_batch_size=test_batch_size,
                                              num_classes=10,
                                              dataset_name="mnist")

    def create_dataset(self):

        train_set = IMBALANCEMNIST(root=self.data_dir,
                                     imb_type='exp', imb_factor=0.01,
                                     train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))

        test_set = datasets.MNIST(root="data/mnist",
                                  train=False,
                                  download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))

        self.train_dataset = train_set
        self.test_dataset = test_set

    def create_dataloader(self):

        if self.dataset_name == "stl10":
            train_target = self.train_dataset.labels
        else:
            train_target = self.train_dataset.targets

        instances_indices = torch.arange(len(train_target))
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[torch.tensor(train_target) == i].tolist()
            all_classes_indices.append(class_index)

        batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.train_batch_size,
            n_batches=len(train_target) // self.train_batch_size,
            alpha=0,
            kind='fixed'
        )

        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)

        return self.train_loader, self.test_loader