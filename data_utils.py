import torch
from torchvision import transforms, datasets
import random
# for check
import numpy as np

# data partitioner

class Partition(object):
    """ Dataset like object (DataLoader input)"""

    def __init__(self, data, par_index):
        self.data = data
        # print('type of par_index', type(par_index))
        self.index = par_index # list

    def __len__(self):
        # print(type(self.index))
        return len(self.index)
    
    def __getitem__(self, index):
        data_idx = self.index[index]
        # index = list of indices
        return self.data[data_idx]

class DataPartitioner(object):
    """Holds 2d list partitions, and acts like dataset(class Partition)"""
    def __init__(self, sizes, testmask, dataset_name, datapath):
        # input: data is whole trainset or whole testset

        if dataset_name == "CIFAR10":
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
            self.data = datasets.CIFAR10(root=datapath, # original dataset, etc, CIFAR-10
                                            train=True,
                                            download=True,
                                            transform=self.transform) 

            self.testdata = datasets.CIFAR10(root=datapath, # original dataset, etc, CIFAR-10
                                            train=False,
                                            download=True,
                                            transform=self.transform)
        elif dataset_name == "MNIST":
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            
            self.data = datasets.MNIST(root=datapath, # original dataset, etc, CIFAR-10
                                            train=True,
                                            download=True,
                                            transform=self.transform) 

            self.testdata = datasets.MNIST(root=datapath, # original dataset, etc, CIFAR-10
                                            train=False,
                                            download=True,
                                            transform=self.transform)

        self.partitions = [] # 2d list for indices
        self.testset = [] # 1d list acts like partitions, for testset
        self.ratio = sizes # 2d list for data ratio (class x ratio for each class)
        self.testmask = testmask # acts like self.ratio but for testset
        self.valset = []

    def makepartition(self, validation=False, _event=None):
        # makes 2d list partitions and testset
        n_of_class = len(self.testmask)
        
        # make list of indices for each class of dataset
        classes = [[] for _ in range(n_of_class)]
        testclasses = [[] for _ in range(n_of_class)]
        
        for i, item in enumerate(self.data):
            classes[item[1]].append(i)
        for i, item in enumerate(self.testdata):
            testclasses[item[1]].append(i)

        for i in range(n_of_class):
            random.shuffle(classes[i])
            random.shuffle(testclasses[i])
        
        # build list
        for i in range(len(self.ratio)):
            self.partitions.append(list())

        # make partitions
        for i, labels in enumerate(classes): # i is class id, labels is list containing indices of i class
            start_idx = 0
            for j, client in enumerate(self.ratio):
                if j == 0:  # pass if server
                    continue
                end_idx = int(client[i]*len(labels))+start_idx
                self.partitions[j].extend(labels[start_idx : end_idx])
                start_idx = end_idx

        # make testset
        for idx, i in enumerate(self.testmask):
            if i == 1:
                self.testset.extend(testclasses[idx])

        return self.partitions, self.testset

    def use(self, rank):
        return Partition(self.data, self.partitions[rank])
    
    def usetest(self):
        return Partition(self.testdata, self.testset)
    
    def useval(self):
        return Partition(self.valdata, self.valset)
    

def load_partition_data(idx, data_partitioner, batch_size, loader):
    """Returns train_loader and test_loader"""
    
    if loader == "train_loader":
        return torch.utils.data.DataLoader(data_partitioner.use(idx), batch_size=batch_size, shuffle=True, num_workers=2)
    elif loader == "test_loader":
        return torch.utils.data.DataLoader(data_partitioner.usetest(), batch_size=batch_size, shuffle=True, num_workers=2)
    elif loader == "validation_loader":
        return torch.utils.data.DataLoader(data_partitioner.useval(), batch_size=batch_size, shuffle=True, num_workers=2)


def get_n_of_samples(data_partitioner, rank):
        return len(data_partitioner.partitions[rank])