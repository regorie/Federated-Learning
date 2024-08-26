import torch
import torch.nn as nn
import torch.nn.functional as F

import data_utils
from data_dist_maker import sizes_maker
import Models

import tqdm
import copy
import csv
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_of_clients', required=True, default=10)
parser.add_argument('--n_of_classes', default=10)
parser.add_argument('--iidness', default=10)
parser.add_argument('--model_name', required=True, default='CifarCNN')
parser.add_argument('--rnd', default=30)
parser.add_argument('--localepoch', default=4)

args = parser.parse_args()
n_of_client = args.n_of_clients
n_of_classes = args.n_of_classes
iidness = args.iidness
model_name = args.model_name
rnd = args.rnd
localepoch = args.localepoch

sizes, o_classes = sizes_maker(num_client=n_of_client, num_class=n_of_classes, shard_per_client=iidness, validation=False)

testmask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

results = [[] for _ in range(n_of_client+1)] # 1 server + clients


def train_(model, train_loader, optimizer, criterion, device, idx, localepoch):

    print("Client {} training!".format(idx))
    model.train()
    for ep in range(localepoch):
        for data in tqdm.tqdm(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

def evaluate_acc(model, test_loader, idx, round):

    model.eval()
    total = 0
    correct = 0
    class_correct = list(0. for _ in range(n_of_classes))
    class_total = list(0. for _ in range(n_of_classes))

    with torch.no_grad():
        for items in test_loader:
            images, labels = items[0].to(device), items[1].to(device)

            output = model.forward(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)

            for i, item in enumerate(labels):
                class_total[item] += 1
                if predicted[i] == item:
                    correct += 1
                    class_correct[item] += 1

        tmp = [100 * class_correct[i]/class_total[i] for i in range(n_of_classes)]
        tmp.append(100 * correct/total)

        results[idx].append(tmp)

        print('round: {} idx: {} total accuracy: {}'.format(round,
                                                            idx,
                                                            100 * correct / total))
        pass

if __name__=="__main__":
    # Configurations
    datapath = "../Data"
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')
    
    # Build model
    models = []
    models.append(getattr(Models, model_name))

    for i in range(n_of_client): # models[0] is global model and rest are clients
        models.append(copy.deepcopy(models[0]))

    for i in range(n_of_client+1):
        models[i].to(device)

    # Set optimizers and loss
    optimizers = []
    for i in range(n_of_client): # Global doesn't need optimizer, thus 10 optimizers
        optimizers.append(torch.optim.SGD(models[i+1].parameters(), lr=0.001, momentum=0.9))

    criterion = nn.CrossEntropyLoss()

    # Data partition
    data_partitioner = data_utils.DataPartitioner(sizes=sizes, testmask=testmask, dataset_name="CIFAR10", datapath=datapath)
    data_partitioner.makepartition()
    
    train_loaders = []
    for i in range(n_of_client): # global doesn't need train loader
        train_loaders.append(data_utils.load_partition_data(idx=i+1, data_partitioner=data_partitioner, batch_size=10, loader="train_loader"))

    test_loader = data_utils.load_partition_data(idx=i, data_partitioner=data_partitioner, batch_size=10, loader="test_loader")

    # Evaluate initial global model
    evaluate_acc(models[0], test_loader, idx=0, round=0)

    # Start training
    for round in range(rnd):

        for i in range(n_of_client):
            train_(models[i+1], train_loaders[i], optimizers[i], criterion, device, idx=i+1, localepoch=localepoch)
        for i in range(n_of_client):
            evaluate_acc(models[i+1], test_loader, idx=i+1, round=round+1)


        # Aggergate
        state_dicts = []
        for i in range(n_of_client+1):
            state_dicts.append(models[i].state_dict())

        for param in models[0].named_parameters():
            # Set global model's state_dict to zero
            state_dicts[0][param[0]].zero_()

            for i in range(n_of_client): # for each param, add client's state_dict
                torch.add(input=state_dicts[0][param[0]], other=state_dicts[i+1][param[0]], alpha=0.1, out=state_dicts[0][param[0]])
        

        # Aggregated models to global and clients
        for i in range(n_of_client+1):
            models[i].load_state_dict(state_dicts[0])

        # Evaluate Global model
        evaluate_acc(models[0], test_loader, idx=0, round=round+1)

    
    # writing results to output files
    accuracy_file = []
    accuracy_file.append('./Results/FedAvg_{}_rounds/FedAvg_niid_{}/FedAvg_Global_accuracy_niid_{}.csv'.format(rnd, iidness, iidness))
    os.makedirs(os.path.dirname(accuracy_file[0]), exist_ok=True)
    for i in range(10):
        accuracy_file.append('./Results/FedAvg_{}_rounds/FedAvg_niid_{}/FedAvg_Client_{}_accuracy_niid_{}.csv'.format(rnd, iidness, i+1, iidness))
        os.makedirs(os.path.dirname(accuracy_file[i+1]), exist_ok=True)

    for i in range(11):
        with open(accuracy_file[i], 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results[i])
