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

sizes, o_classes = sizes_maker(num_client=n_of_client, num_class=n_of_classes, shard_per_client=iidness, validation=True)

testmask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

results = [[] for _ in range(n_of_client+1)] # 1 server + clients

def restricted_softmax(logits, idx, device, alpha):

    m_logits = torch.ones_like(logits[0]).to(device) * alpha
    
    for c in o_classes[idx]:
        m_logits[c] = 1.0
    
    for i in range(len(logits)):
        logits[i] = torch.mul(logits[i], m_logits)

    return logits

def train_(model, train_loader, optimizer, criterion, device, idx, localepoch, alpha):

    print("Client {} training!".format(idx))
    model.train()
    for ep in range(localepoch):
        for data in tqdm.tqdm(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = restricted_softmax(outputs, idx, device, alpha)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

def validation_test(model, train_loader, device):
    model.eval()
    correct = [0. for _ in range(n_of_classes)]
    total_per_class = [0. for _ in range(n_of_classes)]

    with torch.no_grad():
        for data in train_loader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model.forward(images)
            _, predicted = torch.max(outputs, 1)
            for pred, target in zip(predicted, labels):
                total_per_class[target] += 1
                if pred == target:
                    correct[target] += 1

        correct = torch.tensor(correct)

        total_per_class = torch.tensor(total_per_class)
        torch.div(input=correct, other=total_per_class, out=correct)

        return correct

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
    alpha = 0.5 # alpha for RS

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
    validation_loader = data_utils.load_partition_data(idx=n_of_client+1, data_partitioner=data_partitioner, batch_size=10, loader="train_loader")

    # Build corect list for LL algorithm
    correct_lists = [torch.zeros((1, 10)) for i in range(n_of_client)]

    # Evaluate initial global model
    evaluate_acc(models[0], test_loader, idx=0, round=0)

    # Start training
    for round in range(rnd):

        for i in range(n_of_client):
            train_(models[i+1], train_loaders[i], optimizers[i], criterion, device, idx=i+1, localepoch=localepoch, alpha=alpha)
        for i in range(n_of_client):
            evaluate_acc(models[i+1], test_loader, idx=i+1, round=round+1)


        # evaluate with train data
        score = [[0. for _ in range(n_of_classes)] for _ in range(10)]
        correct_sum = [0. for _ in range(n_of_classes)]
        
        for i in range(n_of_client):
            correct_lists[i] = validation_test(models[i+1], validation_loader, device)
            #correct_lists[i] += 1.0

        # calculate score
        correct_sum = [0. for _ in range(n_of_classes)]
        for i in range(n_of_classes):
            for j in range(10):
                correct_sum[i] += correct_lists[j][i]
            correct_sum[i] += 0.0001

        score = torch.tensor(score)
        correct_sum = torch.tensor(correct_sum)

        for i in range(10):
            torch.div(input=correct_lists[i], other=correct_sum, out=score[i])


        # Aggergate
        state_dicts = []
        for i in range(n_of_client+1):
            state_dicts.append(models[i].state_dict())

        for param in models[0].named_parameters():
            if param[0] == 'linear_2.weight' or param[0] == 'linear_2.bias': # name of the last layer parameters

                for j in range(n_of_classes):
                    if correct_sum[j] <= 0.0001: continue

                    state_dicts[0][param[0]][j].zero_() # Set global model's state_dict to zero        
                    for i in range(n_of_client):
                        torch.add(input=state_dicts[0][param[0]][j], other=state_dicts[i+1][param[0]][j], alpha=score[i][j], out=state_dicts[0][param[0]][j])
            
            else:
                state_dicts[0][param[0]].zero_() # Set global model's state_dict to zero
                for i in range(n_of_client): # for each param, add client's state_dict
                    torch.add(input=state_dicts[0][param[0]], other=state_dicts[i+1][param[0]], alpha=0.1, out=state_dicts[0][param[0]])
        

        # Aggregated models to global and clients
        for i in range(n_of_client+1):
            models[i].load_state_dict(state_dicts[0])

        # Evaluate Global model
        evaluate_acc(models[0], test_loader, idx=0, round=round+1)

    
    # writing results to output files
    accuracy_file = []
    accuracy_file.append('./Results/FedCAC_{}_rounds/FedCAC_niid_{}/FedCAC_Global_accuracy_niid_{}.csv'.format(rnd, iidness, iidness))
    os.makedirs(os.path.dirname(accuracy_file[0]), exist_ok=True)
    for i in range(10):
        accuracy_file.append('./Results/FedCAC_{}_rounds/FedCAC_niid_{}/FedCAC_Client_{}_accuracy_niid_{}.csv'.format(rnd, iidness, i+1, iidness))
        os.makedirs(os.path.dirname(accuracy_file[i+1]), exist_ok=True)

    for i in range(11):
        with open(accuracy_file[i], 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results[i])
