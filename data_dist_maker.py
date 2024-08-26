import random

def sizes_maker(num_client, num_class, shard_per_client, validation=False):
    """
    input
    num_client : number of clients
    num_class : number of classes (10 for cifar10)
    shard_per_client : number of shard per client (max number of class per client, if less than num_class)
                        if 0 is given, iid
    output
    sizes : num_client+1 X num_class matrix
    """
    random.seed(42)
    if not validation:
        o_classes = [[] for _ in range(num_client + 1)]
        sizes = [[] for _ in range(num_client + 1)]
        for i in range(1, num_client + 1):
            sizes[i].extend([0.0 for _ in range(num_class)])

        shard_per_class =  (shard_per_client * num_client) / num_class
        class_per_shard = round(1 / shard_per_class, 3)
        remaining_class = [1.0 for _ in range(num_class)]
        remaining_class_list = [i for i in range(num_class)]

        for client in range(1, num_client+1):
            for n in range(shard_per_client):
                choice = random.choice(remaining_class_list)
            
                remaining_class[choice] = round(remaining_class[choice] - class_per_shard, 3)
                if remaining_class[choice] < 0.0:
                    sizes[client][choice] += remaining_class[choice]
                sizes[client][choice] = round(sizes[client][choice] + class_per_shard, 3)
            
                if remaining_class[choice] <= 0.0:
                    remaining_class_list.remove(choice)

                if choice not in o_classes[client]:
                    o_classes[client].append(choice)

        return sizes, o_classes

    elif validation:

        o_classes = [[] for _ in range(num_client + 1)]
        sizes = [[] for _ in range(num_client + 2)]
        
        for i in range(1, num_client + 2):
            sizes[i].extend([0.0 for _ in range(num_class)])

        for i in range(num_class):
            sizes[num_client+1][i] = 0.1

        shard_per_class =  (shard_per_client * num_client) / num_class
        class_per_shard = round(0.9 / shard_per_class, 3)
        remaining_class = [0.9 for _ in range(num_class)]
        remaining_class_list = [i for i in range(num_class)]

        for client in range(1, num_client+1):
            for n in range(shard_per_client):
                choice = random.choice(remaining_class_list)
            
                remaining_class[choice] = round(remaining_class[choice] - class_per_shard, 3)
                if remaining_class[choice] < 0.0:
                    sizes[client][choice] += remaining_class[choice]
                sizes[client][choice] = round(sizes[client][choice] + class_per_shard, 3)
            
                if remaining_class[choice] <= 0.0:
                    remaining_class_list.remove(choice)

                if choice not in o_classes[client]:
                    o_classes[client].append(choice)

        return sizes, o_classes

if __name__=="__main__":
    sizes_maker(num_client=10, num_class=10, shard_per_client=2, validation=False)
