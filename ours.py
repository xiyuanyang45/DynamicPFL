import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
from options import parse_args
from data import *
from net import *
from tqdm import tqdm
from utils import compute_noise_multiplier, compute_fisher_diag
from tqdm.auto import trange, tqdm
import copy
import sys
import random
from torch.optim import Optimizer
import datetime




args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
target_epsilon = args.target_epsilon
target_delta = args.target_delta
clipping_bound = args.clipping_bound
dataset = args.dataset
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.store == True:
    saved_stdout = sys.stdout
    file = open(
        f'./txt/{args.dirStr}/'
        f'dataset {dataset} '
        f'--num_clients {num_clients} '
        f'--local_epoch {local_epoch} '
        f'--global_epoch {global_epoch} '
        f'--batch_size {batch_size} '
        f'--target_epsilon {target_epsilon} '
        f'--target_delta {target_delta} '
        f'--clipping_bound {clipping_bound} '
        f'--fisher_threshold {args.fisher_threshold} '
        f'--lambda_1 {args.lambda_1} '
        f'--lambda_2 {args.lambda_2} '
        f'--lr {args.lr} '
        f'--alpha {args.dir_alpha}'
        f'.txt'
        ,'a'
        )
    sys.stdout = file

def local_update(model, dataloader, global_model):


    fisher_threshold = args.fisher_threshold
    model = model.to(device)
    global_model = global_model.to(device)

    w_glob = [param.clone().detach() for param in global_model.parameters()]

    fisher_diag = compute_fisher_diag(model, dataloader)


    u_loc, v_loc = [], []
    for param, fisher_value in zip(model.parameters(), fisher_diag):
        u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
        u_loc.append(u_param)
        v_loc.append(v_param)

    u_glob, v_glob = [], []
    for global_param, fisher_value in zip(global_model.parameters(), fisher_diag):
        u_param = (global_param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (global_param * (fisher_value <= fisher_threshold)).clone().detach()
        u_glob.append(u_param)
        v_glob.append(v_param)

    for u_param, v_param, model_param in zip(u_loc, v_glob, model.parameters()):
        model_param.data = u_param + v_param

    saved_u_loc = [u.clone() for u in u_loc]

    def custom_loss(outputs, labels, param_diffs, reg_type):
        ce_loss = F.cross_entropy(outputs, labels)
        if reg_type == "R1":
            reg_loss = (args.lambda_1 / 2) * torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))

        elif reg_type == "R2":
            C = args.clipping_bound
            norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
            reg_loss = (args.lambda_2 / 2) * torch.norm(norm_diff - C)

        else:
            raise ValueError("Invalid regularization type")

        return ce_loss + reg_loss
    

    optimizer1 = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.local_epoch):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer1.zero_grad()
            outputs = model(data)
            param_diffs = [u_new - u_old for u_new, u_old in zip(model.parameters(), w_glob)]
            loss = custom_loss(outputs, labels, param_diffs, "R1")
            loss.backward()
            with torch.no_grad():
                for model_param, u_param in zip(model.parameters(), u_loc):
                    model_param.grad *= (u_param != 0)
            optimizer1.step()
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.local_epoch):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer2.zero_grad()
            outputs = model(data)
            param_diffs = [model_param - w_old for model_param, w_old in zip(model.parameters(), w_glob)]
            loss = custom_loss(outputs, labels, param_diffs, "R2")
            loss.backward()
            with torch.no_grad():
                for model_param, v_param in zip(model.parameters(), v_glob):
                    model_param.grad *= (v_param != 0)
            optimizer2.step()

    with torch.no_grad():
        update = [(new_param - old_param).clone() for new_param, old_param in zip(model.parameters(), w_glob)]


    model = model.to('cpu')
    return update








def test(client_model, client_testloader):
    client_model.eval()
    client_model = client_model.to(device)

    num_data = 0


    correct = 0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = client_model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_data += labels.size(0)
    
    accuracy = 100.0 * correct / num_data

    client_model = client_model.to('cpu')

    return accuracy

def main():

    mean_acc_s = []
    acc_matrix = []
    if dataset == 'MNIST':

        train_dataset, test_dataset = get_mnist_datasets()
        clients_train_set = get_clients_datasets(train_dataset, num_clients)
        client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
        clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in clients_train_set]
        clients_test_loaders = [DataLoader(test_dataset) for i in range(num_clients)]

        clients_models = [mnistNet() for _ in range(num_clients)]
        global_model = mnistNet()
    elif dataset == 'CIFAR10':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)

        clients_models = [cifar10Net() for _ in range(num_clients)]
        global_model = cifar10Net()
    elif dataset == 'FEMNIST':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)

        clients_models = [femnistNet() for _ in range(num_clients)]
        global_model = femnistNet()
    elif dataset == 'SVHN':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients)

        clients_models = [SVHNNet() for _ in range(num_clients)]
        global_model = SVHNNet()
    else:
        print('undifined dataset')
        assert 1==0
    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())
    noise_multiplier = compute_noise_multiplier(target_epsilon, target_delta, global_epoch, local_epoch, batch_size, client_data_sizes)
    print(noise_multiplier)
    if args.no_noise:
        noise_multiplier = 0
    for epoch in trange(global_epoch):
        sampled_client_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))
        sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
        clients_model_updates = []
        clients_accuracies = []
        for idx, (client_model, client_trainloader, client_testloader) in enumerate(zip(sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders)):
            if not args.store:
                tqdm.write(f'client:{idx+1}/{args.num_clients}')
            client_update = local_update(client_model, client_trainloader, global_model)
            clients_model_updates.append(client_update)
            accuracy = test(client_model, client_testloader)
            clients_accuracies.append(accuracy)
        print(clients_accuracies)
        mean_acc_s.append(sum(clients_accuracies)/len(clients_accuracies))
        acc_matrix.append(clients_accuracies)
        sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
        sampled_client_weights = [
            sampled_client_data_size / sum(sampled_client_data_sizes)
            for sampled_client_data_size in sampled_client_data_sizes
        ]
        clipped_updates = []
        for idx, client_update in enumerate(clients_model_updates):
            if not args.no_clip:
                norm = torch.sqrt(sum([torch.sum(param ** 2) for param in client_update]))
                clip_rate = max(1, (norm / clipping_bound))
                clipped_update = [(param / clip_rate) for param in client_update]
            else:
                clipped_update = client_update
            clipped_updates.append(clipped_update)
        noisy_updates = []
        for clipped_update in clipped_updates:
            noise_stddev = torch.sqrt(torch.tensor((clipping_bound**2) * (noise_multiplier**2) / num_clients))
            noise = [torch.randn_like(param) * noise_stddev for param in clipped_update]
            noisy_update = [clipped_param + noise_param for clipped_param, noise_param in zip(clipped_update, noise)]
            noisy_updates.append(noisy_update)
        aggregated_update = [
            torch.sum(
                torch.stack(
                    [
                        noisy_update[param_index] * sampled_client_weights[idx]
                        for idx, noisy_update in enumerate(noisy_updates)
                    ]
                ),
                dim=0,
            )
            for param_index in range(len(noisy_updates[0]))
        ]
        with torch.no_grad():
            for global_param, update in zip(global_model.parameters(), aggregated_update):
                global_param.add_(update)
    char_set = '1234567890abcdefghijklmnopqrstuvwxyz'
    ID = ''
    for ch in random.sample(char_set, 5):
        ID = f'{ID}{ch}'
    print(
        f'===============================================================\n'
        f'task_ID : '
        f'{ID}\n'
        f'main_yxy\n'
        f'noise_multiplier : {noise_multiplier}\n'
        f'mean accuracy : \n'
        f'{mean_acc_s}\n'
        f'acc matrix : \n'
        f'{torch.tensor(acc_matrix)}\n'
        f'===============================================================\n'
    )


if __name__ == '__main__':
    main()

