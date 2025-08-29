import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset , Subset

import numpy as np


def get_dataloaders_iid(batch_size=32, num_clients=5,seed=0):
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    targets = torch.tensor(full_dataset.targets)
    mask = (targets == 0) | (targets == 1)
    indices = torch.where(mask)[0]

    full_dataset = Subset(full_dataset, indices)

    data_per_client = len(full_dataset) // num_clients

    indices = torch.randperm(len(full_dataset))
    client_datasets = [
        torch.utils.data.Subset(full_dataset, indices[i * data_per_client:(i + 1) * data_per_client])
        for i in range(num_clients)
    ]

    dataloaders = {
        cid: DataLoader(ds, batch_size=batch_size, shuffle=True)
        for cid, ds in enumerate(client_datasets)
    }

    return dataloaders


def get_dataloaders_group_iid_ap(batch_size=32, group1=[0, 1, 2], group2=[3, 4,5,6] , group3 = [7,8,9] ,seed = 0):
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    targets = torch.tensor(full_dataset.targets)
    mask = (targets == 0) | (targets == 1)
    indices = torch.where(mask)[0]
    full_dataset = Subset(full_dataset, indices)

    total_len = len(full_dataset)

    indices = torch.randperm(total_len)

    groups = [group1, group2,group3]
    total_clients = sum(len(g) for g in groups)
    
    group_sizes = [len(g) for g in groups]
    group_ratios = [s / total_clients for s in group_sizes]
    group_lengths = [int(r * total_len) for r in group_ratios]

    group_lengths[-1] = total_len - sum(group_lengths[:-1])

    dataloaders = {}
    start_idx = 0
    for group, length in zip(groups, group_lengths):
        group_indices = indices[start_idx:start_idx + length]
        start_idx += length

        client_len = len(group_indices) // len(group)
        for i, cid in enumerate(group):
            client_indices = group_indices[i * client_len: (i + 1) * client_len]
            subset = torch.utils.data.Subset(full_dataset, client_indices)
            dataloaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return dataloaders

def get_dataloaders_one_label_per_client(batch_size=32, num_clients=5,seed = 0):
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    targets = torch.tensor(full_dataset.targets)
    mask = (targets == 0) | (targets == 1)
    indices = torch.where(mask)[0]

    full_dataset = Subset(full_dataset, indices)

    # separate indices for each label
    label_indices = {label: [] for label in range(10)}
    for idx, (_, target) in enumerate(full_dataset):
        label_indices[target].append(idx)

    # one unique label per client
    if num_clients > 10:
        raise ValueError("There are only 10 unique labels in MNIST; num_clients must be <= 10.")
    #labels = [1,3,4,5,8]
    dataloaders = {}
    for cid in range(num_clients):
        label = cid  # Each client gets one unique label
        indices = label_indices[label]
        subset = Subset(full_dataset, indices)
        dataloaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return dataloaders

def get_dataloaders_label_per_ap(batch_size=32, ap_to_clients=None, ap_to_labels=None, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    targets = torch.tensor(full_dataset.targets)
    mask = (targets == 0) | (targets == 1)
    indices = torch.where(mask)[0]
    full_dataset = Subset(full_dataset, indices)


    label_indices = {label: [] for label in range(10)}
    for idx, (_, target) in enumerate(full_dataset):
        label_indices[target].append(idx)

    for label in label_indices:
        np.random.shuffle(label_indices[label])

    dataloaders = {}

    for ap_id, clients in ap_to_clients.items():
        labels = ap_to_labels[ap_id]

        combined_indices = []
        for label in labels:
            combined_indices.extend(label_indices[label])

        np.random.shuffle(combined_indices)

        split_size = len(combined_indices) // len(clients)

        for i, cid in enumerate(clients):
            client_indices = combined_indices[i * split_size: (i + 1) * split_size]
            subset = Subset(full_dataset, client_indices)
            dataloaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return dataloaders

def get_dataloaders_label_distribution_per_ap(batch_size=32, ap_to_clients=None, ap_to_labels=None, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    targets = torch.tensor(full_dataset.targets)
    mask = (targets == 0) | (targets == 1)
    indices = torch.where(mask)[0]
    full_dataset = Subset(full_dataset, indices)

    # we map  each label to its list of indices
    label_indices = {label: [] for label in range(10)}
    for idx, (_, target) in enumerate(full_dataset):
        label_indices[target].append(idx)

    # shuffle each label's indices
    for label in label_indices:
        np.random.shuffle(label_indices[label])

    label_pointers = {label: 0 for label in range(10)}

    dataloaders = {}

    for ap_id, clients in ap_to_clients.items():
        labels = ap_to_labels[ap_id]  # list of labels assigned to this AP

        total_clients = len(clients)

        # combined list of indices for the AP's labels
        combined_indices = []

        samples_per_label = np.random.randint(500, 1500, size=len(labels))  

        for i, label in enumerate(labels):
            start = label_pointers[label]
            end = start + samples_per_label[i]
            indices = label_indices[label][start:end]

            label_pointers[label] += len(indices)

            combined_indices.extend(indices)

        np.random.shuffle(combined_indices)

        split_size = len(combined_indices) // total_clients

        for i, cid in enumerate(clients):
            client_indices = combined_indices[i * split_size: (i + 1) * split_size]
            subset = Subset(full_dataset, client_indices)
            dataloaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return dataloaders

"""
def get_dataloaders_one_label_per_client_ap(batch_size=32, num_clients=5 , group1 , group2):
    torch.manual_seed(0)

    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download full MNIST training set once
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Separate indices for each label
    label_indices = {label: [] for label in range(10)}
    for idx, (_, target) in enumerate(full_dataset):
        label_indices[target].append(idx)

    # Assign one unique label per client
    if num_clients > 10:
        raise ValueError("There are only 10 unique labels in MNIST; num_clients must be <= 10.")
    #labels = [1,3,4,5,8]
    dataloaders = {}
    for cid in range(num_clients):
        
        label = cid  # Each client gets one unique label
        indices = label_indices[label]
        subset = Subset(full_dataset, indices)
        dataloaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return dataloaders

"""
