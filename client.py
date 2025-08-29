import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


"""
def compute_stochastic_gradient(model, dataloader, num_batches=3, l2_lambda=0.01):
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Initialize gradient accumulator
    gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    count = 0
    for x, y in dataloader:
        model.zero_grad()
        x = x.view(x.size(0), -1)  # Flatten the 28x28 images to vectors of size 784
        y_pred = model(x)
        
        ce_loss = criterion(y_pred, y)
        l2_penalty = sum((param ** 2).sum() for param in model.parameters())
        loss = ce_loss + l2_lambda * l2_penalty

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] += param.grad.detach()
            #gradients[name] = param.grad.detach().clone()

        count += 1
        if count >= num_batches:
            break
    for name in gradients:
        gradients[name] /= count

    return gradients
"""
def compute_stochastic_gradient(model, dataloader, num_batches=3, l2_lambda=0.01):
    model.train()
    criterion = nn.CrossEntropyLoss()

    gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    count = 0

    for x, y in dataloader:
        x = x.view(x.size(0), -1)
        y_pred = model(x)

        ce_loss = criterion(y_pred, y)
        l2_penalty = sum((param ** 2).sum() for param in model.parameters())
        loss = ce_loss + l2_lambda * l2_penalty

        # Compute gradients using autograd.grad
        grads = torch.autograd.grad(
            loss,
            [param for param in model.parameters()],
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        # accumulate over minibatches
        for (name, _), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                gradients[name] += grad.detach()

        count += 1
        if count >= num_batches:
            break

    # Average the gradients over all the mini batches
    for name in gradients:
        gradients[name] /= count

    return gradients

"""
def compute_stochastic_gradient(model, dataloader, l2_lambda=0.01):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for x, y in dataloader:
        x = x.requires_grad_() 
        y_pred = model(x)
        ce_loss = criterion(y_pred, y)

        l2_penalty = sum((param ** 2).sum() for param in model.parameters())
        loss = ce_loss + l2_lambda * l2_penalty

        gradients = torch.autograd.grad(
            loss, 
            [param for param in model.parameters()],
            create_graph=False,
            retain_graph=False,
            allow_unused=True  
        )

        grad_dict = {
            name: grad.detach().clone() if grad is not None else None
            for (name, param), grad in zip(model.named_parameters(), gradients)
        }

        return grad_dict  

"""
def compute_local_update(model, dataloader , num_batches=3):
    lr = 0.1 #local learning rate
    model_copy = copy.deepcopy(model)
    
    for k in range(num_batches):
        for name, param in model.named_parameters():
            grad = compute_stochastic_gradient(model , dataloader,1)
            with torch.no_grad():
                param -= lr * grad[name]
    delta = {name : model.state_dict()[name] - model_copy.state_dict()[name] for name in model.state_dict()}
    return delta

def compute_online_bias(X_m , delta):
    mean_X_m = np.mean(X_m)
    if mean_X_m > delta : 
        cm_chapeau = mean_X_m 
    else :
        cm_chapeau = delta
    return cm_chapeau


