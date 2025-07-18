import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List, Union

from ml.model import RecurrentModel


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_out: torch.Tensor, s: torch.Tensor):
        cross_entropy = nn.CrossEntropyLoss()
        s_out = s_out.reshape(-1, s_out.shape[2])
        s = s.reshape(-1)
        loss = cross_entropy(s_out, s)
        return loss


def train(
    model: RecurrentModel,
    dataloader: DataLoader,
    device: torch.device = torch.device('cpu'),
    epochs: int = 1000,
    lr: float = 0.01,
    print_losses: bool = False
) -> List:
    '''training loop'''

    # initialize loss and optimizer
    criterion = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # initialize loss list
    losses = []

    # training loop
    for t in range(epochs):

        # optimize over training data
        for (x, s, ids) in dataloader:
            
            # compute loss
            s_out = model.logits(x.to(device), ids.to(device))
            loss = criterion(s_out.to(device), s.to(device))
            
            # zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            if model.type in ['RNN', 'GRU']:  # gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # store and print latest loss
        if t % 10 == 0:
            losses.append(loss.item())
        if print_losses and (t % 100 == 0):
            print(f't = {t}  loss = {loss.item():.6f}')                
    
    return losses


def test(
    model: RecurrentModel,
    dataloader: DataLoader,
    device: torch.device = torch.device('cpu'),
    print_accuracy: bool = False,
    return_accuracy: bool = False
) -> Union[str, bool]:
    '''testing loop'''
    
    with torch.no_grad():
        
        # set model to evaluation mode
        model.eval()
    
        # perform evaluation
        total_correct = 0
        total_samples = 0
        for (x, s, ids) in dataloader:
            
            # send tensors to device
            x = x.to(device)
            s = s.to(device)
            ids = ids.to(device)

            # forward pass
            s_out = model.predict(x, ids).to(device)
    
            # check if correct, add to total samples
            total_correct += ((s_out == s).sum(1) == s.shape[1]).sum().item()
            total_samples += s.shape[0]

        # calculate overall accuracy
        accuracy = total_correct / total_samples

        # print and return if specified
        if print_accuracy:
            print(f'Accuracy on testing set: {accuracy:.4f}')
        if return_accuracy:
            return accuracy


def eval(
    model: RecurrentModel,
    training_dataloader: DataLoader,
    testing_dataloader: DataLoader,
    device: torch.device = torch.device('cpu'),
    epochs: int = 1000,
    lr: float = 0.01,
    log_interval: int = 10,
    print_loss_and_acc: bool = True
) -> Tuple[List, List, List]:
    '''evaluation loop including loss, training accuracies, and testing accuracies'''

    # initialize loss and optimizer (and scheduler if RNN)
    criterion = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if model.type == 'RNN':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=1000
        )

    # initialize loss and accuracy lists
    losses = []
    training_accs = []
    testing_accs = []

    # training loop
    for t in range(epochs):

        # compute and store training and testing accuracies
        if t % log_interval == 0:
            with torch.no_grad():
                model.eval()
                training_acc = test(model, training_dataloader, device=device, return_accuracy=True)
                testing_acc = test(model, testing_dataloader, device=device, return_accuracy=True)
                training_accs.append(training_acc)
                testing_accs.append(testing_acc)
            model.train()

        # optimize over training data
        for (x, s, ids) in training_dataloader:

            # send tensors to device
            x = x.to(device)
            s = s.to(device)
            ids = ids.to(device)

            # compute loss
            s_out = model.logits(x, ids).to(device)
            loss = criterion(s_out, s)
    
            # zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            if model.type in ['RNN', 'GRU']:  # gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # step the scheduler if using RNN
        if model.type == 'RNN':
            scheduler.step(loss)

        # store loss
        if t % log_interval == 0:
            losses.append(loss.item())

        # print loss and training/testing accuracies if specified
        if (t % log_interval == 0) and print_loss_and_acc:
            print(f't = {t}\nloss = {loss.item():.6f}, training_acc = {training_acc:.3f}, testing_acc = {testing_acc:.3f}\n') 
        
    return losses, training_accs, testing_accs