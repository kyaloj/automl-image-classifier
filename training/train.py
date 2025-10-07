import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

def train_model(model, train_loader, test_loader, epochs=10, 
                lr=0.001, device='mps', optimizer_name='Adam', weight_decay=0.0):
    """
    Train a model and return best accuracy
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on ('cuda', 'mps', or 'cpu')
        optimizer_name: Optimizer to use ('Adam', 'SGD', or 'AdamW')
        weight_decay: Weight decay for regularization
    
    Returns:
        best_acc: Best test accuracy achieved
    """
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer based on name
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/len(train_loader):.3f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}% | Time: {epoch_time:.1f}s')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            
        
        # Estimate remaining time
        if epoch < epochs - 1:
            remaining_epochs = epochs - (epoch + 1)
            estimated_time = remaining_epochs * epoch_time / 60
            print(f'  Estimated time remaining: {estimated_time:.1f} minutes')
    
    return best_acc
