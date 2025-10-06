import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

def train_model(model, train_loader, test_loader, epochs=2, 
                lr=0.001, device='cuda'):
    """Train a model and return best accuracy"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        epoch_start_time = time.time()  # Track time
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
        print(f'Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'saved_models/baseline_best.pth')

        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}% | Time: {epoch_time:.1f}s')
        
        # Estimate remaining time
        remaining_epochs = epochs - (epoch + 1)
        estimated_time = remaining_epochs * epoch_time / 60
        print(f'Estimated time remaining: {estimated_time:.1f} minutes')
    
    return best_acc

if __name__ == '__main__':
    from data.dataset import get_cifar10_loaders
    from models.cnn import SimpleCNN
    
    # Setup
    # Better device detection for Mac
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
    else:
        device = 'cpu'

    print(f'Using device: {device}')
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    # Create model
    model = SimpleCNN(num_classes=10, dropout=0.5)
    
    # Train
    best_acc = train_model(
        model, train_loader, test_loader,
        epochs=5, lr=0.001, device=device
    )
    
    print(f'\nBaseline Best Accuracy: {best_acc:.2f}%')
