import os
import torch
from training.train import train_model


if __name__ == '__main__':
    from data.dataset import get_cifar10_loaders
    from models.cnn import SimpleCNN
    import config
    
    # Better device detection for Mac
    if torch.cuda.is_available():
        device = 'cuda'
        print('Using NVIDIA GPU (CUDA)')
    elif torch.backends.mps.is_available():
        device = 'mps'
        print('Using Apple Silicon GPU (MPS)')
    else:
        device = 'cpu'
        print('Using CPU')
    
    print(f'Using device: {device}')
    
    # Load data
    print('\nLoading CIFAR-10 dataset...')
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    # Create model
    print('Creating model...')
    model = SimpleCNN(num_classes=config.NUM_CLASSES, dropout=0.5)
    
    # Train
    print('\nStarting baseline training...')
    best_acc = train_model(
        model, train_loader, test_loader,
        epochs=config.NUM_EPOCHS, lr=0.001, device=device,
        optimizer_name='Adam', weight_decay=0.0
    )
    
    print(f'\n{"="*50}')
    print(f'Training Complete!')
    print(f'Baseline Best Accuracy: {best_acc:.2f}%')
    print(f'{"="*50}')

     # Save final model with all metadata
    save_path = 'saved_models/baseline_model.pth'
    os.makedirs('saved_models', exist_ok=True)
    
    print(f'\nSaving model to {save_path}...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': best_acc,
        'training_epochs': config.NUM_EPOCHS,
        'optimization_epochs': config.NUM_EPOCHS,
        'num_classes': config.NUM_CLASSES,
    }, save_path)
    
    print(f'✅ Model saved successfully!')