import json
import os
import torch
from data.dataset import get_cifar10_loaders
from models.cnn import SimpleCNN
from training.train import train_model
import config

def train_final_model():
    """Train final model with best hyperparameters from optimization"""
    
    print("="*60)
    print("TRAINING FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS")
    print("="*60)
    
    # Load best hyperparameters from optimization
    results_path = 'results/optimization_results.json'
    
    if not os.path.exists(results_path):
        print(f"\n Error: {results_path} not found!")
        print("Please run optimization first: python -m training.optimize")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    best_params = results['best_params']
    
    print("\n✅ Loaded best hyperparameters from optimization:")
    print(json.dumps(best_params, indent=2))
    print(f"\nOptimization achieved: {results['best_accuracy']:.2f}% accuracy")
    
    # Device detection (Mac-friendly)
    if torch.cuda.is_available():
        device = 'cuda'
        print('Using device: NVIDIA GPU (CUDA)')
    elif torch.backends.mps.is_available():
        device = 'mps'
        print('Using device: Apple Silicon GPU (MPS)')
    else:
        device = 'cpu'
        print('Using device: CPU')
    
    print()
    
    # Load data with optimal batch size
    print(f"Loading CIFAR-10 with batch_size={best_params['batch_size']}...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=best_params['batch_size']
    )
    
    # Create model with optimal dropout
    print(f"Creating SimpleCNN with dropout={best_params['dropout']:.3f}...")
    model = SimpleCNN(num_classes=config.NUM_CLASSES, dropout=best_params['dropout'])
    
    # Get the optimal optimizer parameters
    lr = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    optimizer_name = best_params['optimizer']
    
    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {lr:.6f}")
    print(f"Weight decay: {weight_decay:.6f}")
    print()
    
    # Train with best hyperparameters for longer
    print("Starting training...")
    print("-"*60)
    
    best_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config.FINAL_TRAINING_EPOCHS,
        lr=lr,
        device=device,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay
    )
    
    print("-"*60)
    print(f'\n✅ Final Model Training Complete!')
    print(f'Final Test Accuracy: {best_acc:.2f}%')
    print(f'Improvement: {best_acc - results["best_accuracy"]:.2f}% (from longer training)')
    
    # Save final model with all metadata
    save_path = 'saved_models/final_model.pth'
    os.makedirs('saved_models', exist_ok=True)
    
    print(f'\nSaving model to {save_path}...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': best_params,
        'accuracy': best_acc,
        'optimization_accuracy': results['best_accuracy'],
        'training_epochs': config.NUM_EPOCHS,
        'optimization_epochs': config.NUM_EPOCHS,
        'final_training_epochs': config.FINAL_TRAINING_EPOCHS,
        'num_classes': config.NUM_CLASSES,
        'timestamp': results['timestamp']
    }, save_path)
    
    print(f'✅ Model saved successfully!')
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Optimization accuracy: {results['best_accuracy']:.2f}%")
    print(f"Final model accuracy:  {best_acc:.2f}%")
    print("="*60)
    
    return best_acc

if __name__ == '__main__':
    train_final_model()