# import sys
# import os

# # Add project root to path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_cifar10_loaders
from models.cnn import SimpleCNN
import json
from datetime import datetime
import config
import time

def objective(trial):
    """Optuna objective function to maximize"""
    
    # Suggest hyperparameters FROM CONFIG
    lr = trial.suggest_float(
        'learning_rate',
        config.SEARCH_SPACE['learning_rate']['low'],
        config.SEARCH_SPACE['learning_rate']['high'],
        log=config.SEARCH_SPACE['learning_rate']['log']
    )
    
    batch_size = trial.suggest_categorical(
        'batch_size',
        config.SEARCH_SPACE['batch_size']['choices']
    )
    
    optimizer_name = trial.suggest_categorical(
        'optimizer',
        config.SEARCH_SPACE['optimizer']['choices']
    )
    
    weight_decay = trial.suggest_float(
        'weight_decay',
        config.SEARCH_SPACE['weight_decay']['low'],
        config.SEARCH_SPACE['weight_decay']['high'],
        log=config.SEARCH_SPACE['weight_decay']['log']
    )
    
    dropout = trial.suggest_float(
        'dropout',
        config.SEARCH_SPACE['dropout']['low'],
        config.SEARCH_SPACE['dropout']['high'],
        log=config.SEARCH_SPACE['dropout']['log']
    )
    
    # Device detection
    if torch.cuda.is_available():
        device = 'cuda'
    elif config.USE_MPS and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # Load data with suggested batch size
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    
    # Create model
    model = SimpleCNN(num_classes=10, dropout=dropout).to(device)
    
    # Setup optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:  # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop - USE CONFIG FOR NUM_EPOCHS
    num_epochs = config.NUM_EPOCHS
    
    for epoch in range(num_epochs):
        # Training
        epoch_start_time = time.time()  # Track time
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        
        # Report intermediate value for pruning
        trial.report(accuracy, epoch)
        
        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}: Test Accuracy = {accuracy:.2f}% | Time: {epoch_time:.1f}s')
        
        # Estimate remaining time
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time = remaining_epochs * epoch_time / 60
        print(f'Estimated time remaining: {estimated_time:.1f} minutes')
    
    # Return final accuracy (Optuna will maximize this)
    return accuracy

def run_optimization(n_trials=None, timeout=None):
    """
    Run Optuna optimization
    
    Args:
        n_trials: Number of trials (default: from config)
        timeout: Timeout in seconds (default: from config)
    """
    
    # Use config values if not provided
    if n_trials is None:
        n_trials = config.MAX_TRIALS
    if timeout is None:
        timeout = config.TIMEOUT
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize accuracy
        sampler=TPESampler(seed=42),  # Bayesian optimization
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    print("Starting Bayesian Optimization...")
    print(f"Running {n_trials} trials")
    if timeout:
        print(f"Timeout: {timeout/3600:.1f} hours")
    else:
        print("No timeout - will complete all trials")
    print(f"Training {config.NUM_EPOCHS} epochs per trial\n")
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*50)
    print("Optimization Complete!")
    print("="*50)
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Accuracy: {study.best_value:.2f}%")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_trial': study.best_trial.number,
        'best_accuracy': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_epochs': config.NUM_EPOCHS,
            'max_trials': config.MAX_TRIALS,
            'search_space': config.SEARCH_SPACE
        }
    }
    
    with open('results/optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save study for visualization
    study.trials_dataframe().to_csv('results/trials.csv', index=False)
    
    return study

if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    # Now uses config.py values by default!
    study = run_optimization()
    