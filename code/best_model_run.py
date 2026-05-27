import os
import sys
import torch
import datetime
import matplotlib.pyplot as plt 

# ==========================================
# 1. Import all required components from the core module
# ==========================================
from supply_chain_core import Logger, set_seed, train_and_predict, SupplyChainDataset

import __main__
__main__.SupplyChainDataset = SupplyChainDataset

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"best_model_log_{timestamp}.txt"
    sys.stdout = Logger(log_filename)
    print(f"System logs will also be saved to: {log_filename}")

    set_seed(42)

    # ==========================================
    # 2. Fill in the optimal parameters obtained from grid search
    # ==========================================
    best_params = {
        'num_layers': 4,        
        'hidden_dim': 64,       
        'lr': 0.01,             
        'weight_decay': 1e-5,   
        'dropout_rate': 0.2,    
        'neg_weight': 0.1       
    }

    BEST_EPOCHS = 2000  # Set your best epoch number here

    print(f"\nCurrent optimal parameter configuration:\n{best_params}")
    print(f"Configured training epochs: {BEST_EPOCHS}\n")

    # ==========================================
    # 3. Load locally cached data
    # ==========================================
    cache_file = './dataset/processed_supply_chain_130k.pt'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cached dataset {cache_file} not found. Please run preprocessing first!")
    
    print("Loading cached GPU dataset at high speed...")
    ds = torch.load(cache_file, map_location='cpu', weights_only=False) 
    ds.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 4. Start formal training and prediction
    # ==========================================
    output_folder = f'./best_model_results_{timestamp}'
    
    # [Modified here]: Receive the third return value loss_history
    trained_model, _, loss_history = train_and_predict(
        dataset=ds, 
        params=best_params, 
        epochs=BEST_EPOCHS, 
        output_dir=output_folder, 
        search_mode=False 
    )
    
    # ==========================================
    # 5. Plot and save the loss curve [Newly added code block]
    # ==========================================
    print("\nPlotting training loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, BEST_EPOCHS + 1),
        loss_history,
        label='Train Loss',
        color='#1f77b4',
        linewidth=2
    )

    plt.title('Training Loss Curve (Best Model)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save the figure to the output directory
    loss_plot_path = os.path.join(output_folder, 'train_loss_curve.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 6. Save model weights
    # ==========================================
    model_save_path = os.path.join(output_folder, 'supply_chain_gnn_best.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    
    print("\n" + "=" * 50)
    print(f"Model weights saved to: {model_save_path}")
    print(f"Per-epoch prediction CSV files and loss curve can be found in: {output_folder}")
    print("=" * 50 + "\n")
