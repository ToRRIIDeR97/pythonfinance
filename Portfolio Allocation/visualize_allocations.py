"""Script to visualize asset allocations from backtest results."""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import importlib.util

def load_allocations():
    """Load allocations from the backtest results by running main.py with appropriate args."""
    try:
        # Import the main module to access its functionality
        from main import run_optimization
        
        # Set up parameters similar to what would be passed from command line
        class Args:
            def __init__(self):
                self.overall_start_date = '2020-01-01'
                self.overall_end_date = '2024-12-31'
                self.opt_objective = 'sharpe'
                self.rebalance_freq = 'monthly'
                self.run_optimization = True
        
        print(f"Running backtest from {Args().overall_start_date} to {Args().overall_end_date}...")
        
        # Run the optimization which will generate the allocations
        result = run_optimization(Args())
        
        if 'Allocations' in result and not result['Allocations'].empty:
            allocations = result['Allocations']
            print("\nAllocations data loaded successfully:")
            print(f"Shape: {allocations.shape}")
            print("\nFirst 5 rows:")
            print(allocations.head())
            print("\nLast 5 rows:")
            print(allocations.tail())
            
            # Plot the allocations
            plt.figure(figsize=(14, 8))
            
            # Plot each asset's allocation over time
            for column in allocations.columns:
                plt.plot(allocations.index, allocations[column] * 100, label=column, linewidth=2)
            
            plt.title('Asset Allocation Over Time', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Allocation (%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save the figure
            output_dir = 'reports'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'allocations_plot.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"\nPlot saved to: {output_path}")
            
            # Show the plot
            plt.show()
            
        else:
            print("No allocation data found in backtest results.")
            if 'Allocations' in result:
                print(f"Allocations type: {type(result['Allocations'])}")
            else:
                print("'Allocations' key not found in results.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_allocations()
