"""Quick script to plot the allocation data from the latest backtest."""
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import glob

def plot_allocations():
    # Load the performance data from the latest backtest
    reports_dir = 'reports'
    latest_file = None
    latest_time = 0
    
    # Find the most recent performance data file
    for file in glob.glob(os.path.join(reports_dir, 'performance_data_*.pkl')):
        file_time = os.path.getmtime(file)
        if file_time > latest_time:
            latest_time = file_time
            latest_file = file
    
    if not latest_file:
        print("No backtest results found in the reports directory.")
        return
    
    print(f"Loading data from: {latest_file}")
    
    try:
        # Load the performance data
        with open(latest_file, 'rb') as f:
            performance_data = pickle.load(f)
        
        # Extract allocations
        if 'Allocations' in performance_data and not performance_data['Allocations'].empty:
            allocations = performance_data['Allocations']
            
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
            
            # Show the plot
            plt.show()
            
            # Save the plot
            plot_path = os.path.join(reports_dir, 'allocations_plot.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            print(f"Plot saved to: {plot_path}")
            
        else:
            print("No allocation data found in the performance data.")
            if 'Allocations' in performance_data:
                print(f"Allocations type: {type(performance_data['Allocations'])}")
            else:
                print("'Allocations' key not found in performance data.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_allocations()
