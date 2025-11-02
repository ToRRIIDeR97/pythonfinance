import os
import io
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
from PIL import Image

class PDFReport(FPDF):
    def __init__(self, title="Portfolio Backtest Report"):
        super().__init__()  # Initialize FPDF parent class
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'C')
        self.ln(10)
        self.set_font('Arial', '', 12)
    
    def add_heading(self, text, level=1):
        self.set_font('Arial', 'B', 16 - (level * 2) if level < 4 else 12)
        self.cell(0, 10, text, 0, 1, 'L')
        self.ln(5)
    
    def add_paragraph(self, text):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, text)
        self.ln(5)
    
    def add_table(self, data, headers=None, col_widths=None):
        # Set font for table
        self.set_font('Arial', '', 10)
        
        # Calculate column widths if not provided
        if col_widths is None:
            col_widths = [self.w / (len(data[0]) + 1)] * len(data[0])
        
        # Add headers if provided
        if headers:
            self.set_font('Arial', 'B', 10)
            for i, header in enumerate(headers):
                self.cell(col_widths[i], 10, str(header), 1, 0, 'C')
            self.ln()
            self.set_font('Arial', '', 10)
        
        # Add data rows
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 10, str(item), 1, 0, 'C')
            self.ln()
        self.ln(5)
    
    def add_mpl_figure(self, fig, width=180):
        """Add a matplotlib figure to the PDF"""
        try:
            # Create a bytes buffer to save the figure
            buf = io.BytesIO()
            
            # Save the figure to the buffer
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            
            # Add the image to the PDF
            self.image(buf, x=10, w=width)
            self.ln(5)
            
            # Close the buffer
            buf.close()
            plt.close(fig)
            
        except Exception as e:
            print(f"[ERROR] In add_mpl_figure: {e}")
            # Add error message to PDF
            self.set_font('Arial', 'I', 10)
            self.set_text_color(255, 0, 0)
            self.cell(0, 10, f'Error generating chart: {str(e)}', 0, 1)
            self.set_text_color(0, 0, 0)
            self.ln(5)

def create_parameters_table(report, backtest_params):
    """Add a table of backtest parameters to the report"""
    report.set_font('Arial', 'B', 14)
    report.cell(0, 10, 'Backtest Parameters', 0, 1)
    report.ln(5)
    
    # Define parameter groups - only include optimization if it was run
    param_groups = {
        'Backtest Type': ['backtest_type'],
        'Date Range': ['start_date', 'end_date'],
        'Assets': ['assets'],
        'Trading': ['initial_capital', 'trading_cost_percent', 'slippage_percent']
    }
    
    # Only add optimization group if optimization was run
    if backtest_params.get('opt_objective') is not None:
        param_groups['Optimization'] = ['opt_objective', 'rebalance_freq']
    
    # Get all unique parameter names
    all_params = []
    for group_params in param_groups.values():
        all_params.extend(group_params)
    
    # Calculate column widths
    col_width = 95
    row_height = 8
    
    # Add parameters by group
    for group_name, params in param_groups.items():
        report.set_font('Arial', 'B', 12)
        report.cell(0, row_height, group_name, 0, 1)
        report.ln(2)
        
        # Add parameters in two columns
        for i, param in enumerate(params):
            if param in backtest_params:
                # First column (parameter name)
                report.set_font('Arial', 'B', 10)
                report.cell(col_width, row_height, param.replace('_', ' ').title(), 1, 0, 'L')
                
                # Second column (parameter value)
                report.set_font('Arial', '', 10)
                param_value = backtest_params[param]
                if isinstance(param_value, (list, tuple, dict)):
                    param_value = ', '.join(str(x) for x in param_value) if not isinstance(param_value, dict) else str(param_value)
                report.cell(col_width, row_height, str(param_value), 1, 1, 'L')
        
        report.ln(5)
    
    report.ln(5)

def generate_performance_summary(performance_data, title, initial_capital=100000):
    """Generate a performance summary table"""
    if not performance_data:
        return []
    
    # Extract performance metrics
    final_value = performance_data.get('FinalValue', 0)
    cagr = performance_data.get('CAGR', 0) * 100  # Convert to percentage
    max_drawdown = performance_data.get('MaxDrawdown', 0) * 100  # Convert to percentage
    volatility = performance_data.get('Volatility', 0) * 100  # Annualized volatility in %
    sharpe = performance_data.get('Sharpe', 0)
    sortino = performance_data.get('Sortino', 0)
    total_cost = performance_data.get('TotalTradingCost', 0)
    
    # Format the values
    metrics = [
        ["Initial Capital", f"${initial_capital:,.2f}"],
        ["Final Portfolio Value", f"${final_value:,.2f}"],
        ["Total Return", f"{((final_value - initial_capital) / initial_capital * 100):.2f}%"],
        ["CAGR", f"{cagr:.2f}%"],
        ["Annualized Volatility", f"{volatility:.2f}%"],
        ["Sharpe Ratio (Rf=0%)", f"{sharpe:.2f}"],
        ["Sortino Ratio (Rf=0%)", f"{sortino:.2f}"],
        ["Maximum Drawdown", f"{max_drawdown:.2f}%"],
        ["Total Trading Costs", f"${total_cost:,.2f}"]
    ]
    
    return metrics

def create_portfolio_plot(performance_data, title):
    """Create a portfolio value plot using matplotlib"""
    if not performance_data or 'ReturnsData' not in performance_data:
        print("[DEBUG] No ReturnsData in performance_data")
        return None
    
    returns_data = performance_data['ReturnsData']
    print(f"[DEBUG] ReturnsData columns: {returns_data.columns.tolist()}")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # Try different possible column names for portfolio value
        value_col = None
        possible_value_cols = ['Portfolio_Value', 'portfolio_value', 'value', 'Value', 'PortfolioValue']
        
        for col in possible_value_cols:
            if col in returns_data.columns:
                value_col = col
                break
                
        if value_col is None:
            print(f"[ERROR] Could not find portfolio value column in: {returns_data.columns.tolist()}")
            return None
            
        print(f"[DEBUG] Using column '{value_col}' for portfolio value")
        
        # Plot portfolio value
        ax.plot(returns_data.index, returns_data[value_col], 'b-', linewidth=2, label='Portfolio Value')
        
        # Set title and labels
        ax.set_title(f"{title} - Portfolio Value Over Time", fontsize=14)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Portfolio Value ($)", fontsize=10)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,.0f}".format(x)))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"[ERROR] Error creating portfolio plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_allocation_chart(performance_data, title):
    """Create an asset allocation chart using matplotlib"""
    if not performance_data or 'Allocations' not in performance_data:
        print("[DEBUG] No Allocations data in performance_data for allocation chart")
        return None
    
    try:
        allocations = performance_data['Allocations']
        print(f"[DEBUG] Allocations data type: {type(allocations)}")
        print(f"[DEBUG] Allocations columns: {allocations.columns.tolist()}")
        
        if allocations is None or allocations.empty:
            print("[DEBUG] Allocations DataFrame is empty")
            return None
            
        # Create the plot with a larger figure size
        plt.figure(figsize=(14, 8))
        
        # Plot each asset's allocation over time as a line
        for column in allocations.columns:
            plt.plot(allocations.index, allocations[column] * 100, 
                    label=column, linewidth=2)
        
        # Add title and labels with better formatting
        plt.title('Asset Allocation Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Allocation (%)', fontsize=12)
        
        # Add grid with better styling
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend outside the plot area
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        print(f"[ERROR] Error creating allocation chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_drawdown_plot(performance_data, title):
    """Create a drawdown plot using matplotlib"""
    if not performance_data or 'ReturnsData' not in performance_data:
        print("[DEBUG] No ReturnsData in performance_data for drawdown plot")
        return None
    
    returns_data = performance_data['ReturnsData']
    print(f"[DEBUG] Drawdown - ReturnsData columns: {returns_data.columns.tolist()}")
    
    try:
        # Try different possible column names for portfolio value
        value_col = None
        possible_value_cols = ['Portfolio_Value', 'portfolio_value', 'value', 'Value', 'PortfolioValue']
        
        for col in possible_value_cols:
            if col in returns_data.columns:
                value_col = col
                break
                
        if value_col is None:
            print(f"[ERROR] Could not find portfolio value column for drawdown in: {returns_data.columns.tolist()}")
            return None
            
        print(f"[DEBUG] Using column '{value_col}' for drawdown calculation")
        
        # Calculate running maximum
        running_max = returns_data[value_col].cummax()
        
        # Calculate drawdown
        drawdown = (returns_data[value_col] - running_max) / running_max * 100
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Fill between for drawdown
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        
        # Plot the drawdown line
        ax.plot(drawdown.index, drawdown, 'r-', linewidth=1, alpha=0.8)
        
        # Add zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        
        # Set title and labels
        ax.set_title(f"{title} - Drawdown Over Time", fontsize=14)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Drawdown (%)", fontsize=10)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}%".format(x)))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"[ERROR] Error creating drawdown plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_report(performance_data, initial_capital, output_path="backtest_report.pdf", 
                   title="Portfolio Backtest Report", backtest_params=None):
    """
    Generate a PDF report with performance metrics and visualizations
    
    Args:
        performance_data (dict): Dictionary containing performance metrics and data
        initial_capital (float): Initial portfolio value
        output_path (str): Path to save the PDF report
        title (str): Title for the report
        backtest_params (dict, optional): Dictionary of backtest parameters
    
    Returns:
        str: Path to the generated PDF report
    """
    print("\n=== Starting PDF Report Generation ===")
    print(f"Current working directory: {os.getcwd()}")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Ensuring reports directory exists: {output_dir}")
    print(f"Reports directory is writable: {os.access(output_dir, os.W_OK) if os.path.exists(output_dir) else 'Directory does not exist'}")
    
    # Default backtest parameters if not provided
    if backtest_params is None:
        backtest_params = {}
    
    if not performance_data:
        raise ValueError("No performance data provided")
    
    # Set matplotlib backend to 'Agg' to avoid display issues
    matplotlib.use('Agg')
    
    try:
        # Create PDF report
        report = PDFReport(title=title)
        
        # Add title and date
        report.add_page()
        report.set_font('Arial', 'B', 16)
        report.cell(0, 10, title, 0, 1, 'C')
        report.ln(5)
        
        # Add generation date
        report.set_font('Arial', '', 12)
        report.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        report.ln(10)
        
        # Add backtest parameters if available
        if backtest_params:
            create_parameters_table(report, backtest_params)
            report.add_page()
        
        # Add performance summary
        report.set_font('Arial', 'B', 14)
        report.cell(0, 10, 'Performance Summary', 0, 1)
        report.ln(5)
        
        # Add performance metrics
        metrics = [
            ('Initial Capital', f"${initial_capital:,.2f}"),
            ('Final Portfolio Value', f"${performance_data.get('FinalValue', 0):,.2f}"),
            ('CAGR', f"{performance_data.get('CAGR', 0)*100:.2f}%"),
            ('Annualized Volatility', f"{performance_data.get('Volatility', 0)*100:.2f}%"),
            ('Sharpe Ratio', f"{performance_data.get('Sharpe', 0):.2f}"),
            ('Sortino Ratio', f"{performance_data.get('Sortino', 0):.2f}"),
            ('Maximum Drawdown', f"{performance_data.get('MaxDrawdown', 0)*100:.2f}%"),
            ('Total Trading Costs', f"${performance_data.get('TotalTradingCost', 0):,.2f}")
        ]
        
        # Create a table with metrics
        col_width = 90
        row_height = 10
        for i, (label, value) in enumerate(metrics):
            if i % 2 == 0 and i > 0:
                report.ln(row_height)
            report.set_font('Arial', 'B', 10)
            report.cell(col_width, row_height, label, 1, 0, 'L')
            report.set_font('Arial', '', 10)
            report.cell(col_width, row_height, str(value), 1, 0, 'R')
        
        # Add a page break
        report.add_page()
        
        # Add portfolio value plot
        report.set_font('Arial', 'B', 14)
        report.cell(0, 10, 'Portfolio Value Over Time', 0, 1)
        report.ln(5)
        
        portfolio_plot = create_portfolio_plot(performance_data, title)
        if portfolio_plot:
            # Save the plot to a BytesIO object
            img_data = io.BytesIO()
            portfolio_plot.savefig(img_data, format='png', bbox_inches='tight', dpi=100)
            plt.close(portfolio_plot)
            
            # Add the image to the PDF
            img_data.seek(0)
            report.image(img_data, x=10, w=190)
            img_data.close()
        
        # Add drawdown plot
        report.add_page()
        report.set_font('Arial', 'B', 14)
        report.cell(0, 10, 'Drawdown Over Time', 0, 1)
        report.ln(5)
        
        drawdown_plot = create_drawdown_plot(performance_data, title)
        if drawdown_plot:
            # Save the plot to a BytesIO object
            img_data = io.BytesIO()
            drawdown_plot.savefig(img_data, format='png', bbox_inches='tight', dpi=100)
            plt.close(drawdown_plot)
            
            # Add the image to the PDF
            img_data.seek(0)
            report.image(img_data, x=10, w=190)
            img_data.close()
        
        # Add allocation chart if data is available
        if 'Allocations' in performance_data and performance_data['Allocations'] is not None:
            report.add_page()
            report.set_font('Arial', 'B', 14)
            report.cell(0, 10, 'Asset Allocation Over Time', 0, 1)  # Updated title
            report.ln(5)
            
            # Create the allocation chart using our new function
            allocation_chart = create_allocation_chart(performance_data, title)
            if allocation_chart:
                try:
                    # Save the plot to a BytesIO object
                    img_data = io.BytesIO()
                    allocation_chart.savefig(img_data, format='png', bbox_inches='tight', dpi=100)
                    plt.close(allocation_chart)
                    
                    # Add the image to the PDF
                    img_data.seek(0)
                    report.image(img_data, x=10, w=190)
                    img_data.close()
                except Exception as e:
                    print(f"[ERROR] Error adding allocation chart to PDF: {e}")
                    import traceback
                    traceback.print_exc()
                    report.add_paragraph("Error generating allocation chart")
        
        # Add allocations table if available
        if 'Allocations' in performance_data and not performance_data['Allocations'].empty:
            report.add_page()
            report.set_font('Arial', 'B', 14)
            report.cell(0, 10, 'Final Asset Allocations', 0, 1)
            report.ln(5)
            
            # Get allocations data
            allocations = performance_data['Allocations']
            
            # Table header
            report.set_font('Arial', 'B', 10)
            report.cell(100, 10, 'Asset', 1, 0, 'C')
            report.cell(80, 10, 'Allocation %', 1, 1, 'C')
            
            # Table rows
            report.set_font('Arial', '', 10)
            
            # Handle different types of allocations data
            if hasattr(allocations, 'items'):
                # For dict-like objects (including pandas Series)
                for asset, alloc in allocations.items():
                    report.cell(100, 10, str(asset), 1, 0, 'L')
                    alloc_value = float(alloc.iloc[0]) if hasattr(alloc, 'iloc') else float(alloc)
                    report.cell(80, 10, f"{alloc_value*100:.2f}%", 1, 1, 'R')
            elif hasattr(allocations, 'iterrows'):
                # For DataFrames
                for _, row in allocations.iterrows():
                    if len(row) >= 2:
                        report.cell(100, 10, str(row[0]), 1, 0, 'L')
                        report.cell(80, 10, f"{float(row[1])*100:.2f}%", 1, 1, 'R')
            elif isinstance(allocations, (list, tuple)):
                # For lists/tuples of (asset, allocation) pairs
                for item in allocations:
                    if len(item) >= 2:
                        asset, alloc = item[0], item[1]
                        report.cell(100, 10, str(asset), 1, 0, 'L')
                        report.cell(80, 10, f"{float(alloc)*100:.2f}%", 1, 1, 'R')
                    else:
                        print(f"[WARNING] Unexpected allocation item format: {item}")
            else:
                print(f"[WARNING] Unsupported allocations type: {type(allocations)}")
        
        # Save the report
        report.output(output_path)
        
        print(f"[SUCCESS] Report saved to: {os.path.abspath(output_path)}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
        return os.path.abspath(output_path)
    
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        raise
