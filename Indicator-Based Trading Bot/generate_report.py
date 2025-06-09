import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from datetime import datetime

# Custom styles for the report
def get_custom_styles():
    styles = getSampleStyleSheet()
    
    # Helper function to add style if it doesn't exist
    def add_style(style_name, **kwargs):
        if style_name not in styles:
            styles.add(ParagraphStyle(name=style_name, **kwargs))
    
    # Title style
    add_style(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=colors.HexColor('#2E4053')
    )
    
    # Subtitle style
    add_style(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.HexColor('#34495E')
    )
    
    # Section header style
    add_style(
        'Section',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2874A6')
    )
    
    # Normal text style
    add_style(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=10
    )
    
    # Table header style
    add_style(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    
    return styles

def create_performance_table(data, styles):
    """Create a table with performance metrics"""
    table_data = [
        ["Metric", "RSI", "MACD", "MA Cross", "Converging"],
        ["Total Return (%)", 
         f"{data['rsi']['total_return']*100:.2f}",
         f"{data['macd']['total_return']*100:.2f}",
         f"{data['ma']['total_return']*100:.2f}",
         f"{data['conv']['total_return']*100:.2f}"],
        ["Sharpe Ratio", 
         f"{data['rsi']['sharpe']:.2f}",
         f"{data['macd']['sharpe']:.2f}",
         f"{data['ma']['sharpe']:.2f}",
         f"{data['conv']['sharpe']:.2f}"],
        ["Max Drawdown (%)", 
         f"{data['rsi']['max_dd']*100:.2f}",
         f"{data['macd']['max_dd']*100:.2f}",
         f"{data['ma']['max_dd']*100:.2f}",
         f"{data['conv']['max_dd']*100:.2f}"],
        ["Win Rate (%)", 
         f"{data['rsi']['win_rate']*100:.1f}",
         f"{data['macd']['win_rate']*100:.1f}",
         f"{data['ma']['win_rate']*100:.1f}",
         f"{data['conv']['win_rate']*100:.1f}"],
        ["Final Value ($)", 
         f"{data['rsi']['final_value']:,.2f}",
         f"{data['macd']['final_value']:,.2f}",
         f"{data['ma']['final_value']:,.2f}",
         f"{data['conv']['final_value']:,.2f}"]
    ]
    
    # Create table
    table = Table(table_data, colWidths=[120, 80, 80, 80, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),  # Header background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EBF5FB')),  # Table body background
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#D6EAF8')),  # Grid lines
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#2E86C1')),  # Table border
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    # Highlight best performer for each metric
    metrics = ['Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)']
    for i, metric in enumerate(metrics):
        row_idx = i + 1  # +1 because of header row
        values = [float(x.replace(',', '')) for x in table_data[row_idx][1:]]
        if i == 2:  # For win rate, higher is better
            best_col = np.argmax(values) + 1  # +1 because of first column
        else:
            best_col = np.argmax(values) + 1  # +1 because of first column
        
        table.setStyle(TableStyle([
            ('BACKGROUND', (best_col, row_idx), (best_col, row_idx), colors.HexColor('#ABEBC6')),  # Light green
            ('TEXTCOLOR', (best_col, row_idx), (best_col, row_idx), colors.HexColor('#145A32')),  # Dark green
            ('FONTNAME', (best_col, row_idx), (best_col, row_idx), 'Helvetica-Bold'),
        ]))
    
    return table

def create_equity_curve_plot(portfolio_values, output_path):
    """Create equity curve plot and save to file"""
    plt.figure(figsize=(12, 6))
    
    for label, values in portfolio_values.items():
        plt.plot(values.index, values, label=label, linewidth=2)
    
    plt.title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_path, 'equity_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_drawdown_plot(drawdowns, output_path):
    """Create drawdown plot and save to file"""
    plt.figure(figsize=(12, 4))
    
    for label, values in drawdowns.items():
        plt.fill_between(values.index, values * 100, 0, alpha=0.3, label=label)
    
    plt.title('Drawdown', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_path, 'drawdown.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_pdf_report(output_path, report_data):
    """Generate the PDF report"""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        os.path.join(output_path, 'trading_performance_report.pdf'),
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    # Get styles
    styles = get_custom_styles()
    
    # Create story (content)
    story = []
    
    # Add title and date
    story.append(Paragraph("Trading Strategy Performance Report", styles['Title']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
        styles['Subtitle']
    ))
    story.append(Spacer(1, 30))
    
    # Add executive summary
    story.append(Paragraph("Executive Summary", styles['Section']))
    story.append(Paragraph(
        "This report presents the performance analysis of four trading strategies: "
        "RSI, MACD, Moving Average Crossover, and a Converging Signals strategy that "
        "combines all three. The analysis includes key performance metrics, equity curves, "
        "and drawdown analysis.",
        styles['BodyText']
    ))
    
    # Add performance metrics table
    story.append(Spacer(1, 20))
    story.append(Paragraph("Performance Metrics", styles['Section']))
    story.append(Spacer(1, 10))
    
    # Create and add performance table
    table = create_performance_table(report_data, styles)
    story.append(table)
    
    # Add equity curve
    story.append(PageBreak())
    story.append(Paragraph("Equity Curve", styles['Section']))
    story.append(Spacer(1, 10))
    
    # Create and add equity curve plot
    equity_plot_path = create_equity_curve_plot(
        {
            'RSI': report_data['rsi']['equity_curve'],
            'MACD': report_data['macd']['equity_curve'],
            'MA Cross': report_data['ma']['equity_curve'],
            'Converging': report_data['conv']['equity_curve']
        },
        output_path
    )
    
    story.append(Image(equity_plot_path, width=500, height=250))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Figure 1: Equity curves of all strategies over the backtest period.",
        styles['BodyText']
    ))
    
    # Add drawdown analysis
    story.append(PageBreak())
    story.append(Paragraph("Drawdown Analysis", styles['Section']))
    story.append(Spacer(1, 10))
    
    # Create and add drawdown plot
    drawdown_plot_path = create_drawdown_plot(
        {
            'RSI': report_data['rsi']['drawdown'],
            'MACD': report_data['macd']['drawdown'],
            'MA Cross': report_data['ma']['drawdown'],
            'Converging': report_data['conv']['drawdown']
        },
        output_path
    )
    
    story.append(Image(drawdown_plot_path, width=500, height=200))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Figure 2: Drawdown analysis of all strategies over the backtest period.",
        styles['BodyText']
    ))
    
    # Add conclusions
    story.append(PageBreak())
    story.append(Paragraph("Conclusions and Recommendations", styles['Section']))
    story.append(Spacer(1, 10))
    
    # Find best strategy
    strategies = ['rsi', 'macd', 'ma', 'conv']
    best_strat = max(strategies, key=lambda x: report_data[x]['sharpe'])
    best_strat_name = {
        'rsi': 'RSI',
        'macd': 'MACD',
        'ma': 'Moving Average Crossover',
        'conv': 'Converging Signals'
    }[best_strat]
    
    conclusions = [
        "Based on the backtest results:",
        f"• The best performing strategy was {best_strat_name} with a Sharpe ratio of {report_data[best_strat]['sharpe']:.2f}.",
        f"• The highest return was achieved by {best_strat_name} with a total return of {report_data[best_strat]['total_return']*100:.2f}%.",
        f"• The most stable strategy was {best_strat_name} with a maximum drawdown of {report_data[best_strat]['max_dd']*100:.2f}%.",
        "\nRecommendations:",
        "1. Consider implementing the best performing strategy in a paper trading environment.",
        "2. Conduct further optimization of strategy parameters.",
        "3. Perform walk-forward analysis to validate strategy robustness.",
        "4. Consider implementing risk management rules to limit drawdowns."
    ]
    
    for line in conclusions:
        if line.startswith("•"):
            story.append(Paragraph(line, styles['BodyText']))
            story.append(Spacer(1, 5))
        elif line.endswith(":"):
            story.append(Paragraph(f"<b>{line}</b>", styles['BodyText']))
            story.append(Spacer(1, 5))
        elif line.startswith("Recommendations:"):
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>{line}</b>", styles['BodyText']))
            story.append(Spacer(1, 5))
        elif line[0].isdigit():
            story.append(Paragraph(line, styles['BodyText']))
            story.append(Spacer(1, 5))
        else:
            story.append(Paragraph(line, styles['BodyText']))
    
    # Add footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This report was generated automatically. Past performance is not indicative of future results.",
        ParagraphStyle('Footer', parent=styles['Italic'], fontSize=8, textColor=colors.grey)
    ))
    
    # Build the PDF
    doc.build(story)
    
    # Clean up temporary plot files
    for file in [equity_plot_path, drawdown_plot_path]:
        try:
            os.remove(file)
        except:
            pass
    
    return os.path.join(output_path, 'trading_performance_report.pdf')

# Example usage
if __name__ == "__main__":
    # This is a sample structure of the data you would pass to the function
    # In a real scenario, you would extract this from your backtest results
    sample_data = {
        'rsi': {
            'total_return': 1.7398,
            'sharpe': 1.02,
            'max_dd': -0.2176,
            'win_rate': 0.5289,
            'final_value': 273983.83,
            'equity_curve': pd.Series(np.random.normal(0.001, 0.02, 252).cumsum() + 1, 
                                    index=pd.date_range('2020-01-01', periods=252)),
            'drawdown': pd.Series(np.random.uniform(-0.05, 0, 252), 
                                index=pd.date_range('2020-01-01', periods=252)).cumsum()
        },
        'macd': {
            'total_return': 1.7398,
            'sharpe': 1.02,
            'max_dd': -0.2176,
            'win_rate': 0.5289,
            'final_value': 273983.83,
            'equity_curve': pd.Series(np.random.normal(0.001, 0.02, 252).cumsum() + 1, 
                                    index=pd.date_range('2020-01-01', periods=252)),
            'drawdown': pd.Series(np.random.uniform(-0.05, 0, 252), 
                                index=pd.date_range('2020-01-01', periods=252)).cumsum()
        },
        'ma': {
            'total_return': 1.7398,
            'sharpe': 1.02,
            'max_dd': -0.2176,
            'win_rate': 0.5289,
            'final_value': 273983.83,
            'equity_curve': pd.Series(np.random.normal(0.001, 0.02, 252).cumsum() + 1, 
                                    index=pd.date_range('2020-01-01', periods=252)),
            'drawdown': pd.Series(np.random.uniform(-0.05, 0, 252), 
                                index=pd.date_range('2020-01-01', periods=252)).cumsum()
        },
        'conv': {
            'total_return': 0.0960,
            'sharpe': -0.0183,
            'max_dd': -0.0263,
            'win_rate': 0.7200,
            'final_value': 109604.58,
            'equity_curve': pd.Series(np.random.normal(0.001, 0.02, 252).cumsum() + 1, 
                                    index=pd.date_range('2020-01-01', periods=252)),
            'drawdown': pd.Series(np.random.uniform(-0.05, 0, 252), 
                                index=pd.date_range('2020-01-01', periods=252)).cumsum()
        }
    }
    
    # Generate the report
    report_path = generate_pdf_report('reports', sample_data)
    print(f"Report generated at: {report_path}")
