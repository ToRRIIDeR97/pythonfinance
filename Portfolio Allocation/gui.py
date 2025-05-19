import sys
import io
import contextlib
import argparse # Need to construct args object for backtester
import pandas as pd # For date conversion
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, 
    QFormLayout, QDateEdit, QSpinBox, QDoubleSpinBox, QCheckBox, 
    QComboBox, QTabWidget, QSplitter
)
from PySide6.QtCore import QDate, Qt, QThread, Signal

# Import backtester logic
import backtester
import numpy as np # For averaging results
from pandas.tseries.offsets import DateOffset # For rolling dates
from joblib import Parallel, delayed # For parallel rolling execution

# --- Worker Thread --- 
class Worker(QThread):
    progress = Signal(str)  # To send log messages
    results = Signal(list, dict) # Send list of results and average dict
    finished = Signal()     # To signal completion

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self._is_running = True

    def run(self):
        """Execute the backtest logic in a separate thread."""
        log_stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(log_stream):
                # Reconstruct args object for backtester functions
                args = argparse.Namespace(
                    run_baseline=self.settings['run_baseline'],
                    run_initial_taa=self.settings['run_initial_taa'],
                    run_optimization=self.settings['run_optimization'],
                    opt_objective=self.settings['opt_objective'],
                    run_rolling=self.settings['run_rolling'],
                    rolling_window_years=self.settings['rolling_window_years'],
                    rolling_step_months=self.settings['rolling_step_months'],
                    overall_start_date=self.settings['overall_start_date'],
                    overall_end_date=self.settings['overall_end_date']
                )
                
                # Get other configs that were global in main.py
                base_alloc = self.settings['base_alloc']
                initial_adj = self.settings['initial_adj']
                ticker_map_config = self.settings['ticker_map']
                assets_order_config = sorted(base_alloc.keys()) # Derive from base_alloc
                regimes_order_config = ["Bull", "Bear", "Stagnation"] # Hardcode or make configurable?
                cost_bps = self.settings['cost_bps']
                
                # Extract execution and regime params from settings
                initial_capital_param = self.settings['initial_capital']
                rebalance_freq_param = self.settings['rebalance_frequency']
                indicator_tf_param = self.settings['indicator_timeframe']
                regime_proxy_param = self.settings['regime_proxy']
                ema_f_param = self.settings['emaFastLen']
                ema_s_param = self.settings['emaSlowLen']
                atr_l_param = self.settings['emaMarginATRLen']
                atr_m_param = self.settings['emaMarginATRMult']

                all_periods_results = []
                average_results_summary = {}

                if args.run_rolling:
                    print("--- Generating Rolling Window Periods --- ")
                    overall_start = pd.to_datetime(args.overall_start_date)
                    overall_end = pd.to_datetime(args.overall_end_date)
                    window_years = args.rolling_window_years
                    step_months = args.rolling_step_months
                    
                    periods = [] 
                    current_start = overall_start
                    while self._is_running: # Check if cancelled
                        current_end = current_start + DateOffset(years=window_years)
                        if current_end > overall_end:
                            current_end = overall_end
                            if current_start < current_end - DateOffset(months=1):
                                periods.append((current_start, current_end))
                            break 
                        periods.append((current_start, current_end))
                        next_start = current_start + DateOffset(months=step_months)
                        if next_start >= overall_end: 
                             break 
                        current_start = next_start
                    
                    if not self._is_running: raise InterruptedError("Backtest cancelled")
                    
                    print(f"Generated {len(periods)} rolling periods.")
                    print("--- Starting Parallel Rolling Window Backtests --- ")
                    self.progress.emit(log_stream.getvalue()) # Send initial logs
                    log_stream = io.StringIO() # Reset stream

                    # Run backtests in parallel
                    # NOTE: Capturing stdout from parallel jobs directly is tricky.
                    # Print statements within backtester.run_backtest_for_period will appear 
                    # in the console where app.py is run, not necessarily in the GUI log smoothly.
                    # For better GUI logging, would need to modify run_backtest_for_period 
                    # to accept a callback/signal emitter.
                    all_periods_results = Parallel(n_jobs=-1)(delayed(backtester.run_backtest_for_period)(
                        p_start, p_end, args, 
                        ticker_map_config, base_alloc, initial_adj, assets_order_config, regimes_order_config, 
                        cost_bps,
                        # Pass additional params
                        initial_capital=initial_capital_param,
                        rebalance_frequency=rebalance_freq_param,
                        indicator_timeframe=indicator_tf_param,
                        regime_proxy_ticker=regime_proxy_param,
                        emaFastLen=ema_f_param,
                        emaSlowLen=ema_s_param,
                        emaMarginATRLen=atr_l_param,
                        emaMarginATRMult=atr_m_param
                        ) for p_start, p_end in periods)

                    # Aggregate results
                    valid_results = [res for res in all_periods_results if res is not None]
                    num_successful_periods = len(valid_results)
                    print(f"\n{'='*20} ROLLING BACKTEST SUMMARY ({num_successful_periods} successful periods) {'='*20}")
                    
                    if num_successful_periods > 0:
                        avg_metrics = {}
                        metric_keys = [k for k in valid_results[0].keys() if isinstance(valid_results[0][k], (int, float))]
                        for key in metric_keys:
                            metric_values = [res.get(key, np.nan) for res in valid_results]
                            try: 
                                avg_metrics[key] = np.nanmean(metric_values)
                            except TypeError: 
                                print(f"Warning: Could not calculate average for metric '{key}'")
                                avg_metrics[key] = None 
                        average_results_summary = avg_metrics # Store the calculated averages
                        # Print average results to log as well
                        print("Average Performance Metrics Across Rolling Windows:")
                        if avg_metrics.get('CAGR') is not None: print(f"  Average CAGR: {avg_metrics['CAGR']:.2%}")
                        # ... print other averages ... 
                        if avg_metrics.get('Sharpe') is not None: print(f"  Average Sharpe Ratio (Rf=0%): {avg_metrics['Sharpe']:.2f}")

                    else:
                        print("No successful periods completed to calculate average results.")
                    
                else: # Single Period Execution
                    print("--- Starting Single Period Backtest --- ")
                    single_start = pd.to_datetime(args.overall_start_date)
                    single_end = pd.to_datetime(args.overall_end_date)
                    self.progress.emit(log_stream.getvalue()) # Send initial logs
                    log_stream = io.StringIO() # Reset stream
                    
                    # Run single period with all params
                    result = backtester.run_backtest_for_period(single_start, single_end, args, 
                                              ticker_map_config, base_alloc, initial_adj, assets_order_config, regimes_order_config,
                                              cost_bps,
                                              # Pass additional params
                                              initial_capital=initial_capital_param,
                                              rebalance_frequency=rebalance_freq_param,
                                              indicator_timeframe=indicator_tf_param,
                                              regime_proxy_ticker=regime_proxy_param,
                                              emaFastLen=ema_f_param,
                                              emaSlowLen=ema_s_param,
                                              emaMarginATRLen=atr_l_param,
                                              emaMarginATRMult=atr_m_param
                                              )
                    all_periods_results = [result] # Store single result in the list
                    
                    # Note: The Sharpe update prompt inside run_backtest_for_period will be 
                    # redirected here. We might need a different mechanism for GUI interaction.
                
                print("\nBacktest processing finished.")

        except InterruptedError as e:
             print(f"\nExecution cancelled: {e}")
        except Exception as e:
            print(f"\nError during backtest execution: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.progress.emit(log_stream.getvalue()) # Send any remaining logs
            self.results.emit(all_periods_results, average_results_summary) # Send results list and average dict
            self.finished.emit()

    def stop(self):
        self._is_running = False
        # Note: Stopping parallel jobs forcefully is complex and not implemented here.
        # This mainly helps stop the period generation loop.
        print("Worker stop requested.")

# --- Main Window --- 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portfolio Backtester")
        self.setGeometry(100, 100, 1200, 800) # x, y, width, height
        self.worker = None # Placeholder for the thread

        # --- Main Layout --- 
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Use a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Panel: Settings & Controls --- 
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        splitter.addWidget(left_panel)

        self.create_settings_panel(left_layout)
        self.create_controls_panel(left_layout)
        left_layout.addStretch() # Push controls to the top

        # --- Right Panel: Output --- 
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)

        self.create_output_panel(right_layout)

        # Set initial sizes for the splitter panels
        splitter.setSizes([400, 800]) # Adjust as needed

    def create_settings_panel(self, parent_layout):
        settings_group = QGroupBox("Settings")
        parent_layout.addWidget(settings_group)
        settings_layout = QVBoxLayout(settings_group)

        # --- Tab Widget for Different Settings Categories ---
        tab_widget = QTabWidget()
        settings_layout.addWidget(tab_widget)

        # General Tab
        general_tab = QWidget()
        tab_widget.addTab(general_tab, "General")
        general_layout = QFormLayout(general_tab)

        self.start_date_edit = QDateEdit(QDate(2015, 1, 1)) # Default start
        self.end_date_edit = QDateEdit(QDate.currentDate())
        self.initial_capital_spin = QSpinBox()
        self.initial_capital_spin.setRange(1000, 10000000)
        self.initial_capital_spin.setSingleStep(1000)
        self.initial_capital_spin.setValue(100000)
        self.rebalance_combo = QComboBox()
        self.rebalance_combo.addItems(['monthly', 'quarterly', 'yearly', 'daily'])
        self.trading_cost_spin = QSpinBox()
        self.trading_cost_spin.setSuffix(" bps")
        self.trading_cost_spin.setValue(5)

        general_layout.addRow("Overall Start Date:", self.start_date_edit)
        general_layout.addRow("Overall End Date:", self.end_date_edit)
        general_layout.addRow("Initial Capital ($):", self.initial_capital_spin)
        general_layout.addRow("Rebalance Frequency:", self.rebalance_combo)
        general_layout.addRow("Trading Cost (bps):", self.trading_cost_spin)

        # Regime Tab
        regime_tab = QWidget()
        tab_widget.addTab(regime_tab, "Regime")
        regime_layout = QFormLayout(regime_tab)
        # Add placeholders for regime settings
        self.regime_proxy_edit = QLineEdit("VOO")
        self.ema_fast_spin = QSpinBox()
        self.ema_fast_spin.setRange(5, 200)
        self.ema_fast_spin.setValue(30)
        self.ema_slow_spin = QSpinBox()
        self.ema_slow_spin.setRange(10, 400)
        self.ema_slow_spin.setValue(60)
        self.atr_len_spin = QSpinBox()
        self.atr_len_spin.setRange(5, 200)
        self.atr_len_spin.setValue(60)
        self.atr_mult_spin = QDoubleSpinBox()
        self.atr_mult_spin.setRange(0.1, 5.0)
        self.atr_mult_spin.setSingleStep(0.05)
        self.atr_mult_spin.setValue(0.30)
        self.atr_mult_spin.setDecimals(2)

        regime_layout.addRow("Proxy Ticker:", self.regime_proxy_edit)
        regime_layout.addRow("EMA Fast Length:", self.ema_fast_spin)
        regime_layout.addRow("EMA Slow Length:", self.ema_slow_spin)
        regime_layout.addRow("ATR Length:", self.atr_len_spin)
        regime_layout.addRow("ATR Multiplier:", self.atr_mult_spin)
        # TODO: Add Indicator Timeframe (daily, weekly, monthly)

        # Allocation Tab
        allocation_tab = QWidget()
        tab_widget.addTab(allocation_tab, "Allocation")
        allocation_layout = QVBoxLayout(allocation_tab)
        allocation_layout.addWidget(QLabel("Baseline SAA (JSON format):"))
        self.baseline_alloc_edit = QTextEdit("{\n  \"DEF\": 0.25,\n  \"WLD\": 0.30,\n  \"AGG\": 0.35,\n  \"CRY\": 0.10,\n  \"CASH\": 0.00\n}")
        allocation_layout.addWidget(self.baseline_alloc_edit)
        allocation_layout.addWidget(QLabel("Initial TAA (JSON format):"))
        self.initial_taa_edit = QTextEdit("{\n  \"Bull\": {\"AGG\": 0.18, \"CASH\": 0.5, \"CRY\": -0.11, \"DEF\": -0.5, \"WLD\": -0.5},\n  \"Bear\": {\"AGG\": -0.44, \"CASH\": -0.38, \"CRY\": -0.5, \"DEF\": 0.5, \"WLD\": -0.22},\n  \"Stagnation\": {\"AGG\": -0.46, \"CASH\": 0.49, \"CRY\": -0.1, \"DEF\": -0.49, \"WLD\": -0.43}\n}")
        allocation_layout.addWidget(self.initial_taa_edit)

        # Backtest Type Tab
        backtest_tab = QWidget()
        tab_widget.addTab(backtest_tab, "Backtest Type")
        backtest_layout = QVBoxLayout(backtest_tab)
        self.run_baseline_check = QCheckBox("Run Baseline SAA")
        self.run_initial_taa_check = QCheckBox("Run Initial TAA")
        self.run_optimization_check = QCheckBox("Run Optimization")
        self.opt_objective_combo = QComboBox()
        self.opt_objective_combo.addItems(['sharpe', 'cagr', 'mdd', 'all'])
        self.run_optimization_check.toggled.connect(self.opt_objective_combo.setEnabled)
        self.opt_objective_combo.setEnabled(False)

        backtest_layout.addWidget(self.run_baseline_check)
        backtest_layout.addWidget(self.run_initial_taa_check)
        backtest_layout.addWidget(self.run_optimization_check)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Optimization Objective:"))
        hbox.addWidget(self.opt_objective_combo)
        backtest_layout.addLayout(hbox)
        backtest_layout.addStretch()

        # Rolling Tab
        rolling_tab = QWidget()
        tab_widget.addTab(rolling_tab, "Rolling Window")
        rolling_layout = QFormLayout(rolling_tab)
        self.run_rolling_check = QCheckBox("Enable Rolling Backtest")
        self.rolling_window_spin = QSpinBox()
        self.rolling_window_spin.setRange(1, 20)
        self.rolling_window_spin.setValue(5)
        self.rolling_step_spin = QSpinBox()
        self.rolling_step_spin.setRange(1, 12)
        self.rolling_step_spin.setValue(1)
        self.rolling_window_spin.setEnabled(False)
        self.rolling_step_spin.setEnabled(False)
        self.run_rolling_check.toggled.connect(self.rolling_window_spin.setEnabled)
        self.run_rolling_check.toggled.connect(self.rolling_step_spin.setEnabled)

        rolling_layout.addRow(self.run_rolling_check)
        rolling_layout.addRow("Window (Years):", self.rolling_window_spin)
        rolling_layout.addRow("Step (Months):", self.rolling_step_spin)

    def create_controls_panel(self, parent_layout):
        controls_group = QGroupBox("Controls")
        parent_layout.addWidget(controls_group)
        controls_layout = QHBoxLayout(controls_group)

        self.start_button = QPushButton("Start Backtest")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False) # Initially disabled

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.cancel_button)

        # Connect signals/slots
        self.start_button.clicked.connect(self.start_backtest)
        self.cancel_button.clicked.connect(self.cancel_backtest)

    def create_output_panel(self, parent_layout):
        output_tabs = QTabWidget()
        parent_layout.addWidget(output_tabs)

        # Log Tab
        log_tab = QWidget()
        output_tabs.addTab(log_tab, "Log")
        log_layout = QVBoxLayout(log_tab)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFontFamily("Courier") # Use monospace font
        log_layout.addWidget(self.log_output)

        # Results Tab
        results_tab = QWidget()
        output_tabs.addTab(results_tab, "Results Summary")
        results_layout = QVBoxLayout(results_tab)
        self.results_output = QTextEdit("Performance metrics will appear here.") # Placeholder, maybe use QTableWidget
        self.results_output.setReadOnly(True)
        results_layout.addWidget(self.results_output)

        # Plot Tab
        plot_tab = QWidget()
        output_tabs.addTab(plot_tab, "Plots")
        plot_layout = QVBoxLayout(plot_tab)
        plot_layout.addWidget(QLabel("Plot placeholder - requires QWebEngineView"))
        # TODO: Add QWebEngineView for Plotly plots

    # --- Backtest Execution and Signal Handling --- 
    def start_backtest(self):
        self.log_output.clear() # Clear previous logs
        self.results_output.setText("Running backtest...")
        self.log_output.append("Starting backtest...")

        # Gather settings from GUI
        try:
            # Basic settings
            settings = {
                'overall_start_date': self.start_date_edit.date().toString("yyyy-MM-dd"),
                'overall_end_date': self.end_date_edit.date().toString("yyyy-MM-dd"),
                'initial_capital': self.initial_capital_spin.value(),
                'rebalance_frequency': self.rebalance_combo.currentText(),
                'cost_bps': self.trading_cost_spin.value(),
                # Regime settings
                'regime_proxy': self.regime_proxy_edit.text(),
                'indicator_timeframe': 'daily', # TODO: Add widget for this
                'emaFastLen': self.ema_fast_spin.value(),
                'emaSlowLen': self.ema_slow_spin.value(),
                'emaMarginATRLen': self.atr_len_spin.value(),
                'emaMarginATRMult': self.atr_mult_spin.value(),
                # Backtest types
                'run_baseline': self.run_baseline_check.isChecked(),
                'run_initial_taa': self.run_initial_taa_check.isChecked(),
                'run_optimization': self.run_optimization_check.isChecked(),
                'opt_objective': self.opt_objective_combo.currentText() if self.run_optimization_check.isChecked() else None,
                # Rolling settings
                'run_rolling': self.run_rolling_check.isChecked(),
                'rolling_window_years': self.rolling_window_spin.value(),
                'rolling_step_months': self.rolling_step_spin.value(),
                # Allocations (load from text edits - requires json parsing)
                'base_alloc': eval(self.baseline_alloc_edit.toPlainText()), # Use eval carefully, json.loads preferred
                'initial_adj': eval(self.initial_taa_edit.toPlainText()), # Use eval carefully, json.loads preferred
                # Global config from main.py (assuming these are relatively static)
                # TODO: Make ticker_map configurable in GUI?
                'ticker_map': backtester.ticker_map if hasattr(backtester, 'ticker_map') else {
                    "DEF": "SCHD", "WLD": "VT", "AGG": "QQQ", "CRY": "ETH-USD", "CASH": "CASH" 
                }
            }
            # Basic validation
            if not (settings['run_baseline'] or settings['run_initial_taa'] or settings['run_optimization']):
                 self.log_output.append("Error: Please select at least one backtest type to run.")
                 self.results_output.setText("Error: No backtest type selected.")
                 return
                 
        except Exception as e:
            self.log_output.append(f"Error reading settings: {e}")
            self.results_output.setText(f"Error reading settings: {e}")
            return

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        # Create and start the worker thread
        self.worker = Worker(settings)
        self.worker.progress.connect(self.update_log)
        self.worker.results.connect(self.show_results)
        self.worker.finished.connect(self.on_backtest_finished)
        self.worker.start()

    def cancel_backtest(self):
        if self.worker and self.worker.isRunning():
            self.log_output.append("\nAttempting to cancel backtest...")
            self.worker.stop()
            # Note: Actual stopping might take time or may not be fully effective
            # depending on where the backtester process is.
            self.cancel_button.setEnabled(False) # Disable cancel once requested

    def on_backtest_finished(self):
        self.log_output.append("\nBacktest thread finished.")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None # Clear worker reference

    def update_log(self, message):
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum()) # Auto-scroll

    def show_results(self, all_results, avg_summary):
        self.results_output.clear()
        if not all_results:
            self.results_output.setText("No results generated.")
            return

        summary_text = ""    
        # Display individual period results if not too many
        if self.worker.settings['run_rolling'] and len(all_results) > 1:
             summary_text += f"ROLLING BACKTEST SUMMARY ({len(all_results)} periods)\n"
             summary_text += "="*40 + "\n"
             # Display average results first
             if avg_summary:
                 summary_text += "Average Performance Metrics:\n"
                 if avg_summary.get('CAGR') is not None: summary_text += f"  Avg CAGR: {avg_summary['CAGR']:.2%}\n"
                 if avg_summary.get('Volatility') is not None: summary_text += f"  Avg Volatility: {avg_summary['Volatility']:.2%}\n"
                 if avg_summary.get('Sharpe') is not None: summary_text += f"  Avg Sharpe: {avg_summary['Sharpe']:.2f}\n"
                 if avg_summary.get('Sortino') is not None: summary_text += f"  Avg Sortino: {avg_summary['Sortino']:.2f}\n"
                 if avg_summary.get('MaxDrawdown') is not None: summary_text += f"  Avg Max Drawdown: {avg_summary['MaxDrawdown']:.2%}\n"
                 summary_text += "-"*40 + "\n"
             else:
                 summary_text += "(Could not calculate average metrics)\n"
                 summary_text += "-"*40 + "\n"
                 
        # Display results for the first period (or single period)
        first_result = all_results[0]
        if first_result:
             if not self.worker.settings['run_rolling']:
                 summary_text += f"SINGLE PERIOD RESULTS ({self.worker.settings['overall_start_date']} - {self.worker.settings['overall_end_date']})\n"
             else:
                 summary_text += f"RESULTS FOR FIRST PERIOD\n" # Indicate it's just the first if rolling
             summary_text += "-"*40 + "\n"
             summary_text += f"  Final Value: ${first_result.get('FinalValue', 0):,.2f}\n"
             summary_text += f"  CAGR: {first_result.get('CAGR', 0):.2%}\n"
             summary_text += f"  Volatility: {first_result.get('Volatility', 0):.2%}\n"
             summary_text += f"  Sharpe Ratio: {first_result.get('Sharpe', float('nan')):.2f}\n"
             summary_text += f"  Sortino Ratio: {first_result.get('Sortino', float('nan')):.2f}\n"
             summary_text += f"  Max Drawdown: {first_result.get('MaxDrawdown', 0):.2%}\n"
             summary_text += f"  Total Costs: ${first_result.get('TotalTradingCost', 0):,.0f}\n"
        else:
            summary_text += "No valid results for the first period.\n"

        self.results_output.setText(summary_text)
        # TODO: Potentially display full results in a table (QTableWidget)
        # TODO: Trigger plot generation/display

# Example of running the window directly for testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 