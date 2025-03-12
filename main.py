import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt  # if needed

#####################################
# Helper Functions
#####################################

def get_metric_data(xls, sheet_name, metrics, skiprows=4, metric_col=1, data_col_start=2, data_col_end=7):
    df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=skiprows, header=None)
    selected = {}
    for metric in metrics:
        for idx, row in df.iterrows():
            if str(row[metric_col]).strip().lower() == metric.lower():
                data = row.iloc[data_col_start:data_col_end]
                data_numeric = pd.to_numeric(data, errors='coerce')
                selected[metric] = data_numeric.values
                break
    num_years = data_col_end - data_col_start
    result = pd.DataFrame.from_dict(
        selected,
        orient='index',
        columns=[f"Year{i}" for i in range(1, num_years+1)]
    )
    return result

def calculate_cagr(values):
    arr = np.array(values)
    if len(arr) < 2 or arr[0] <= 0 or arr[-1] <= 0:
        return None
    return (arr[-1] / arr[0]) ** (1 / (len(arr) - 1)) - 1

def plot_metrics_plotly(df_section, title):
    plot_df = df_section.drop(columns="CAGR") if "CAGR" in df_section.columns else df_section
    fig = go.Figure()
    for metric in plot_df.index:
        y_values = pd.to_numeric(plot_df.loc[metric], errors='coerce').values
        yoy = [None] * len(y_values)
        for i in range(1, len(y_values)):
            prev = y_values[i-1]
            curr = y_values[i]
            if pd.notnull(prev) and prev != 0:
                yoy[i] = (curr - prev) / prev
        hover_tpl = '<b>' + metric + '</b><br>Year: %{x}<br>Value: %{y:.2f}<br>YoY: %{customdata:.2%}<extra></extra>'
        fig.add_trace(go.Scatter(
            x=plot_df.columns,
            y=y_values,
            mode='lines+markers',
            name=metric,
            customdata=yoy,
            hovertemplate=hover_tpl
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Amount",
        legend_title="Metrics",
        hovermode="x unified"
    )
    return fig

#####################################
# New Helper: Compute Free Cash Flow
#####################################
def get_free_cash_flow(xls, sheet_cf, skiprows=4, metric_col=1, data_col_start=2, data_col_end=7):
    """
    Attempts to retrieve a row named 'Free Cash Flow' from the Cash Flow sheet.
    If not found, computes it as:
         FCF = Cash Provided by Operations – Capital Spending
    Returns a DataFrame with index 'Free Cash Flow'.
    """
    df_fcf = get_metric_data(xls, sheet_cf, ["Free Cash Flow"], skiprows=skiprows, 
                             metric_col=metric_col, data_col_start=data_col_start, data_col_end=data_col_end)
    if not df_fcf.empty:
        return df_fcf
    else:
        df_cfo = get_metric_data(xls, sheet_cf, ["Cash Provided by Operations"], skiprows=skiprows, 
                                 metric_col=metric_col, data_col_start=data_col_start, data_col_end=data_col_end)
        df_capex = get_metric_data(xls, sheet_cf, ["Capital Spending"], skiprows=skiprows, 
                                   metric_col=metric_col, data_col_start=data_col_start, data_col_end=data_col_end)
        if df_cfo.empty or df_capex.empty:
            return pd.DataFrame()
        cfo_values = pd.to_numeric(df_cfo.iloc[0], errors='coerce')
        capex_values = pd.to_numeric(df_capex.iloc[0], errors='coerce')
        fcf_values = cfo_values - capex_values
        fcf_df = pd.DataFrame([fcf_values], index=["Free Cash Flow"], columns=df_cfo.columns)
        return fcf_df

#####################################
# Global Setup: Historical Years (Descending Order)
#####################################
hist_years = ['2024', '2023', '2022', '2021', '2020']

#####################################
# Section Functions – Using Pre-selected Sheet Names & Unique Keys
#####################################

def display_income_statement(xls, stock_label, sheet_income, key_suffix=""):
    st.subheader(f"{stock_label} – Income Statement")
    all_metrics = ["Net Sales", "Gross Profit", "Cost of Products Sold", 
                   "Marketing, Research & General Expenses", "Operating Profit", 
                   "Interest Income", "Interest Expense", "Net Income"]
    selected_metrics = st.multiselect(
        f"Select Income Statement metrics for {stock_label}", 
        all_metrics, default=all_metrics, key=f"{stock_label}_inc_metrics{key_suffix}"
    )
    df = get_metric_data(xls, sheet_income, selected_metrics)
    if not df.empty:
        df.columns = hist_years
        cagr_dict = {m: calculate_cagr(pd.to_numeric(df.loc[m].dropna(), errors='coerce')) for m in df.index}
        df["CAGR"] = pd.Series(cagr_dict)
        st.dataframe(df, use_container_width=True)
        fig = plot_metrics_plotly(df, f"{stock_label} Income Statement Metrics")
        st.plotly_chart(fig, use_container_width=True, key=f"{stock_label}_inc_plot{key_suffix}")
    else:
        st.warning(f"No Income Statement data found for {stock_label}.")

def display_balance_sheet(xls, stock_label, sheet_balance, key_suffix=""):
    st.subheader(f"{stock_label} – Balance Sheet")
    all_metrics = ["Inventories", "Cash and Cash Equivalents", "Accounts Receivable, Net", 
                   "Total Current Assets", "Total Long Term Assets", "TOTAL ASSETS", 
                   "Total Current Liabilities", "Total Long Term Liabilities", "Total Liabilities", 
                   "Total Stockholders' Equity", "Outstanding Shares"]
    selected_metrics = st.multiselect(
        f"Select Balance Sheet metrics for {stock_label}", 
        all_metrics, default=all_metrics, key=f"{stock_label}_balance_metrics{key_suffix}"
    )
    df = get_metric_data(xls, sheet_balance, selected_metrics)
    if not df.empty:
        df.columns = hist_years
        cagr_dict = {m: calculate_cagr(pd.to_numeric(df.loc[m].dropna(), errors='coerce')) for m in df.index}
        df["CAGR"] = pd.Series(cagr_dict)
        st.dataframe(df, use_container_width=True)
        fig = plot_metrics_plotly(df, f"{stock_label} Balance Sheet Metrics")
        st.plotly_chart(fig, use_container_width=True, key=f"{stock_label}_bs_plot{key_suffix}")
    else:
        st.warning(f"No Balance Sheet data found for {stock_label}.")

def display_cash_flow(xls, stock_label, sheet_cf, key_suffix=""):
    st.subheader(f"{stock_label} – Cash Flow Analysis")
    # Allow selection of metrics; note that when "Free Cash Flow" is selected, we compute it.
    all_metrics = ["Cash Provided by Operations", "Capital Spending", "Free Cash Flow", "Depreciation and Amortization"]
    selected_metrics = st.multiselect(
        f"Select Cash Flow metrics for {stock_label}", 
        all_metrics, default=all_metrics, key=f"{stock_label}_cf_metrics{key_suffix}"
    )
    # Get data for selected metrics excluding "Free Cash Flow"
    df = get_metric_data(xls, sheet_cf, [m for m in selected_metrics if m != "Free Cash Flow"])
    
    # If Free Cash Flow is selected, compute it as CFO minus Capex
    if "Free Cash Flow" in selected_metrics:
        df_cfo = get_metric_data(xls, sheet_cf, ["Cash Provided by Operations"], skiprows=4, metric_col=1, data_col_start=2, data_col_end=7)
        df_capex = get_metric_data(xls, sheet_cf, ["Capital Spending"], skiprows=4, metric_col=1, data_col_start=2, data_col_end=7)
        if not df_cfo.empty and not df_capex.empty:
            cfo_values = pd.to_numeric(df_cfo.iloc[0], errors='coerce')
            capex_values = pd.to_numeric(df_capex.iloc[0], errors='coerce')
            fcf_values = cfo_values - capex_values
            df_fcf = pd.DataFrame([fcf_values], index=["Free Cash Flow"], columns=df_cfo.columns)
        else:
            df_fcf = pd.DataFrame()
        df = pd.concat([df, df_fcf])
    
    if not df.empty:
        df.columns = hist_years
        cagr_dict = {m: calculate_cagr(pd.to_numeric(df.loc[m].dropna(), errors='coerce')) for m in df.index}
        df["CAGR"] = pd.Series(cagr_dict)
        st.dataframe(df, use_container_width=True)
        fig = plot_metrics_plotly(df, f"{stock_label} Cash Flow Metrics")
        st.plotly_chart(fig, use_container_width=True, key=f"{stock_label}_cf_plot{key_suffix}")
    else:
        st.warning(f"No Cash Flow data found for {stock_label}.")

def display_operating_metrics(xls, stock_label, sheet_op, key_suffix=""):
    st.subheader(f"{stock_label} – Operating Metrics")
    all_metrics = [
        "Gross Margin", "SGA Margin", "Operating Margin", "EBITDA Margin", "Net Margin",
        "Free Cash Flow Margin", "Operating Cash Flow Margin", "Price to Book", "Price to Tangible Book",
        "Price to Sales", "Price to Gross Profit", "Price to Operating Cash Flow", "Price to Operating Income",
        "Price to Earnings", "Price to Free Cash Flow", "Return On Assets", "Return On Capital Employed",
        "Return On Equity", "Retained Earning Per Share", "Book Value Per Share", "Free Cash Flow Per Share",
        "Operating Cash Flow Per Share", "Revenue Per Share", "Current Ratio", "Debt to Equity Ratio", "Shares Outstanding", 
        "Operating Cash Flow Ratio", "Quick Ratio", "Debts to Assets Ratio", "Interest Coverage Ratio", "Equtiy Ratio"
    ]
    selected_metrics = st.multiselect(
        f"Select Operating Metrics for {stock_label}", 
        all_metrics, default=all_metrics, key=f"{stock_label}_op_metrics{key_suffix}"
    )
    df = get_metric_data(xls, sheet_op, selected_metrics)
    if not df.empty:
        df.columns = hist_years
        cagr_dict = {m: calculate_cagr(pd.to_numeric(df.loc[m].dropna(), errors='coerce')) for m in df.index}
        df["CAGR"] = pd.Series(cagr_dict)
        st.dataframe(df, use_container_width=True)
        fig = plot_metrics_plotly(df, f"{stock_label} Operating Metrics")
        st.plotly_chart(fig, use_container_width=True, key=f"{stock_label}_op_plot{key_suffix}")
    else:
        st.info(f"No Operating Metrics data found for {stock_label}.")

def display_fixed_variable_costs(xls, stock_label, sheet_inc, sheet_cf, key_suffix=""):
    st.subheader(f"{stock_label} – Fixed vs Variable Costs")
    df_fixed_cf = get_metric_data(xls, sheet_cf, ["Depreciation and Amortization"])
    df_fixed_inc = get_metric_data(xls, sheet_inc, ["General and Administrative Expenses"])
    fixed_list = [df for df in [df_fixed_cf, df_fixed_inc] if not df.empty]
    if fixed_list:
        df_fixed = pd.concat(fixed_list)
    else:
        df_fixed = pd.DataFrame()
    df_variable = get_metric_data(xls, sheet_inc, ["Cost of Products Sold", "Marketing, Research, General Expenses"])
    if not df_fixed.empty or not df_variable.empty:
        df_costs = pd.concat([df_fixed, df_variable])
        df_costs.columns = hist_years
        df_net_sales = get_metric_data(xls, sheet_inc, ["Net Sales"])
        if df_net_sales.empty:
            st.warning("No Net Sales data found; cannot compute percentages.")
        else:
            df_net_sales.columns = hist_years
            net_sales = pd.to_numeric(df_net_sales.iloc[0], errors='coerce')
            def format_cost_cell(x, col):
                pct = (x / net_sales[col]) if net_sales[col] != 0 else np.nan
                return f"{x:.2f}\n({pct:.2%})"
            df_costs_formatted = df_costs.copy()
            for col in df_costs.columns:
                df_costs_formatted[col] = df_costs[col].apply(lambda x: format_cost_cell(x, col))
            st.dataframe(df_costs_formatted, use_container_width=True)
            fig_line = go.Figure()
            for metric in df_costs.index:
                y_values = pd.to_numeric(df_costs.loc[metric], errors='coerce').values
                fig_line.add_trace(go.Scatter(
                    x=df_costs.columns,
                    y=y_values,
                    mode='lines+markers',
                    name=metric,
                    hovertemplate='<b>' + metric + '</b><br>Year: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ))
            fig_line.update_layout(title="Fixed & Variable Costs (Raw Values)", xaxis_title="Year", yaxis_title="Cost")
            st.plotly_chart(fig_line, use_container_width=True, key=f"{stock_label}_fv_line_plot{key_suffix}")
            fig_bar = go.Figure()
            for metric in df_costs.index:
                y_pct = (pd.to_numeric(df_costs.loc[metric], errors='coerce') / net_sales).values
                fig_bar.add_trace(go.Bar(
                    x=df_costs.columns,
                    y=y_pct,
                    name=metric
                ))
            fig_bar.update_layout(title="Costs as Percentage of Revenue", xaxis_title="Year", yaxis_title="Percentage", yaxis=dict(tickformat=".0%"), barmode="group")
            st.plotly_chart(fig_bar, use_container_width=True, key=f"{stock_label}_fv_bar_plot{key_suffix}")
    else:
        st.warning(f"No Fixed/Variable cost data found for {stock_label}.")

def get_outstanding_shares(xls, stock_label, sheet_balance, key_suffix=""):
    df_os = get_metric_data(xls, sheet_balance, ["Outstanding Shares"])
    if not df_os.empty:
        os_val = pd.to_numeric(df_os.iloc[0].dropna(), errors='coerce')
        if not os_val.empty:
            return float(os_val.iloc[0])
    return st.number_input(f"Enter Outstanding Shares for {stock_label}", value=1_000_000.0, step=1000.0, key=f"{stock_label}_os_input{key_suffix}")

def display_dcf_analysis(xls, stock_label, sheet_cf, sheet_balance, key_suffix=""):
    st.subheader(f"{stock_label} – DCF Analysis")
    # Use the new helper to compute FCF from Cash Provided by Operations minus Capital Spending
    df_fcf = get_free_cash_flow(xls, sheet_cf, skiprows=4, metric_col=1, data_col_start=2, data_col_end=7)
    if df_fcf.empty:
        st.error("Could not compute Free Cash Flow (ensure that either a 'Free Cash Flow' row exists or that 'Cash Provided by Operations' and 'Capital Spending' are present).")
        return None
    free_cf_data = pd.to_numeric(df_fcf.iloc[0].dropna(), errors='coerce')
    if len(free_cf_data) < 5:
        st.error("Not enough FCF data (need at least 5 years).")
        return None
    fcf_last5 = free_cf_data[-5:].values.astype(float)
    growth_rates = [(fcf_last5[i] / fcf_last5[i-1] - 1) for i in range(1, len(fcf_last5))]
    avg_growth = np.mean(growth_rates)
    std_growth = np.std(growth_rates)
    st.write(f"**Avg Growth Rate (5 Years):** {avg_growth:.2%}")
    st.write(f"**Std Deviation of Growth:** {std_growth:.2%}")
    scenarios = {
        "Conservative": avg_growth - 0.75 * std_growth,
        "Expected": avg_growth,
        "Aggressive": avg_growth + 0.75 * std_growth
    }
    projections = {}
    for scenario, g_rate in scenarios.items():
        proj_vals = []
        last_val = fcf_last5[-1]
        for _ in range(5):
            next_val = last_val * (1 + g_rate)
            proj_vals.append(next_val)
            last_val = next_val
        projections[scenario] = proj_vals
    proj_df = pd.DataFrame.from_dict(projections, orient="index", columns=[f"Year{i}" for i in range(1,6)])
    st.dataframe(proj_df, use_container_width=True)
    wacc = st.number_input(f"Enter WACC for {stock_label} (decimal)", value=0.10, min_value=0.0, step=0.01, key=f"{stock_label}_wacc{key_suffix}")
    terminal_growth = 0.02
    dcf_values = {}
    for scenario, proj_vals in projections.items():
        discounted = [proj_vals[i] / ((1 + wacc) ** (i+1)) for i in range(5)]
        terminal_val = (proj_vals[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        discounted_terminal = terminal_val / ((1 + wacc) ** 5)
        total_val = sum(discounted) + discounted_terminal
        dcf_values[scenario] = total_val
    dcf_df = pd.DataFrame.from_dict(dcf_values, orient="index", columns=["Enterprise Value"])
    st.dataframe(dcf_df, use_container_width=True)
    outstanding = get_outstanding_shares(xls, stock_label, sheet_balance, key_suffix=key_suffix)
    st.write(f"**Outstanding Shares:** {outstanding:,.0f}")
    per_share = {scenario: val / outstanding for scenario, val in dcf_values.items()}
    per_share_df = pd.DataFrame.from_dict(per_share, orient="index", columns=["Per Share Value"])
    st.write("**DCF Valuation Per Share:**")
    st.dataframe(per_share_df, use_container_width=True)
    safe_zone = {scenario: val * 0.75 for scenario, val in per_share.items()}
    safe_zone_df = pd.DataFrame.from_dict(safe_zone, orient="index", columns=["Safe Zone Price"])
    st.write("**Safe Zone Price (25% Margin):**")
    st.dataframe(safe_zone_df, use_container_width=True)
    current_market_price = st.number_input(f"Enter Current Market Price for {stock_label}", value=120.0, step=1.0, key=f"{stock_label}_cmp{key_suffix}")
    recommendation = {scenario: "BUY" if current_market_price < safe_zone[scenario] else "SELL" for scenario in per_share.keys()}
    rec_df = pd.DataFrame.from_dict(recommendation, orient="index", columns=["Recommendation"])
    st.write("**Buy/Sell Recommendation:**")
    st.dataframe(rec_df, use_container_width=True)
    fig_proj = plot_metrics_plotly(proj_df, f"{stock_label} – Projected FCF (Not Discounted)")
    st.plotly_chart(fig_proj, use_container_width=True, key=f"{stock_label}_dcf_proj_plot{key_suffix}")
    return outstanding

def display_additional_valuation_multiples(xls, stock_label, outstanding_shares, sheet_income, sheet_cf, sheet_balance, key_suffix=""):
    st.subheader(f"{stock_label} – Additional Valuation Multiples")
    op_sheet = st.selectbox(f"Select Operating Metrics sheet for {stock_label} (for multiples)", 
                             xls.sheet_names, key=f"{stock_label}_op_mult{key_suffix}")
    df_op = get_metric_data(xls, op_sheet, [
        "Price to Free Cash Flow", "Price to Earnings", "Price to Operating Income",
        "Price to Gross Profit", "Price to Book", "Price to Tangible Book"
    ])
    if df_op.empty:
        st.warning(f"No Operating Metrics data for valuation multiples found for {stock_label}.")
        return
    if "CAGR" in df_op.columns:
        df_op_val = df_op.drop(columns="CAGR")
    else:
        df_op_val = df_op.copy()
    # For P/FCF, use computed FCF; for others, use underlying data from Income or Balance sheets.
    df_fcf = get_free_cash_flow(xls, sheet_cf, skiprows=4, metric_col=1, data_col_start=2, data_col_end=7)
    df_inc = get_metric_data(xls, sheet_income, ["Net Income", "Operating Profit", "Gross Profit"])
    df_bs = get_metric_data(xls, sheet_balance, ["Total Stockholders' Equity", "Tangible Book Value"])
    
    valuation_mapping = {
        "P/FCF": ("Price to Free Cash Flow", "Free Cash Flow", df_fcf),
        "P/E": ("Price to Earnings", "Net Income", df_inc),
        "P/Operating Income": ("Price to Operating Income", "Operating Profit", df_inc),
        "P/Gross Profit": ("Price to Gross Profit", "Gross Profit", df_inc),
        "P/B": ("Price to Book", "Total Stockholders' Equity", df_bs),
        "P/Tangible Book": ("Price to Tangible Book", "Tangible Book Value", df_bs)
    }
    
    def project_underlying(source_df, underlying_metric, years=5):
        if "CAGR" in source_df.columns:
            source_df = source_df.drop(columns="CAGR")
        try:
            hist_series = pd.to_numeric(source_df.loc[underlying_metric].dropna(), errors='coerce')
        except KeyError:
            return None, None
        if hist_series.empty:
            return None, None
        latest_val = hist_series.iloc[0]
        cagr = calculate_cagr(hist_series)
        if cagr is None:
            cagr = 0
        projected = [latest_val * ((1 + cagr) ** i) for i in range(1, years+1)]
        return projected, cagr
    
    multiples_proj_results = {}
    for method, (mult_row, underlying_metric, source_df) in valuation_mapping.items():
        if mult_row not in df_op_val.index:
            multiples_proj_results[method] = "Multiple data not found"
            continue
        hist_mult = pd.to_numeric(df_op_val.loc[mult_row].dropna(), errors='coerce').values
        if len(hist_mult) < 5:
            multiples_proj_results[method] = "Not enough historical multiple data"
            continue
        avg_mult = np.mean(hist_mult)
        std_mult = np.std(hist_mult)
        conservative_mult = avg_mult - 0.75 * std_mult
        expected_mult = avg_mult
        aggressive_mult = avg_mult + 0.75 * std_mult
        if underlying_metric not in source_df.index:
            multiples_proj_results[method] = "Underlying metric data not found"
            continue
        hist_underlying = pd.to_numeric(source_df.loc[underlying_metric].dropna(), errors='coerce').values
        if len(hist_underlying) < 5:
            multiples_proj_results[method] = "Not enough historical underlying data"
            continue
        projected_underlying, underlying_cagr = project_underlying(source_df, underlying_metric, years=5)
        if projected_underlying is None:
            multiples_proj_results[method] = "Underlying metric data not found or empty"
            continue
        proj_prices_conservative = [projected_underlying[i] * conservative_mult for i in range(5)]
        proj_prices_expected     = [projected_underlying[i] * expected_mult for i in range(5)]
        proj_prices_aggressive   = [projected_underlying[i] * aggressive_mult for i in range(5)]
        multiples_proj_results[method] = {
            "Multiple (Conservative)": conservative_mult,
            "Multiple (Expected)": expected_mult,
            "Multiple (Aggressive)": aggressive_mult,
            "Target Prices (Conservative)": proj_prices_conservative,
            "Target Prices (Expected)": proj_prices_expected,
            "Target Prices (Aggressive)": proj_prices_aggressive,
            "Underlying CAGR": underlying_cagr
        }
    
    available_methods = [m for m in multiples_proj_results.keys() if isinstance(multiples_proj_results[m], dict)]
    selected_methods = st.multiselect(f"Select valuation multiples to include for {stock_label}", available_methods, default=available_methods, key=f"{stock_label}_mult_sel{key_suffix}")
    
    def format_cell(value):
        per_share = value / outstanding_shares
        return f"{value:.2f}\n(${per_share:.4f}/share)"
    
    master_rows = []
    def compute_cagr(series):
        if series[0] <= 0 or len(series) < 2:
            return None
        return (series[-1] / series[0]) ** (1 / (len(series) - 1)) - 1
    
    for method in selected_methods:
        result = multiples_proj_results[method]
        for scenario in ["Conservative", "Expected", "Aggressive"]:
            raw_prices = result[f"Target Prices ({scenario})"]
            per_share_prices = [price / outstanding_shares for price in raw_prices]
            cagr_val = compute_cagr(np.array(per_share_prices))
            if scenario == "Conservative":
                mult_val = result["Multiple (Conservative)"]
            elif scenario == "Expected":
                mult_val = result["Multiple (Expected)"]
            else:
                mult_val = result["Multiple (Aggressive)"]
            master_rows.append({
                "Metric": f"{method} ({scenario})",
                "Multiple": f"{mult_val:.2f}",
                "Year1": format_cell(raw_prices[0]),
                "Year2": format_cell(raw_prices[1]),
                "Year3": format_cell(raw_prices[2]),
                "Year4": format_cell(raw_prices[3]),
                "Year5": format_cell(raw_prices[4]),
                "CAGR": f"{cagr_val:.2%}" if cagr_val is not None else "N/A"
            })
    
    df_master_methods = pd.DataFrame(master_rows)
    st.write(f"### {stock_label} – Valuation Multiples (Individual Methods)")
    st.dataframe(df_master_methods, use_container_width=True)
    
    fig_individual = go.Figure()
    for method in selected_methods:
        result = multiples_proj_results[method]
        if not isinstance(result, dict):
            continue
        for scenario in ["Conservative", "Expected", "Aggressive"]:
            raw_prices = result[f"Target Prices ({scenario})"]
            per_share_prices = [price / outstanding_shares for price in raw_prices]
            fig_individual.add_trace(go.Scatter(
                x=[f"Year{i}" for i in range(1, 6)],
                y=per_share_prices,
                mode='lines+markers',
                name=f"{method} ({scenario})"
            ))
    fig_individual.update_layout(
        title=f"{stock_label} – Individual Method Projections (Per Share)",
        xaxis_title="Projection Year",
        yaxis_title="Target Price per Share",
        hovermode="x unified"
    )
    st.plotly_chart(fig_individual, use_container_width=True, key=f"{stock_label}_mult_ind_plot{key_suffix}")
    
    agg_data = {"Conservative": [], "Expected": [], "Aggressive": []}
    for scenario in ["Conservative", "Expected", "Aggressive"]:
        method_arrays = []
        for method in selected_methods:
            arr = np.array(multiples_proj_results[method][f"Target Prices ({scenario})"], dtype=float) / outstanding_shares
            method_arrays.append(arr)
        if method_arrays:
            avg_array = np.nanmean(np.array(method_arrays), axis=0)
            agg_data[scenario] = avg_array
        else:
            agg_data[scenario] = [np.nan] * 5
    df_master_avg = pd.DataFrame(agg_data, index=[f"Year{i}" for i in range(1, 6)]).T
    master_cagr = {}
    for scenario in df_master_avg.index:
        values = df_master_avg.loc[scenario].astype(float).values
        cagr_avg = compute_cagr(values)
        master_cagr[scenario] = f"{cagr_avg:.2%}" if cagr_avg is not None else "N/A"
    df_master_avg["CAGR"] = pd.Series(master_cagr)
    st.write(f"### {stock_label} – Scenario Averages")
    st.dataframe(df_master_avg.style.format({
        "Year1": "{:.4f}",
        "Year2": "{:.4f}",
        "Year3": "{:.4f}",
        "Year4": "{:.4f}",
        "Year5": "{:.4f}"
    }), use_container_width=True)
    
    fig_master = go.Figure()
    for scenario in df_master_avg.index:
        y_values = df_master_avg.loc[scenario, df_master_avg.columns[:-1]].astype(float).values
        fig_master.add_trace(go.Scatter(
            x=[f"Year{i}" for i in range(1, 6)],
            y=y_values,
            mode='lines+markers',
            name=scenario
        ))
    fig_master.update_layout(
        title=f"{stock_label} – Master Averaged Target Price Projections (Per Share)",
        xaxis_title="Projection Year",
        yaxis_title="Average Target Price per Share",
        hovermode="x unified"
    )
    st.plotly_chart(fig_master, use_container_width=True, key=f"{stock_label}_mult_master_plot{key_suffix}")

#####################################
# Comparison Bar Chart Functions
#####################################

def compare_bar_chart(xls, sheet1, sheet2, metric, category, num_years):
    df1 = get_metric_data(xls, sheet1, [metric])
    df2 = get_metric_data(xls, sheet2, [metric])
    if df1.empty or df2.empty:
        st.warning(f"Metric '{metric}' not found in one of the sheets for {category}.")
        return
    df1.columns = hist_years
    df2.columns = hist_years
    years_used = hist_years[:num_years]
    values1 = pd.to_numeric(df1.loc[metric][years_used], errors='coerce').values
    values2 = pd.to_numeric(df2.loc[metric][years_used], errors='coerce').values
    fig = go.Figure(data=[
        go.Bar(name='Stock 1', x=years_used, y=values1),
        go.Bar(name='Stock 2', x=years_used, y=values2)
    ])
    fig.update_layout(barmode='group',
                      title=f"{category} Comparison: {metric} (over last {num_years} years)",
                      xaxis_title="Year",
                      yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True, key=f"comp_bar_{category}_{metric}")

def compare_fcf_projections_bar_chart(xls, sheet_cf1, sheet_cf2, scenario, num_years):
    df_fcf1 = get_free_cash_flow(xls, sheet_cf1, skiprows=4, metric_col=1, data_col_start=2, data_col_end=7)
    df_fcf2 = get_free_cash_flow(xls, sheet_cf2, skiprows=4, metric_col=1, data_col_start=2, data_col_end=7)
    if df_fcf1.empty or df_fcf2.empty:
        st.warning("FCF data not found in one of the Cash Flow sheets.")
        return
    free_cf1 = pd.to_numeric(df_fcf1.iloc[0].dropna(), errors='coerce')
    free_cf2 = pd.to_numeric(df_fcf2.iloc[0].dropna(), errors='coerce')
    if len(free_cf1) < 5 or len(free_cf2) < 5:
        st.warning("Not enough FCF data for projections.")
        return
    fcf_last5_1 = free_cf1[-5:].values.astype(float)
    fcf_last5_2 = free_cf2[-5:].values.astype(float)
    def project_fcf(fcf_array):
        gr = [(fcf_array[i] / fcf_array[i-1] - 1) for i in range(1, len(fcf_array))]
        avg_gr = np.mean(gr)
        std_gr = np.std(gr)
        scenarios = {
            "Conservative": avg_gr - 0.75 * std_gr,
            "Expected": avg_gr,
            "Aggressive": avg_gr + 0.75 * std_gr
        }
        proj = []
        last_val = fcf_array[-1]
        for _ in range(5):
            last_val = last_val * (1 + scenarios[scenario])
            proj.append(last_val)
        return proj
    proj1 = project_fcf(fcf_last5_1)
    proj2 = project_fcf(fcf_last5_2)
    years_used = [f"Year{i}" for i in range(1, num_years+1)]
    proj1 = proj1[:num_years]
    proj2 = proj2[:num_years]
    fig = go.Figure(data=[
        go.Bar(name='Stock 1', x=years_used, y=proj1),
        go.Bar(name='Stock 2', x=years_used, y=proj2)
    ])
    fig.update_layout(barmode='group',
                      title=f"FCF Projections Comparison ({scenario} scenario, over {num_years} years)",
                      xaxis_title="Projection Year",
                      yaxis_title="Projected FCF")
    st.plotly_chart(fig, use_container_width=True, key=f"comp_bar_FCF_{scenario}")

def compare_valuation_multiples_bar_chart(xls, sheet_op1, sheet_inc1, sheet_cf1, sheet_balance1,
                                            sheet_op2, sheet_inc2, sheet_cf2, sheet_balance2,
                                            method, scenario, num_years):
    df_op1 = get_metric_data(xls, sheet_op1, [method])
    df_op2 = get_metric_data(xls, sheet_op2, [method])
    if df_op1.empty or df_op2.empty:
        st.warning(f"Valuation multiple '{method}' not found for one of the stocks.")
        return
    mapping = {
        "P/FCF": ("Free Cash Flow", sheet_cf1, sheet_cf2),
        "P/E": ("Net Income", sheet_inc1, sheet_inc2),
        "P/Operating Income": ("Operating Profit", sheet_inc1, sheet_inc2),
        "P/Gross Profit": ("Gross Profit", sheet_inc1, sheet_inc2),
        "P/B": ("Total Stockholders' Equity", sheet_balance1, sheet_balance2),
        "P/Tangible Book": ("Tangible Book Value", sheet_balance1, sheet_balance2)
    }
    if method not in mapping:
        st.warning(f"Method {method} not supported for comparison.")
        return
    underlying_metric, underlying_sheet1, underlying_sheet2 = mapping[method]
    df_under1 = get_metric_data(xls, underlying_sheet1, [underlying_metric])
    df_under2 = get_metric_data(xls, underlying_sheet2, [underlying_metric])
    if df_under1.empty or df_under2.empty:
        st.warning(f"Underlying metric '{underlying_metric}' not found for one of the stocks.")
        return
    df_under1.columns = hist_years
    df_under2.columns = hist_years
    mult1 = pd.to_numeric(df_op1.loc[method].dropna(), errors='coerce').mean()
    mult2 = pd.to_numeric(df_op2.loc[method].dropna(), errors='coerce').mean()
    def project_value(df_under):
        values = pd.to_numeric(df_under.iloc[0].dropna(), errors='coerce').values
        if len(values) < 5:
            return None
        cagr = calculate_cagr(values[-5:])
        if cagr is None:
            cagr = 0
        proj = []
        last = values[-1]
        for _ in range(5):
            last = last * (1 + cagr)
            proj.append(last)
        return proj
    proj1 = project_value(df_under1)
    proj2 = project_value(df_under2)
    if proj1 is None or proj2 is None:
        st.warning("Not enough data for projecting underlying metric for valuation multiples.")
        return
    target1 = [proj1[i] * mult1 for i in range(num_years)]
    target2 = [proj2[i] * mult2 for i in range(num_years)]
    years_used = [f"Year{i}" for i in range(1, num_years+1)]
    fig = go.Figure(data=[
        go.Bar(name='Stock 1', x=years_used, y=target1),
        go.Bar(name='Stock 2', x=years_used, y=target2)
    ])
    fig.update_layout(barmode='group',
                      title=f"Valuation Multiples Comparison: {method} ({scenario} scenario) over {num_years} years",
                      xaxis_title="Projection Year",
                      yaxis_title="Target Price")
    st.plotly_chart(fig, use_container_width=True, key=f"comp_bar_valuation_{method}_{scenario}")

#####################################
# Main Multi-Stock Dashboard (Single Excel File)
#####################################
uploaded_file = st.file_uploader("Upload one Excel file containing all sheets for both stocks", type=["xlsx"], key="single_file")
if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    
    # Sidebar: Select sheets for Stock 1 and Stock 2
    st.sidebar.subheader("Sheet Selection")
    st.sidebar.write("**Stock 1 Sheets**")
    sheet_income1 = st.sidebar.selectbox("Stock 1 Income Statement sheet", xls.sheet_names, key="s1_income")
    sheet_balance1 = st.sidebar.selectbox("Stock 1 Balance Sheet sheet", xls.sheet_names, key="s1_balance")
    sheet_cf1 = st.sidebar.selectbox("Stock 1 Cash Flow sheet", xls.sheet_names, key="s1_cf")
    sheet_op1 = st.sidebar.selectbox("Stock 1 Operating Metrics sheet", xls.sheet_names, key="s1_op")
    
    st.sidebar.write("**Stock 2 Sheets**")
    sheet_income2 = st.sidebar.selectbox("Stock 2 Income Statement sheet", xls.sheet_names, key="s2_income")
    sheet_balance2 = st.sidebar.selectbox("Stock 2 Balance Sheet sheet", xls.sheet_names, key="s2_balance")
    sheet_cf2 = st.sidebar.selectbox("Stock 2 Cash Flow sheet", xls.sheet_names, key="s2_cf")
    sheet_op2 = st.sidebar.selectbox("Stock 2 Operating Metrics sheet", xls.sheet_names, key="s2_op")
    
    tab1, tab2, tab3 = st.tabs(["Stock 1 Dashboard", "Stock 2 Dashboard", "Comparison Dashboard"])
    
    with tab1:
        st.title("Stock 1 Dashboard")
        display_income_statement(xls, "Stock 1", sheet_income1, key_suffix="_s1")
        display_balance_sheet(xls, "Stock 1", sheet_balance1, key_suffix="_s1")
        display_cash_flow(xls, "Stock 1", sheet_cf1, key_suffix="_s1")
        display_operating_metrics(xls, "Stock 1", sheet_op1, key_suffix="_s1")
        display_fixed_variable_costs(xls, "Stock 1", sheet_income1, sheet_cf1, key_suffix="_s1")
        outstanding1 = display_dcf_analysis(xls, "Stock 1", sheet_cf1, sheet_balance1, key_suffix="_s1")
        if outstanding1 is None:
            outstanding1 = get_outstanding_shares(xls, "Stock 1", sheet_balance1, key_suffix="_s1")
        st.write("---")
        display_additional_valuation_multiples(xls, "Stock 1", outstanding1, sheet_income1, sheet_cf1, sheet_balance1, key_suffix="_s1")
    
    with tab2:
        st.title("Stock 2 Dashboard")
        display_income_statement(xls, "Stock 2", sheet_income2, key_suffix="_s2")
        display_balance_sheet(xls, "Stock 2", sheet_balance2, key_suffix="_s2")
        display_cash_flow(xls, "Stock 2", sheet_cf2, key_suffix="_s2")
        display_operating_metrics(xls, "Stock 2", sheet_op2, key_suffix="_s2")
        display_fixed_variable_costs(xls, "Stock 2", sheet_income2, sheet_cf2, key_suffix="_s2")
        outstanding2 = display_dcf_analysis(xls, "Stock 2", sheet_cf2, sheet_balance2, key_suffix="_s2")
        if outstanding2 is None:
            outstanding2 = get_outstanding_shares(xls, "Stock 2", sheet_balance2, key_suffix="_s2")
        st.write("---")
        display_additional_valuation_multiples(xls, "Stock 2", outstanding2, sheet_income2, sheet_cf2, sheet_balance2, key_suffix="_s2")
    
    with tab3:
        st.title("Comparison Dashboard")
        st.subheader("Income Statement Comparison")
        col1, col2 = st.columns(2)
        with col1:
            compare_bar_chart(xls, sheet_income1, sheet_income2, "Net Sales", "Income Statement", 3)
        with col2:
            compare_bar_chart(xls, sheet_income1, sheet_income2, "Net Income", "Income Statement", 3)
        
        st.subheader("Balance Sheet Comparison")
        col1, col2 = st.columns(2)
        with col1:
            compare_bar_chart(xls, sheet_balance1, sheet_balance2, "Total Current Assets", "Balance Sheet", 3)
        with col2:
            compare_bar_chart(xls, sheet_balance1, sheet_balance2, "Outstanding Shares", "Balance Sheet", 3)
        
        st.subheader("Fixed vs Variable Costs Comparison")
        compare_bar_chart(xls, sheet_income1 if "General and Administrative Expenses" in get_metric_data(xls, sheet_income1, ["General and Administrative Expenses"]).index 
                         else sheet_cf1, 
                         sheet_income2 if "General and Administrative Expenses" in get_metric_data(xls, sheet_income2, ["General and Administrative Expenses"]).index 
                         else sheet_cf2, 
                         "General and Administrative Expenses", "Fixed vs Variable Costs", 3)
        
        st.subheader("Cash Flow Comparison")
        compare_bar_chart(xls, sheet_cf1, sheet_cf2, "Cash Provided by Operations", "Cash Flow", 3)
        
        st.subheader("Free Cash Flow Projections Comparison")
        scenario_choice = st.selectbox("Select FCF projection scenario", ["Conservative", "Expected", "Aggressive"], key="comp_fcf_scenario")
        compare_fcf_projections_bar_chart(xls, sheet_cf1, sheet_cf2, scenario_choice, 3)
        
        st.subheader("Valuation Multiples Comparison")
        val_method = st.selectbox("Select Valuation Multiple Method", ["P/FCF", "P/E", "P/Operating Income", "P/Gross Profit", "P/B", "P/Tangible Book"], key="comp_val_method")
        val_scenario = st.selectbox("Select Valuation Scenario", ["Conservative", "Expected", "Aggressive"], key="comp_val_scenario")
        compare_valuation_multiples_bar_chart(xls, sheet_op1, sheet_income1, sheet_cf1, sheet_balance1,
                                              sheet_op2, sheet_income2, sheet_cf2, sheet_balance2,
                                              val_method, val_scenario, 3)
else:
    st.info("Please upload one Excel file containing all sheets for both stocks.")







        























 















































