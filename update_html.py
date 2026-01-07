#!/usr/bin/env python3
"""
Script to execute notebooks and generate HTML files.
This script runs the notebooks and exports the generated HTML files.
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def execute_notebook_with_papermill(notebook_path, output_path=None):
    """Execute a Jupyter notebook using papermill."""
    print(f"Executing {notebook_path}...")
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    
    result = subprocess.run(
        ["papermill", str(notebook_path), str(output_path), "--log-output"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error executing {notebook_path}:")
        print(result.stderr)
        return False
    print(f"✅ Successfully executed {notebook_path}")
    return True

def export_first_notebook_to_html():
    """
    Export the Plotly figure from options_stock_expected_move.ipynb to index.html.
    This function executes the notebook and extracts the figure to create HTML.
    """
    print("Processing options_stock_expected_move.ipynb...")
    
    # Execute the notebook first using papermill
    executed_notebook = "options_stock_expected_move_executed.ipynb"
    if not execute_notebook_with_papermill("options_stock_expected_move.ipynb", executed_notebook):
        return False
    
    # Now we need to extract the figure and export it
    # We'll use a Python script that imports from the executed notebook
    export_script = """
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go

# Re-run the notebook code to get the figure
# (This is necessary because we can't easily extract variables from executed notebooks)

# ==== Parameters ====
today = dt.date.today()
spy_ticker = yf.Ticker("SPY")
stock_price = spy_ticker.info['regularMarketPrice']

vix_ticker = yf.Ticker("^VIX")
iv = vix_ticker.info['regularMarketPrice']/100

std = 1
risk_free_rate = 0.041

# ==== Generate Trading Days (excluding weekends) ====
def get_next_trading_day(date, days_ahead):
    current_date = date
    trading_days = 0
    while trading_days < days_ahead:
        current_date += dt.timedelta(days=1)
        if current_date.weekday() < 5:
            trading_days += 1
    return current_date

def get_trading_days_list(start_date, num_days):
    trading_days = []
    current_date = start_date
    for i in range(num_days):
        next_trading_day = get_next_trading_day(current_date, 1)
        trading_days.append(next_trading_day)
        current_date = next_trading_day
    return trading_days

# ==== Get Daily Moves for First 10 Trading Days ====
daily_dates = get_trading_days_list(today, 10)
daily_df = pd.DataFrame({"expiration_day": daily_dates})
daily_df["expiration_day"] = pd.to_datetime(daily_df["expiration_day"])
daily_df["DTE"] = (daily_df["expiration_day"] - pd.to_datetime(today)).dt.days
daily_df["expected_move"] = stock_price * iv * std * (daily_df["DTE"] / 365) ** 0.5
daily_df["upper"] = stock_price + daily_df["expected_move"]
daily_df["lower"] = stock_price - daily_df["expected_move"]
daily_df["type"] = "Daily"

# ==== Generate Weekly Expirations ====
start_weekly = daily_dates[-1] + dt.timedelta(days=1)
days_until_friday = (4 - start_weekly.weekday()) % 7
if days_until_friday == 0:
    days_until_friday = 7
next_friday = start_weekly + dt.timedelta(days=days_until_friday)
end_date = dt.date(2025, 12, 31)
fridays = pd.date_range(next_friday, end_date, freq="W-FRI").date

weekly_df = pd.DataFrame({"expiration_day": fridays})
weekly_df["expiration_day"] = pd.to_datetime(weekly_df["expiration_day"])
weekly_df["DTE"] = (weekly_df["expiration_day"] - pd.to_datetime(today)).dt.days
weekly_df["expected_move"] = stock_price * iv * std * (weekly_df["DTE"] / 365) ** 0.5
weekly_df["upper"] = stock_price + weekly_df["expected_move"]
weekly_df["lower"] = stock_price - weekly_df["expected_move"]
weekly_df["type"] = "Weekly"

# ==== Combine DataFrames ====
today_row = pd.DataFrame({
    "expiration_day": [pd.to_datetime(today)],
    "DTE": [0],
    "expected_move": [0],
    "upper": [stock_price],
    "lower": [stock_price],
    "type": ["Today"]
})

df = pd.concat([today_row, daily_df, weekly_df], ignore_index=True)

# ==== Calculate deltas ====
current_price_rounded = round(stock_price)
strike_range = 15
strike_prices = list(range(current_price_rounded - strike_range,
                          current_price_rounded + strike_range + 1))

all_expiration_dates = daily_dates + list(fridays)

def calculate_call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

delta_table_data = {}
for exp_date in all_expiration_dates:
    dte = (exp_date - today).days
    time_to_exp = dte / 365.0
    row_data = {}
    for strike in strike_prices:
        call_delta = calculate_call_delta(stock_price, strike, time_to_exp, risk_free_rate, iv)
        row_data[strike] = call_delta
    delta_table_data[exp_date] = row_data

delta_2d_df = pd.DataFrame.from_dict(delta_table_data, orient='index')
delta_2d_df.index = pd.to_datetime(delta_2d_df.index)
delta_2d_df = delta_2d_df.sort_index()

# ==== Create Interactive Plotly Chart ====
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['expiration_day'],
    y=df['upper'],
    mode='lines+markers',
    name='',
    line=dict(color='blue', width=2),
    marker=dict(size=4, color='blue'),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=df['expiration_day'],
    y=df['lower'],
    mode='lines+markers',
    name='',
    line=dict(color='blue', width=2),
    marker=dict(size=4, color='blue'),
    fill='tonexty',
    fillcolor='rgba(0,100,200,0.1)',
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=[df['expiration_day'].min(), df['expiration_day'].max()],
    y=[stock_price, stock_price],
    mode='lines',
    line=dict(color='gray', width=2, dash='dot'),
    showlegend=False,
    hoverinfo='skip',
    name='Current Price Line'
))

fig.add_annotation(
    x=df["expiration_day"].iloc[len(df)//2],
    y=stock_price,
    text=f'Current Price:  ${stock_price}',
    showarrow=False,
    font=dict(size=10, color='gray', family="Arial Black"),
    bgcolor='white',
    bordercolor='gray',
    borderwidth=1,
    xanchor='left',
    yanchor='middle'
)

daily_mask = df["type"] == "Daily"
daily_count = 0
annotations = []

for i, row in df[daily_mask].iterrows():
    if i > 0:
        if daily_count % 5 == 0:
            annotations.append(dict(
                x=row["expiration_day"],
                y=row["upper"] + 4,
                text=f"{row['upper']:.3f}",
                showarrow=False,
                font=dict(size=8, color="blue", family="Arial Black"),
                xanchor="center",
                yanchor="bottom"
            ))
            annotations.append(dict(
                x=row["expiration_day"],
                y=row["lower"] - 4,
                text=f"{row['lower']:.3f}",
                showarrow=False,
                font=dict(size=8, color="blue", family="Arial Black"),
                xanchor="center",
                yanchor="top"
            ))
        daily_count += 1

weekly_mask = df["type"] == "Weekly"
for i, row in df[weekly_mask].iterrows():
    annotations.append(dict(
        x=row["expiration_day"],
        y=row["upper"] + 2,
        text=f"{row['upper']:.3f}",
        showarrow=False,
        font=dict(size=10, color="blue", family="Arial Black"),
        xanchor="center",
        yanchor="bottom"
    ))
    annotations.append(dict(
        x=row["expiration_day"],
        y=row["lower"] - 2,
        text=f"{row['lower']:.3f}",
        showarrow=False,
        font=dict(size=10, color="blue", family="Arial Black"),
        xanchor="center",
        yanchor="top"
    ))

hover_data = []
for exp_date, row in delta_2d_df.iterrows():
    for strike_price, probability in row.items():
        hover_data.append({
            'expiration_date': exp_date,
            'strike_price': strike_price,
            'probability': probability
        })

hover_df = pd.DataFrame(hover_data)

fig.add_trace(go.Scatter(
    x=hover_df['expiration_date'],
    y=hover_df['strike_price'],
    mode='markers',
    marker=dict(
        size=8,
        color='rgba(0,0,0,0)',
        line=dict(width=0)
    ),
    showlegend=False,
    hovertemplate='<b>Strike Price: $%{y}</b><br>' +
                  'Expiration: %{x|%b %d}<br>' +
                  'Probability Above: %{customdata:.1%}<br>' +
                  '<extra></extra>',
    customdata=hover_df['probability'],
    name='Probabilities'
))

selected_daily = df[df["type"] == "Daily"].iloc[::5]
weekly_dates = df[df["type"] == "Weekly"].iloc[::1]
all_selected_dates = list(selected_daily["expiration_day"]) + list(weekly_dates["expiration_day"])
all_selected_dates = sorted(all_selected_dates)

fig.update_layout(
    title=dict(
        text=f"Probability Analysis<br>Mode: ITM | Range: 68.27% (±1σ) | Volatility: {iv*100:.2f}%",
        x=0.5,
        font=dict(size=14, family="Arial Black")
    ),
    xaxis_title='Expiration Date',
    yaxis_title='Stock Price ($)',
    hovermode='closest',
    width=1400,
    height=700,
    showlegend=False,
    plot_bgcolor='white',
    annotations=annotations,
    shapes=[
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="black", width=1),
            fillcolor="rgba(0,0,0,0)"
        )
    ],
    xaxis=dict(
        tickvals=all_selected_dates,
        ticktext=[d.strftime('%Y-%m-%d') for d in all_selected_dates],
        tickangle=70,
        gridcolor='lightgray',
        gridwidth=1,
        linecolor='black',
        linewidth=1,
        mirror=True,
        showspikes=True,
        spikecolor="gray",
        spikethickness=1,
        spikedash="dot",
        spikemode="across"
    ),
    yaxis=dict(
        tickformat='$,.0f',
        gridcolor='lightgray',
        gridwidth=1,
        linecolor='black',
        linewidth=1,
        mirror=True,
        showspikes=True,
        spikecolor="gray",
        spikethickness=1,
        spikedash="dot",
        spikemode="across"
    ),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="black",
        font_size=12
        
    )
)

# Export to HTML
plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

html_template = (
    "<html>\\n"
    "<head>\\n"
    "    <title>SPY Options Expected Move</title>\\n"
    '    <meta charset="utf-8">\\n'
    "    <style>\\n"
    "        body { font-family: Arial; margin: 40px; }\\n"
    "        nav a { text-decoration: none; color: #0077b5; }\\n"
    "    </style>\\n"
    "</head>\\n"
    "<body>\\n"
    "\\n"
    "    <!-- Load Navbar -->\\n"
    '    <div id="navbar-placeholder"></div>\\n'
    "\\n"
    "    <h1>📊 SPY Options Expected Move</h1>\\n"
    "\\n"
    "    <!-- Insert Plotly chart -->\\n"
    + plotly_html + "\\n"
    "\\n"
    '    <footer style="margin-top:50px; font-size:14px; color:gray;">\\n'
    "        Created by Nianguang Zhao | Hosted on GitHub Pages\\n"
    "    </footer>\\n"
    "\\n"
    "    <script>\\n"
    "        // Load navbar.html into the placeholder\\n"
    '        fetch("navbar.html")\\n'
    "            .then(response => response.text())\\n"
    "            .then(data => {\\n"
    '                document.getElementById("navbar-placeholder").innerHTML = data;\\n'
    "            });\\n"
    "    </script>\\n"
    "\\n"
    "</body>\\n"
    "</html>\\n"
)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("✅ Exported index.html successfully!")
"""
    
    # Write and execute the export script
    with open("_temp_export.py", "w") as f:
        f.write(export_script)
    
    try:
        result = subprocess.run(
            [sys.executable, "_temp_export.py"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        if result.returncode != 0:
            print(f"Error exporting index.html:")
            print(result.stderr)
            return False
        print(result.stdout)
        return True
    finally:
        # Clean up temp file
        if os.path.exists("_temp_export.py"):
            os.remove("_temp_export.py")

def main():
    """Main function to execute notebooks and generate HTML files."""
    print("=" * 60)
    print("Starting HTML update process...")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Process first notebook (options_stock_expected_move.ipynb)
    if not export_first_notebook_to_html():
        print("❌ Failed to generate index.html")
        sys.exit(1)
    
    # Process second notebook (stock_tracker.ipynb)
    # This notebook already exports to HTML, so we just need to execute it
    executed_notebook = "stock_tracker_executed.ipynb"
    if not execute_notebook_with_papermill("stock_tracker.ipynb", executed_notebook):
        print("❌ Failed to generate stock_tracker.html")
        sys.exit(1)
    
    # Clean up executed notebook files
    for temp_file in ["options_stock_expected_move_executed.ipynb", "stock_tracker_executed.ipynb"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up {temp_file}")
    
    print("=" * 60)
    print("✅ All HTML files updated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

