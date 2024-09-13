import pandas as pd
import numpy as np
import json
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf

plt.switch_backend('Agg')  # Use non-interactive backend

def load_config(config_file='port_config.json'):
    with open(config_file, 'r') as f:
        return json.load(f)

def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.pivot(columns='Ticker', values='Adj Close')

def calculate_returns(df):
    return df.pct_change().dropna()

def adjust_weights(weights, num_tickers):
    if len(weights) != num_tickers:
        return [1/num_tickers] * num_tickers
    return weights

class Portfolio:
    def __init__(self, returns, weights, initial_investment):
        self.returns = returns
        self.initial_weights = np.array(adjust_weights(weights, returns.shape[1]))
        self.current_weights = self.initial_weights.copy()
        self.initial_investment = initial_investment
        self.tickers = list(returns.columns)
        self.fetch_dividend_data()
        self.calculate_initial_shares()
        self.update_portfolio()

    def fetch_dividend_data(self):
        self.dividends = yf.download(" ".join(self.tickers), period="1y", actions=True)['Dividends']
        self.prices = yf.download(" ".join(self.tickers), period="1y")['Adj Close']

    def calculate_initial_shares(self):
        self.initial_shares = {}
        for ticker in self.tickers:
            initial_investment_in_ticker = self.initial_investment * self.initial_weights[self.tickers.index(ticker)]
            self.initial_shares[ticker] = initial_investment_in_ticker / self.prices[ticker].iloc[0]

    def calculate_total_dividends(self):
        total_dividends = {}
        for ticker in self.tickers:
            ticker_dividends = self.dividends[ticker].sum()
            weight_ratio = self.current_weights[self.tickers.index(ticker)] / self.initial_weights[self.tickers.index(ticker)]
            total_dividends[ticker] = ticker_dividends * self.initial_shares[ticker] * weight_ratio
        return total_dividends

    def update_weights(self, new_weights):
        self.current_weights = np.array(new_weights)
        self.update_portfolio()

    def update_portfolio(self):
        self.portfolio_returns = self.calculate_portfolio_returns()
        self.portfolio_value = self.calculate_portfolio_value()
        self.portfolio_volatility = self.calculate_portfolio_volatility()
        self.var = self.calculate_var()
        self.sharpe_ratio = self.calculate_sharpe_ratio(0.02)  # Assuming 2% risk-free rate
        self.sortino_ratio = self.calculate_sortino_ratio(0.02)
        self.cumulative_returns = self.calculate_cumulative_returns()
        self.total_dividends = self.calculate_total_dividends()

    def calculate_portfolio_returns(self):
        return np.dot(self.returns, self.current_weights)

    def calculate_portfolio_value(self):
        cumulative_returns = np.cumprod(1 + self.portfolio_returns)
        return self.initial_investment * cumulative_returns[-1]

    def calculate_portfolio_volatility(self):
        return np.sqrt(np.dot(self.current_weights.T, np.dot(self.returns.cov() * 252, self.current_weights)))

    def calculate_var(self, confidence_level=0.95):
        var_percent = -np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
        return var_percent * 100

    def calculate_sharpe_ratio(self, risk_free_rate):
        excess_returns = self.portfolio_returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, risk_free_rate, target_return=0):
        excess_returns = self.portfolio_returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < target_return]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        return (excess_returns.mean() * 252) / downside_deviation if downside_deviation != 0 else 0

    def calculate_cumulative_returns(self):
        weighted_returns = self.returns.mul(self.current_weights, axis=1).sum(axis=1)
        return (1 + weighted_returns).cumprod()

class PortfolioGUI:
    def __init__(self, master, returns, initial_weights, initial_investment):
        self.master = master
        self.returns = returns
        self.initial_weights = adjust_weights(initial_weights, returns.shape[1])
        self.initial_investment = initial_investment
        self.portfolio = Portfolio(returns, self.initial_weights, initial_investment)

        self.master.title("Portfolio Analyzer")
        self.master.geometry("1400x800")  # Increased window size
        self.create_widgets()
        self.create_graph()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.weight_entries = []
        self.asset_checkboxes = []
        self.asset_vars = []

        for i, ticker in enumerate(self.returns.columns):
            label = ttk.Label(left_frame, text=f"{ticker}:")
            label.grid(row=i, column=0, padx=5, pady=5, sticky="e")

            spinbox = ttk.Spinbox(left_frame, from_=0, to=1, increment=0.01, width=10)
            spinbox.set(self.initial_weights[i])
            spinbox.grid(row=i, column=1, padx=5, pady=5)
            spinbox.bind("<KeyRelease>", self.update_portfolio)
            spinbox.bind("<<Increment>>", self.update_portfolio)
            spinbox.bind("<<Decrement>>", self.update_portfolio)
            self.weight_entries.append(spinbox)

            var = tk.BooleanVar(value=True)
            checkbox = ttk.Checkbutton(left_frame, variable=var, command=self.update_graph)
            checkbox.grid(row=i, column=2, padx=5, pady=5)
            self.asset_checkboxes.append(checkbox)
            self.asset_vars.append(var)

        update_button = ttk.Button(left_frame, text="Update Portfolio", command=self.update_portfolio)
        update_button.grid(row=len(self.returns.columns), column=0, columnspan=3, pady=10)

        # Create info frame for text boxes
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        # Portfolio information text box (left side)
        self.portfolio_text = tk.Text(info_frame, height=8, width=50, wrap=tk.WORD)
        self.portfolio_text.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        # Dividend information text box (right side)
        self.dividend_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
        self.dividend_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.fig, self.ax = plt.subplots(figsize=(10, 6))  # Increased figure size
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def create_graph(self):
        self.update_graph()

    def update_graph(self):
        self.ax.clear()
        cumulative_returns = (1 + self.returns).cumprod()

        for i, (ticker, returns) in enumerate(cumulative_returns.items()):
            if self.asset_vars[i].get():
                returns.plot(ax=self.ax, label=ticker)

        self.portfolio.cumulative_returns.plot(ax=self.ax, label='Portfolio', linewidth=3, color='black')

        self.ax.set_title("Cumulative Returns")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Cumulative Return")
        self.ax.legend()
        self.fig.tight_layout()  # Adjust the layout to use the full figure
        self.canvas.draw()

    def update_portfolio(self, event=None):
        new_weights = [float(entry.get()) for entry in self.weight_entries]
        self.portfolio.update_weights(new_weights)
        self.display_results()
        self.update_graph()

    def display_results(self):
        # Clear both text boxes
        self.portfolio_text.delete(1.0, tk.END)
        self.dividend_text.delete(1.0, tk.END)

        # Portfolio information
        portfolio_info = f"Starting Portfolio Value: ${self.initial_investment:,.2f}\n"
        portfolio_info += f"Current Portfolio Value: ${self.portfolio.portfolio_value:,.2f}\n"
        portfolio_info += f"Total Return: {(self.portfolio.portfolio_value / self.initial_investment - 1) * 100:.2f}%\n"
        portfolio_info += f"Annualized Return: {self.portfolio.portfolio_returns.mean() * 252 * 100:.2f}%\n"
        portfolio_info += f"Portfolio Volatility: {self.portfolio.portfolio_volatility * 100:.2f}%\n"
        portfolio_info += f"VaR (95%): {self.portfolio.var:.2f}%\n"
        portfolio_info += f"Sharpe Ratio: {self.portfolio.sharpe_ratio:.4f}\n"
        portfolio_info += f"Sortino Ratio: {self.portfolio.sortino_ratio:.4f}\n"
        
        self.portfolio_text.insert(tk.END, portfolio_info)

        # Dividend information (modified for better wrapping)
        dividend_info = "Dividend Information:\n\n"
        dividend_items = []
        for ticker, dividend in self.portfolio.total_dividends.items():
            dividend_items.append(f"{ticker}: ${dividend:.2f}")
        
        # Join dividend items with commas and spaces to encourage wrapping
        dividend_info += ", ".join(dividend_items)
        
        total_dividends = sum(self.portfolio.total_dividends.values())
        dividend_info += f"\n\nTotal Dividends: ${total_dividends:.2f}\n"
        
        self.dividend_text.insert(tk.END, dividend_info)

    def on_closing(self):
        self.cleanup()
        self.master.destroy()

    def cleanup(self):
        plt.close('all')  # Close all matplotlib figures
        self.fig.clear()
        plt.close(self.fig)
        self.canvas.get_tk_widget().destroy()

def main():
    config = load_config()
    initial_weights = config['portfolio_weights']
    initial_investment = config.get('initial_investment', 1000000)

    df = prepare_data('scraped_yahoo_finance_data.csv')
    returns = calculate_returns(df)

    root = tk.Tk()
    app = PortfolioGUI(root, returns, initial_weights, initial_investment)
    root.mainloop()

    # Ensure all matplotlib figures are closed
    plt.close('all')

if __name__ == "__main__":
    main()

