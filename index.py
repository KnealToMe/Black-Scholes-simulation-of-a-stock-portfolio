import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Black-Scholes formula for European call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Simulate stock price using Geometric Brownian Motion
def simulate_stock(S0, T, r, sigma, steps):
    dt = T / steps
    prices = [S0]
    for _ in range(steps):
        S = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        prices.append(S)
    return prices

# Portfolio setup
stocks = {
    "AAPL": {"S0": 175, "sigma": 0.25},
    "MSFT": {"S0": 330, "sigma": 0.22},
    "GOOGL": {"S0": 140, "sigma": 0.28}
}

options = {
    "AAPL": {"K": 180, "T": 1, "r": 0.05, "sigma": 0.25},
    "MSFT": {"K": 340, "T": 1, "r": 0.05, "sigma": 0.22},
    "GOOGL": {"K": 150, "T": 1, "r": 0.05, "sigma": 0.28}
}

# Number of shares held per stock
shares = {
    "AAPL": 10,
    "MSFT": 5,
    "GOOGL": 8
}

# Simulate and price
steps = 252
portfolio_value = 0
plt.figure(figsize=(10, 6))

for ticker, data in stocks.items():
    prices = simulate_stock(data["S0"], T=1, r=0.05, sigma=data["sigma"], steps=steps)
    final_price = prices[-1]
    equity_value = final_price * shares[ticker]
    plt.plot(prices, label=ticker)
    print(f"{ticker} final simulated price: ${final_price:.2f}")
    print(f"{ticker} equity value (shares × price): ${equity_value:.2f}")
    
    opt = options[ticker]
    call_price = black_scholes_call(final_price, opt["K"], opt["T"], opt["r"], opt["sigma"])
    print(f"{ticker} call option price: ${call_price:.2f}")
    
    portfolio_value += equity_value + call_price

plt.title("Black-Scholes Simulation of a Stock Portfolio")
plt.xlabel("Days")
plt.ylabel("Equity($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nTotal simulated portfolio value (stocks + options): ${portfolio_value:.2f}")
