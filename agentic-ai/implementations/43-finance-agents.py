"""
Auto-generated from 43-finance-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Finance Agents
# Objectives: Core patterns, implementation, optimization
# ======================================================================

class Trader:
    def __init__(self):
        self.position = 0
        self.balance = 100000
    def buy(self, price, shares):
        cost = price * shares
        if self.balance >= cost:
            self.position += shares
            self.balance -= cost
    def sell(self, price, shares):
        self.position -= shares
        self.balance += price * shares

trader = Trader()
trader.buy(100, 10)
print(f'Position: {trader.position}, Balance: {trader.balance}')


class RiskManager:
    def __init__(self, max_loss=1000):
        self.max_loss = max_loss
    def validate_trade(self, entry, stop_loss, size):
        max_loss_trade = (entry - stop_loss) * size
        return max_loss_trade <= self.max_loss

rm = RiskManager()
valid = rm.validate_trade(100, 95, 100)
print(f'Trade valid: {valid}')


import numpy as np

class PortfolioOptimizer:
    def rebalance(self, current_weights, target_weights, prices):
        drifts = {s: abs(c - t) for s, (c, t) in zip(range(len(current_weights)), zip(current_weights, target_weights))}
        if max(drifts.values()) > 0.05:
            return target_weights
        return current_weights

opt = PortfolioOptimizer()
current = [0.5, 0.3, 0.2]
target = [0.4, 0.3, 0.3]
result = opt.rebalance(current, target, [100, 100, 100])
print(f'Rebalanced: {result}')

