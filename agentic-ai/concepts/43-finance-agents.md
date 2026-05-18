# Finance Agents

## Detailed Explanation

Finance agents autonomously make financial decisions: trading (buy/sell securities), risk analysis (portfolio composition, value-at-risk), optimization (minimize risk for target return). Core challenges: real-time latency (milliseconds matter), accuracy critical (wrong decision costs money), compliance requirements (can't do prohibited trades), handling uncertainty (market volatility, incomplete info). Mechanisms: (1) real-time market data feed, (2) quantitative models (Black-Scholes, value-at-risk), (3) risk checks (position limits, max loss), (4) execution (place orders). Advantages: speed (milliseconds vs human hours), consistency (no emotion), 24/7 operation. Challenges: model risk (good backtest ≠ real profits), overfitting (learned patterns don't persist), regulatory compliance (auditing required). Best for: high-frequency trading (HFT), portfolio rebalancing, algorithmic execution, risk monitoring.

## Core Intuition

Imagine a trader who never sleeps, never gets emotional, follows rules perfectly. Agent analyzes market data, makes trading decisions, monitors positions, manages risk. Speed and discipline are advantages over humans.

## How It Works

Data feed → Analysis → Decision → Risk check → Execution → Monitoring:

1. **Market Data** — Real-time price feeds, order books
2. **Analysis** — Compute value, risk, opportunity
3. **Decision** — Buy/sell/hold signal
4. **Risk Check** — Validate against position limits, max loss
5. **Execution** — Place order if checks pass
6. **Monitoring** — Track position, adjust as needed

## Architecture / Trade-offs

**Autonomy Level:**
- **Advisory** — Agent suggests, human approves (safe, slow)
- **Delegated** — Agent executes, human reviews logs (fast, medium risk)
- **Autonomous** — Agent executes within bounds, no approval (fastest, high risk)

**Latency:**
- **Real-time** — Milliseconds (HFT, expensive infrastructure)
- **Intraday** — Seconds (regular trading)
- **Daily** — Minutes/hours (portfolio rebalancing)

**Risk Bounds:**
- **Strict** — Small position limits, max loss caps (safe, limits upside)
- **Moderate** — Balanced position/loss limits
- **Loose** — High limits (maximize returns, risky)

## Interview Q&A

**Q: What regulatory requirements apply to AI agents that provide financial recommendations?**
A: In the US: agents providing personalized investment advice may require registration as an Investment Adviser under the Investment Advisers Act. Automated trading systems are subject to SEC/CFTC regulations. FINRA rules apply to broker-dealer activities. Consumer financial products are subject to CFPB regulations. Key compliance requirements: suitability standards (recommendations must be appropriate for the customer), explainability requirements (customers can request explanations of automated decisions), and audit trail requirements. Consult securities law counsel before deploying agents that recommend specific investments.

**Q: How do you prevent a financial agent from making unauthorized transactions?**
A: Implement a strict authorization model: the agent can only initiate transactions explicitly authorized by the user for that specific transaction (not blanket authorization). Use a transaction confirmation step that requires human approval above a threshold amount. Implement transaction limits per day/week. Log all transaction attempts with full context. Use anomaly detection to flag unusual transaction patterns. Separate the agent's read access (for analysis) from write access (for execution)—the agent should need explicit authorization to move money.

**Q: What are the key risks of using an LLM for financial analysis and how do you mitigate them?**
A: Hallucination of financial figures: LLMs can invent plausible-sounding numbers. Mitigation: ground all numerical claims in retrieved data from authoritative sources, never generate financial numbers from parametric knowledge. Stale data: financial markets change rapidly. Mitigation: require data with timestamps, reject data older than defined freshness thresholds. Overconfidence: LLMs may present uncertain projections as confident. Mitigation: require explicit uncertainty quantification and include base rates for comparison.

**Q: How do you handle conflicts of interest in a financial agent?**
A: Document potential conflicts: if the agent is deployed by a bank, it may favor products offered by that bank. Disclose these conflicts to users. Implement controls: if recommending products, include comparative analysis with competitors rather than recommending only proprietary products. For fiduciary contexts (wealth management), the agent must demonstrably act in the client's interest—audit recommendation patterns to detect systematic bias toward higher-fee products. Have compliance review the agent's recommendation logic.

**Q: What latency requirements apply to financial agents and how do they affect design?**
A: High-frequency trading: microsecond latency—impossible with LLMs (100ms+ inference time). Algorithmic trading: millisecond latency—LLMs are too slow for execution but can be used for signal generation (executed separately). Risk management: second-to-minute latency—LLMs can be used for real-time risk alerts and narrative explanations. Customer-facing analysis: seconds acceptable—full LLM pipeline works. Design the system so time-critical execution uses deterministic rule-based systems, with LLMs handling analysis, explanation, and reporting.

**Q: How do you ensure a financial agent's recommendations are explainable for regulatory compliance?**
A: Regulators and customers have a right to explanation for automated financial decisions. Design for explainability from the start: structure the agent's reasoning as explicit steps (factor analysis, comparison to benchmarks, risk assessment). Log the full reasoning chain, not just the conclusion. For adverse actions (denied loan, flagged transaction), prepare standardized explanation formats that comply with adverse action notice requirements. Test explanations with compliance and legal teams before deployment.


## Best Practices

1. **Rigorous Backtesting** — Test on historical data before live
2. **Paper Trading** — Test with fake money before real capital
3. **Position Limits** — Hard caps on size and concentration
4. **Stop-Loss Orders** — Automatic exit if position goes bad
5. **Circuit Breakers** — Pause if market conditions unusual
6. **Real-Time Monitoring** — Watch P&L every second
7. **Regulatory Compliance** — Document all trades
8. **Drawdown Limits** — Stop if cumulative losses exceed threshold
9. **Diversification** — Don't bet everything on one position
10. **Fallback** — Manual override always available

## Common Pitfalls

**Pitfall 1: Overfitting to History**
Issue: Agent learns patterns from backtest that don't exist in reality.
Fix: Test on out-of-sample data. Use walk-forward validation.

**Pitfall 2: Ignoring Latency**
Issue: Agent makes decision that's correct by the time human sees it.
Fix: Account for execution latency. Plan for slippage.

**Pitfall 3: No Circuit Breaker**
Issue: Agent enters losing streak, keeps doubling down, catastrophic loss.
Fix: Hard stop-loss. Pause trading if losses exceed threshold.

**Pitfall 4: Ignoring Transaction Costs**
Issue: Agent trades frequently. Transaction costs erode profits.
Fix: Include commissions, slippage in backtests.

**Pitfall 5: Correlation Assumption**
Issue: Assume assets uncorrelated in crisis. In crisis, all correlate to 1.
Fix: Test in high-volatility periods. Stress test.

## Code Examples

### Example 1: Simple Trading Decision

```python
import numpy as np

class SimpleTrader:
    def __init__(self, position_limit=100, stop_loss_pct=0.05):
        self.position = 0
        self.position_limit = position_limit
        self.stop_loss_pct = stop_loss_pct
        self.entry_price = None
    
    def should_buy(self, current_price, sma_20, sma_50):
        '''Buy if price above moving averages (simple strategy).'''
        return current_price > sma_20 > sma_50
    
    def should_sell(self, current_price):
        '''Sell on stop-loss or profit target.'''
        if self.entry_price is None:
            return False
        loss_pct = (current_price - self.entry_price) / self.entry_price
        return loss_pct < -self.stop_loss_pct  # Stop loss hit
    
    def execute(self, current_price, sma_20, sma_50):
        '''Trade execution with checks.'''
        if self.position > 0 and self.should_sell(current_price):
            self.position = 0
            return 'SELL'
        
        if self.position == 0 and self.should_buy(current_price, sma_20, sma_50):
            if self.position < self.position_limit:
                self.position += 1
                self.entry_price = current_price
                return 'BUY'
        
        return 'HOLD'
```

### Example 2: Risk-Aware Portfolio Agent

```python
class PortfolioAgent:
    def __init__(self, portfolio_value=100000, max_drawdown=0.1):
        self.portfolio_value = portfolio_value
        self.positions = {}  # {symbol: amount}
        self.max_drawdown = max_drawdown
        self.peak_value = portfolio_value
    
    def compute_value(self, prices):
        '''Current portfolio value.'''
        value = sum(self.positions.get(sym, 0) * prices.get(sym, 0) for sym in prices)
        return value
    
    def check_drawdown(self, current_value):
        '''Check if drawdown exceeded.'''
        drawdown = (self.peak_value - current_value) / self.peak_value
        if drawdown > self.max_drawdown:
            return False  # Exceeded limit
        self.peak_value = max(self.peak_value, current_value)
        return True
    
    def rebalance(self, prices):
        '''Rebalance if weights drift.'''
        current_value = self.compute_value(prices)
        if not self.check_drawdown(current_value):
            print("Drawdown limit exceeded. Liquidating.")
            self.positions = {}  # Sell all
            return
        
        target_weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        for sym, target_weight in target_weights.items():
            target_value = current_value * target_weight
            current_pos_value = self.positions.get(sym, 0) * prices.get(sym, 0)
            # Rebalance if drift > 5%
            if abs(current_pos_value - target_value) / target_value > 0.05:
                self.positions[sym] = int(target_value / prices.get(sym, 1))
```

### Example 3: Risk Management

```python
class RiskManager:
    def __init__(self, max_position_size=50000, max_loss_per_trade=1000):
        self.max_position_size = max_position_size
        self.max_loss_per_trade = max_loss_per_trade
        self.open_positions = []
    
    def validate_trade(self, position_size, entry_price, stop_loss_price):
        '''Validate trade before execution.'''
        # Check position size
        if position_size > self.max_position_size:
            return False, "Position too large"
        
        # Check potential loss
        loss = (entry_price - stop_loss_price) * position_size
        if loss > self.max_loss_per_trade:
            return False, "Loss exceeds limit"
        
        return True, "OK"
    
    def monitor_positions(self, current_prices):
        '''Monitor all open positions.'''
        for pos in self.open_positions:
            current_price = current_prices.get(pos['symbol'])
            unrealized_loss = (pos['entry_price'] - current_price) * pos['size']
            if unrealized_loss > self.max_loss_per_trade:
                return 'CLOSE_POSITION', pos['symbol']
        
        return 'CONTINUE', None
```

## Related Concepts

- Real-Time Agents, Autonomous Agents, Risk Alignment, Error Recovery, Monitoring
