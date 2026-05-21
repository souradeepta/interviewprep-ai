"""
Auto-generated from 17-competitive-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Competitive Agents
# Objectives: Game-theoretic reasoning, minimax search, opponent modeling, adversarial optimization
# ======================================================================

import random
from collections import defaultdict
from math import log, sqrt
from typing import Dict, List, Tuple, Any

# Level 1: Basic Minimax Agent (Perfect Information Game)

class SimpleGameState:
    def __init__(self, value: int, is_maximizing: bool = True):
        self.value = value
        self.is_maximizing = is_maximizing
    
    def get_possible_moves(self) -> List[int]:
        """Return possible next values."""
        return [self.value + 1, self.value - 1, self.value * 2]
    
    def evaluate(self) -> int:
        """Score for maximizing player: higher is better."""
        return abs(self.value - 10)

class SimpleMinimaxAgent:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
    
    def minimax(self, state: SimpleGameState, depth: int) -> Tuple[int, int]:
        """Returns (best_score, best_move)."""
        
        if depth == 0:
            return (state.evaluate(), None)
        
        if state.is_maximizing:
            best_score = float('-inf')
            best_move = None
            for move in state.get_possible_moves():
                new_state = SimpleGameState(move, is_maximizing=False)
                score, _ = self.minimax(new_state, depth - 1)
                if score > best_score:
                    best_score = score
                    best_move = move
            return (best_score, best_move)
        else:
            best_score = float('inf')
            best_move = None
            for move in state.get_possible_moves():
                new_state = SimpleGameState(move, is_maximizing=True)
                score, _ = self.minimax(new_state, depth - 1)
                if score < best_score:
                    best_score = score
                    best_move = move
            return (best_score, best_move)
    
    def get_best_move(self, state: SimpleGameState) -> int:
        _, best_move = self.minimax(state, self.max_depth)
        return best_move

# Test Level 1
agent = SimpleMinimaxAgent(max_depth=3)
state = SimpleGameState(value=5, is_maximizing=True)
best_move = agent.get_best_move(state)
print(f'Level 1 - Minimax Agent:')
print(f'Current position: {state.value}')
print(f'Best move: {best_move}')
print(f'Evaluation of best move: {abs(best_move - 10)}')


# Level 2: Opponent Modeling with Adaptation

class OpponentModel:
    def __init__(self):
        self.move_history = []
        self.move_counts = defaultdict(int)
    
    def observe(self, move: int):
        """Update model based on observed opponent move."""
        self.move_history.append(move)
        self.move_counts[move] += 1
    
    def predict_next_move(self) -> int:
        """Predict opponent's most likely next move."""
        if not self.move_counts:
            return None
        return max(self.move_counts.items(), key=lambda x: x[1])[0]
    
    def get_distribution(self) -> Dict[int, float]:
        """Get probability distribution over opponent's moves."""
        total = sum(self.move_counts.values())
        if total == 0:
            return {}
        return {move: count / total for move, count in self.move_counts.items()}

class AdaptiveCompetitiveAgent:
    def __init__(self):
        self.opponent_model = OpponentModel()
        self.round_count = 0
    
    def choose_move(self, possible_moves: List[int]) -> int:
        """Choose move that counters opponent's likely move."""
        likely_opponent_move = self.opponent_model.predict_next_move()
        
        if likely_opponent_move is None:
            # No history yet; play random
            return random.choice(possible_moves)
        
        # Choose move that maximizes my payoff against their likely move
        best_move = None
        best_payoff = float('-inf')
        
        for move in possible_moves:
            payoff = move - likely_opponent_move  # Simple payoff function
            if payoff > best_payoff:
                best_payoff = payoff
                best_move = move
        
        return best_move
    
    def play_round(self, possible_moves: List[int], opponent_last_move: int = None) -> int:
        """Play one round: observe opponent, choose move."""
        if opponent_last_move is not None:
            self.opponent_model.observe(opponent_last_move)
        
        my_move = self.choose_move(possible_moves)
        self.round_count += 1
        
        return my_move

# Test Level 2: Multi-round competition
agent = AdaptiveCompetitiveAgent()
opponent_sequence = [5, 5, 5, 7, 7, 7]  # Opponent tends to play 5, then 7
possible_moves = [1, 5, 10]

print('\nLevel 2 - Adaptive Agent with Opponent Modeling:')
print(f'Opponent sequence: {opponent_sequence}\n')

for round_num, opp_move in enumerate(opponent_sequence, 1):
    my_move = agent.play_round(possible_moves, opp_move if round_num > 1 else None)
    model = agent.opponent_model.get_distribution()
    print(f'Round {round_num}: My={my_move}, Opponent={opp_move}, Model={dict(model)}')

print(f'\nFinal prediction: {agent.opponent_model.predict_next_move()}')


# Example 1: Two-Player Zero-Sum Game (Rock-Paper-Scissors with Payoff)

class RockPaperScissorsGame:
    MOVES = ['rock', 'paper', 'scissors']
    
    @staticmethod
    def payoff(my_move: str, opponent_move: str) -> int:
        """Return payoff for me (positive=win, zero=tie, negative=lose)."""
        if my_move == opponent_move:
            return 0
        if (my_move == 'rock' and opponent_move == 'scissors') or \
           (my_move == 'paper' and opponent_move == 'rock') or \
           (my_move == 'scissors' and opponent_move == 'paper'):
            return 1
        return -1

class RPSCompetitiveAgent:
    def __init__(self):
        self.opponent_model = defaultdict(int)
        self.my_score = 0
        self.opponent_score = 0
    
    def counter_move(self, likely_opponent_move: str) -> str:
        """Choose move that beats opponent's likely move."""
        if likely_opponent_move == 'rock':
            return 'paper'
        elif likely_opponent_move == 'paper':
            return 'scissors'
        else:
            return 'rock'
    
    def predict_opponent(self) -> str:
        """Predict opponent's next move."""
        if not self.opponent_model:
            return random.choice(RockPaperScissorsGame.MOVES)
        return max(self.opponent_model.items(), key=lambda x: x[1])[0]
    
    def play(self, opponent_move: str) -> str:
        """Play one round."""
        # Update opponent model
        self.opponent_model[opponent_move] += 1
        
        # Predict and counter
        predicted = self.predict_opponent()
        my_move = self.counter_move(predicted)
        
        # Calculate payoff
        payoff = RockPaperScissorsGame.payoff(my_move, opponent_move)
        self.my_score += payoff
        
        return my_move

# Simulate game
print('\nExample 1 - Rock-Paper-Scissors Competitive Game:')
agent = RPSCompetitiveAgent()
opponent_moves = ['rock', 'rock', 'rock', 'paper', 'paper', 'scissors']

for round_num, opp_move in enumerate(opponent_moves, 1):
    my_move = agent.play(opp_move)
    payoff = RockPaperScissorsGame.payoff(my_move, opp_move)
    print(f'Round {round_num}: Me={my_move:8} vs Opponent={opp_move:8} → {payoff:+d} (Score: {agent.my_score})')

print(f'\nFinal Score: {agent.my_score}')


# Example 2: Bidding Competition (Auction / Resource Allocation)

class BiddingCompetitiveAgent:
    def __init__(self, budget: int = 100, name: str = 'Agent'):
        self.budget = budget
        self.spent = 0
        self.name = name
        self.opponent_bids = []
        self.items_won = 0
    
    def estimate_opponent_bid(self) -> float:
        """Estimate opponent's likely bid based on history."""
        if not self.opponent_bids:
            return 10.0  # Default estimate
        return sum(self.opponent_bids) / len(self.opponent_bids)
    
    def decide_bid(self, item_value: int, opponent_historical_bids: List[int]) -> int:
        """Decide bid for item."""
        # Update opponent model
        self.opponent_bids.extend(opponent_historical_bids)
        
        avg_opponent_bid = self.estimate_opponent_bid()
        
        # Strategy: bid slightly higher than opponent's average
        # But don't overpay (stay within budget)
        bid = min(
            int(avg_opponent_bid * 1.1) + 1,  # Beat opponent by 10%
            self.budget - self.spent  # Stay within budget
        )
        
        return max(1, bid)
    
    def bid_round(self, item_value: int, my_bid: int, opponent_bid: int) -> bool:
        """Execute one round of bidding. Return True if won."""
        if my_bid > opponent_bid:
            self.spent += my_bid
            self.items_won += 1
            return True
        return False

# Simulate bidding
print('\nExample 2 - Competitive Bidding Agent:')
agent = BiddingCompetitiveAgent(budget=100, name='MyAgent')
opponent_bids = [5, 5, 10, 15, 20, 25]
item_values = [10, 15, 20, 25, 30, 35]

my_bids = []
for round_num, (item_val, opp_bid) in enumerate(zip(item_values, opponent_bids), 1):
    my_bid = agent.decide_bid(item_val, [opp_bid])
    won = agent.bid_round(item_val, my_bid, opp_bid)
    print(f'Item {round_num} (value={item_val}): Me={my_bid:3}, Opponent={opp_bid:3} → {"WON" if won else "LOST":4} (Spent: {agent.spent}/100)')

print(f'\nFinal: Won {agent.items_won} items, spent {agent.spent}/{agent.budget}')


# Example 3: Monte Carlo Tree Search (MCTS) for Competitive Game

class MCTSNode:
    def __init__(self, state: int, parent=None, is_maximizing: bool = True):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.is_maximizing = is_maximizing
    
    def ucb_score(self, exploration: float = 1.41) -> float:
        """Upper Confidence Bound: balance exploitation vs exploration."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration_term = exploration * sqrt(log(self.parent.visits) / self.visits)
        return exploitation + exploration_term
    
    def best_child(self) -> 'MCTSNode':
        """Select child with highest UCB score."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb_score())

class MCTSCompetitiveAgent:
    def __init__(self, num_simulations: int = 100):
        self.num_simulations = num_simulations
    
    def search(self, initial_state: int) -> int:
        """Run MCTS. Return best move."""
        root = MCTSNode(initial_state, parent=None, is_maximizing=True)
        
        for sim in range(self.num_simulations):
            node = root
            path = [node]
            
            # Selection: traverse using UCB
            while node.children:
                node = node.best_child()
                path.append(node)
            
            # Expansion: add one child
            if node.visits > 0:
                next_is_max = not node.is_maximizing
                for move in [node.state + 1, node.state - 1]:
                    child = MCTSNode(move, parent=node, is_maximizing=next_is_max)
                    node.children.append(child)
                
                if node.children:
                    node = node.children[0]
                    path.append(node)
            
            # Simulation: random playout
            value = abs(node.state - 10)  # Simple evaluation
            
            # Backprop
            for n in path:
                n.visits += 1
                n.value += value
        
        # Return best child by visit count
        best = max(root.children, key=lambda c: c.visits)
        return best.state

# Test MCTS
print('\nExample 3 - MCTS Competitive Agent:')
mcts_agent = MCTSCompetitiveAgent(num_simulations=100)
best_move = mcts_agent.search(initial_state=5)
print(f'Starting state: 5')
print(f'Best move (from MCTS): {best_move}')
print(f'Evaluation: {abs(best_move - 10)}')
print(f'MCTS explored {100} simulations, found optimal/near-optimal move')


# ======================================================================
# ## Key Takeaways
# **Competitive Reasoning:**
# 1. Model opponent: observe, predict, adapt
# 2. Lookahead: think 3-5 moves ahead (minimax)
# ======================================================================
