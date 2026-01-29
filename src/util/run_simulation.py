import numpy as np
import os
import sys

# Set up paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGENTS_DIR = os.path.join(ROOT, "agents")
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sender import Sender
from receiver import Receiver
from game import SignalingGame

def run_simulation(
    T=20000,
    n=2,
    p_observe_state=0.2,
    seed=10,
    record=True
):
    sender = Sender(n_states=n, n_signals=n, seed=seed)
    receiver = Receiver(n_states=n, n_signals=n, n_actions=n, seed=seed + 1)
    game = SignalingGame(sender, receiver, p_observe_state, seed=seed + 2)

    for _ in range(T):
        game.step(record=record)

    return sender, receiver, game


if __name__ == "__main__":
    sender, receiver, game = run_simulation()
    print("Final success rate:", np.mean(game.success_history))
    print(f"Final expected payoff (mixed): {game.expected_payoff_mixed[-1]:.4f}")
    print(f"Final expected payoff (signal only): {game.expected_payoff_signal_only[-1]:.4f}")
    print(f"Final expected payoff (state+signal): {game.expected_payoff_state_signal[-1]:.4f}")