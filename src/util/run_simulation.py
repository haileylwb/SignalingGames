import numpy as np
from sender import Sender
from receiver import Receiver
from game import SignalingGame

def run_simulation(
    T=20000,
    n=2,
    p_observe_state=0.3,
    seed=0,
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
