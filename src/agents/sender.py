import numpy as np

class Sender:
    def __init__(self, n_states: int, n_signals: int, seed=None):
        self.n_states = n_states
        self.n_signals = n_signals
        self.rng = np.random.default_rng(seed)

        self.state_signal_weights = np.ones(
            (n_states, n_signals), dtype=float
        )
      
        self.history = []


    def _sample_from_weights(self, weights):
        probs = weights / weights.sum()
        return self.rng.choice(self.n_signals, p=probs)

  
    def send(self, state: int, record: bool = False) -> int:
        signal = self._sample_from_weights(
            self.state_signal_weights[state]
        )
        if record:
            self.record()
        return signal


    def update(self, state: int, signal: int, success: bool):
        if success:
            self.state_signal_weights[state, signal] += 1

  
    def record(self):
        policy = (
            self.state_signal_weights
            / self.state_signal_weights.sum(axis=1, keepdims=True)
        )
        self.history.append(policy.copy())
