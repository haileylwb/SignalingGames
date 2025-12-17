import numpy as np

class Receiver:
    def __init__(self, n_states: int, n_signals: int, n_actions: int, seed=None):
        self.n_states = n_states
        self.n_signals = n_signals
        self.n_actions = n_actions

        self.rng = np.random.default_rng(seed)
       
        self.signal_action_weights = np.ones((n_signals, n_actions), dtype=float)   # QR(a | m)      
        self.state_action_weights = np.ones((n_states, n_actions), dtype=float)     # QR(a | w)

        self.latest_signal_recommendation = None
        self.latest_state_recommendation = None

        self.signal_action_history = []
        self.state_action_history = []


    def _sample_from_weights(self, weights):
        probs = weights / weights.sum()
        return self.rng.choice(self.n_actions, p=probs)

  
    def choose_action(self, signal: int, world_state: int | None):
        if world_state is not None:
            self.latest_signal_recommendation = self._sample_from_weights(
                self.signal_action_weights[signal]
            )
            self.latest_state_recommendation = self._sample_from_weights(
                self.state_action_weights[world_state]
            )

            pooled_weights = (
                self.signal_action_weights[signal]
                + self.state_action_weights[world_state]
            )
            action = self._sample_from_weights(pooled_weights)

        else:
            self.latest_signal_recommendation = None
            self.latest_state_recommendation = None
            action = self._sample_from_weights(
                self.signal_action_weights[signal]
            )

        return action


    def update(self, signal: int, action: int, success: bool, world_state: int | None):
        if not success:
            return

        if world_state is None:
            self.signal_action_weights[signal, action] += 1
        else:
            if action == self.latest_signal_recommendation:
                self.signal_action_weights[signal, action] += 1
            if action == self.latest_state_recommendation:
                self.state_action_weights[world_state, action] += 1


    def record(self):
        signal_policy = (
            self.signal_action_weights
            / self.signal_action_weights.sum(axis=1, keepdims=True)
        )
        state_policy = (
            self.state_action_weights
            / self.state_action_weights.sum(axis=1, keepdims=True)
        )

        self.signal_action_history.append(signal_policy.copy())
        self.state_action_history.append(state_policy.copy())
