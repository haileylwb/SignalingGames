import numpy as np

class SignalingGame:
    def __init__(self, sender, receiver, p_observe_state: float, seed=None):
        self.sender = sender
        self.receiver = receiver
        self.p_observe_state = p_observe_state
        self.rng = np.random.default_rng(seed)
        self.success_history = []

        self.expected_payoff_signal_only = []
        self.expected_payoff_state_signal = []
        self.expected_payoff_mixed = []


    def _expected_payoff_signal_only(self, sender_policy, receiver_signal_policy):
        n_states, n_signals = sender_policy.shape
        _, n_actions = receiver_signal_policy.shape

        payoff = 0.0
        for w in range(n_states):
            for m in range(n_signals):
                for a in range(n_actions):
                    if a == w:
                        payoff += (
                            (1 / n_states)
                            * sender_policy[w, m]
                            * receiver_signal_policy[m, a]
                        )
        return payoff

    def _pooled_receiver_policy(self, state, signal):
        """
        Compute pooled action probabilities for a specific state-signal pair.
        Uses ball pooling (BP): sum the weights then normalize.
        """
        weights = (self.receiver.state_action_weights[state] + 
                   self.receiver.signal_action_weights[signal])
        return weights / weights.sum()

    
    def _expected_payoff_state_signal(self, sender_policy):
        """
        Compute expected payoff when receiver observes both state and signal.
        Must compute pooled probabilities BEFORE normalizing (ball pooling).
        """
        n_states, n_signals = sender_policy.shape

        payoff = 0.0
        for w in range(n_states):
            for m in range(n_signals):
                # Get pooled action probabilities for this state-signal pair
                pooled_probs = self._pooled_receiver_policy(w, m)
                
                # Probability of correct action under pooling
                p_correct = pooled_probs[w]
                
                payoff += (1 / n_states) * sender_policy[w, m] * p_correct
        
        return payoff


    def step(self, record=False):
        state = self.rng.integers(self.sender.n_states)

        signal = self.sender.send(state, record=record)
        observe_state = self.rng.random() < self.p_observe_state
        world_state_obs = state if observe_state else None
        action = self.receiver.choose_action(signal, world_state_obs)

        # Realized payoff
        success = (action == state)
        self.success_history.append(int(success))

        # Learning updates
        self.sender.update(state, signal, success)
        self.receiver.update(signal, action, success, world_state_obs)

        # Expected Payoff
        sender_policy = (self.sender.state_signal_weights / 
                        self.sender.state_signal_weights.sum(axis=1, keepdims=True))
        receiver_signal_policy = (self.receiver.signal_action_weights / 
                                 self.receiver.signal_action_weights.sum(axis=1, keepdims=True))
        
        payoff_signal_only = self._expected_payoff_signal_only(sender_policy, receiver_signal_policy)
        payoff_state_signal = self._expected_payoff_state_signal(sender_policy)
        payoff_mixed = ((1 - self.p_observe_state) * payoff_signal_only + 
                       self.p_observe_state * payoff_state_signal)

        self.expected_payoff_signal_only.append(payoff_signal_only)
        self.expected_payoff_state_signal.append(payoff_state_signal)
        self.expected_payoff_mixed.append(payoff_mixed)

        if record:
            self.receiver.record()

        return success