import numpy as np

class SignalingGame:
    def __init__(self, sender, receiver, p_observe_state: float, seed=None):
        self.sender = sender
        self.receiver = receiver
        self.p_observe_state = p_observe_state
        self.rng = np.random.default_rng(seed)
        self.success_history = []

  
    def step(self, record=False):
        state = self.rng.integers(self.sender.n_states)
        signal = self.sender.send(state, record=record)
        observe_state = self.rng.random() < self.p_observe_state
        world_state_obs = state if observe_state else None
        action = self.receiver.choose_action(signal, world_state_obs)

        # Payoff
        success = (action == state)
        self.success_history.append(int(success))

        # Learning
        self.sender.update(state, signal, success)
        self.receiver.update(signal, action, success, world_state_obs)

        if record:
            self.receiver.record()

        return success
