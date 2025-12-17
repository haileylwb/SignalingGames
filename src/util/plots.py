import numpy as np
import matplotlib.pyplot as plt

def plot_success(game, window=500):
    success = np.array(game.success_history)

    plt.figure()
    plt.plot(np.cumsum(success) / np.arange(1, len(success) + 1))
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative success rate")
    plt.title("Cumulative Success")
    plt.show()

    if window is not None:
        rolling = np.convolve(success, np.ones(window) / window, mode="valid")
        plt.figure()
        plt.plot(rolling)
        plt.xlabel("Timestep")
        plt.ylabel("Rolling success rate")
        plt.title(f"Rolling Success (window={window})")
        plt.show()


def plot_sender(sender, state=0):
    history = np.array(sender.history)

    plt.figure()
    for m in range(history.shape[2]):
        plt.plot(history[:, state, m], label=f"m{m}")
    plt.xlabel("Timestep")
    plt.ylabel("P(m | w)")
    plt.title(f"Sender policy for state w{state}")
    plt.legend()
    plt.show()


def plot_receiver_signal(receiver, signal=0):
    history = np.array(receiver.signal_action_history)

    plt.figure()
    for a in range(history.shape[2]):
        plt.plot(history[:, signal, a], label=f"a{a}")
    plt.xlabel("Timestep")
    plt.ylabel("P(a | m)")
    plt.title(f"Receiver policy for signal m{signal}")
    plt.legend()
    plt.show()


def plot_realized_vs_expected(game, window=500):
    realized = np.array(game.success_history)
    expected = np.array(game.expected_payoff_mixed)

    plt.figure()
    plt.plot(expected, label="Expected payoff")
    plt.xlabel("Timestep")
    plt.ylabel("Payoff")
    plt.title("Expected Payoff Over Time")
    plt.legend()
    plt.show()

    if window is not None:
        rolling_realized = np.convolve(
            realized, np.ones(window) / window, mode="valid"
        )

        plt.figure()
        plt.plot(rolling_realized, label="Realized payoff (rolling)")
        plt.plot(expected[window - 1:], label="Expected payoff")
        plt.xlabel("Timestep")
        plt.ylabel("Payoff")
        plt.title(f"Realized vs Expected Payoff (window={window})")
        plt.legend()
        plt.show()


def plot_expected_components(game):
    plt.figure()
    plt.plot(game.expected_payoff_signal_only, label="Signal only")
    plt.plot(game.expected_payoff_state_signal, label="State + signal")
    plt.plot(game.expected_payoff_mixed, label="Mixed")
    plt.xlabel("Timestep")
    plt.ylabel("Expected payoff")
    plt.title("Expected Payoff Components")
    plt.legend()
    plt.show()


def plot_receiver_state(receiver, state=0):
    history = np.array(receiver.state_action_history)

    plt.figure()
    for a in range(history.shape[2]):
        plt.plot(history[:, state, a], label=f"a{a}")
    plt.xlabel("Timestep")
    plt.ylabel("P(a | w)")
    plt.title(f"Receiver state policy for w{state}")
    plt.legend()
    plt.show()
