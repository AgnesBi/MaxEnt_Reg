# Heavily based on the Colab notebooks provided by Adam and Chris.
# Last updated: 02/15/2023

## Stochastic gradient descent (SGD)
# Learning proceeds by looping through all the tableaux:
# Step 1: Calculates the predicted probability distribution over the candidates based on the current weights;
# Step 2: Compares it to the given probability distribution
# Step 3: Updates the weights by going in the direction towards the minimum of the objective function.

## Option to add additional, theoretically motivated, built-in bias to the objective function.
# Maximizing the likelihood of the observed data
# +
# Maximize the distance between Markedness and Faithfulness constraints

import csv
import numpy as np
import matplotlib.pyplot as plt


def compute_probabilities(vio: list[list[int]], weight: list[float]) -> list[float]:
    harmonies = np.dot(vio, weight)
    exp = np.exp(-harmonies)
    marginalization = np.sum(exp, axis=0)
    return exp / marginalization


def learn_stochastic(
    constraints: list[str],
    violations: list[list[str]],
    wIdx: list[int],
    eta: float = 0.05,
    iters: int = 10000,
    mu: float = 20,
    sigma: float = 1,
    reg: bool = True,
):
    """Learning via SGD.

    Args:
        constraints: a list of pre-set constraints.
        violations: violation profiles for the tableaux.
        wIdx: index of the winning candidate within each tableau.
        eta: how large the updates should be on each iteration. Defaults to 0.05.
        iters: how many iterations to perform. Defaults to 10000.
        mu: _description_. Defaults to 20.
        sigma: _description_. Defaults to 1.
        reg: whether regularization is added. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Initialization: begins with some random weights.
    w = np.random.uniform(0, 10, (len(constraints), 1))

    # Keeps a history of the change in weights.
    history = [w.squeeze()]

    # Each iteration passes through all the data.
    for _ in range(iters):
        # Generates a random array of indices of the shape (1, length |URs|).
        training_order = np.random.choice(
            a=range(len(violations)), size=len(violations), replace=False
        )

        dL_dw = 0

        # Loops through each tableau in the dataset by random order.
        for t in training_order:
            # Skips the tableaux without a winning candidate.
            if wIdx[t] == None:
                continue

            # Violation profile of the winning candidate.
            winner_violations = np.array([violations[t][wIdx[t]]])

            # Computes probability for each candidate given the current weights.
            P = compute_probabilities(violations[t], w)

            # Computes the gradient.
            dL_dw += np.transpose(
                winner_violations - (violations[t] * P).sum(axis=0, keepdims=True)
            )

        if reg:
            # Adds the gradient of the sum difference prior.
            total_w_fc = 0
            total_w_mc = 0

            for i, con in enumerate(constraints):
                if con.startswith("*"):
                    total_w_mc += w[i][0]
                else:
                    total_w_fc += w[i][0]

            dP_dw = np.zeros(w.shape)
            m_a = np.full(w.shape, mu)
            s_a = np.full(w.shape, sigma)
            total_w_fc_a = np.full(w.shape, total_w_fc)
            total_w_mc_a = np.full(w.shape, total_w_mc)

            for i, con in enumerate(constraints):
                # For Markedness constraints:
                if con.startswith("*"):
                    dP_dw[i] = (total_w_mc_a[i] - total_w_fc_a[i] - m_a[i]) / s_a[i] ** 2
                # For Faithfulness constraints:
                else:
                    dP_dw[i] = (total_w_fc_a[i] - total_w_mc_a[i] + m_a[i]) / s_a[i] ** 2

            dL_dw += dP_dw

        # Updates the weights.
        w = w - eta * dL_dw

        # Replaces negative weights with 0.
        w[w < 0] = 0

        # Store history.
        history.append(w.squeeze())

    return w, history


def sorted_learned_weights(constraint, weight) -> dict[str, float]:
    learned = {}
    for cn, cw in zip(constraint, weight.squeeze()):
        learned[cn] = np.round(cw, 3)

    learned = sorted(learned.items(), key=lambda x: x[1], reverse=True)

    for pair in learned:
        print(f"{pair[0]:<20}: {pair[1]:<10}")

    return dict(learned)


def predict_prob(vio, cand):
    with open("predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["UR/Candidate", "Probability"])
        for t in range(len(vio)):
            P = compute_probabilities(vio[t], w).squeeze()
            writer.writerow([urs[t]])

            for c in range(len(cand[t])):
                writer.writerow([cand[t][c], f"{round(P[c]*100, 1)}%"])


# /* --------------------------------------------------------------------------- */
# /*                 Applies to a toy dataset                                    */
# /* --------------------------------------------------------------------------- */
if __name__ == "__main__":
    from OTSoft_file_reader import get_info

    constraints, urs, candidates, violations, wIdx = get_info(
        "HayesPseudoKorean-RichBase.txt"
    )

    w, history = learn_stochastic(constraints, violations, wIdx)

    # Learning results
    print("Constraint weights after learning:")
    sorted_learned_weights(constraints, w)
    print()

    # print("Probability of the candidates in each tableau after learning:")
    # for t in range(len(violations)):
    #     P = compute_probabilities(violations[t], w).squeeze()

    #     print("=" * 25)
    #     print(f"{urs[t]:<10} Prob")

    #     for c in range(len(candidates[t])):
    #         print(f"{candidates[t][c]:<10} {round(P[c]*100, 1)}%")
    #     print()

    predict_prob(violations, candidates)

    # Plots learning process.
    plt.plot(range(len(history)), history)
    plt.legend(constraints)
    plt.show()
