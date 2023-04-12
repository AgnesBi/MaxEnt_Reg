## MaxEnt
# Base objective: Maximize the probability of the observed distribution.
# Learning proceeds by looping through all the tableaux:
#   Step 1: Calculates the predicted probability distribution over the candidates based on the current weights;
#   Step 2: Compares it to the given probability distribution;
#   Step 3: Updates the weights
#       if stochastic Gradient Descent (SGD): by going in the direction towards the minimum of the objective function.
# Last updated: 04/11/2023

import numpy as np


class MaxEnt:
    """A general-purpose MaxEnt class.

    Objects:
        constraints (np.array[str]): Names of constraints.
        weights (np.array[float]): Violation profiles for the tableaux.

    Methods:
        sorted_by_weights: Returns sorted (constraint, weight) tuples.
        compute_probabilities: Returns the predicted probability distribution for any given violation profile.
    """

    def __init__(self, constraints, weights):
        self.const = constraints
        self.w = weights

    def sorted_by_weights(self):
        """Displays constraints from highest to lowest weights.

        Returns:
            list[tuple[str, float]]: a list of (constraint, weight) tuple sorted from highest to lowest weighted.
        """
        name_weight_pair = {}
        for cn, cw in zip(self.const, self.w.squeeze()):
            name_weight_pair[cn] = np.round(cw, 3)

        sorted_pairs = sorted(
            name_weight_pair.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_pairs

    def compute_probabilities(self, violations):
        """Predicts probabilities for a particular violation profile.

        Args:
            violations (np.array[np.array[int]]): the violation profiles of a list of tableaux.

        Returns:
            np.array[np.array[float]]: predicts probability for each candidate of given URs.
        """
        harmonies = np.dot(violations, self.w)
        exp = np.exp(-harmonies)
        marginalization = np.sum(exp, axis=0)
        return exp / marginalization

    def SGD_learn(self, candidates, violations, srs, *regs, iters=10000, eta=0.05):
        """Updates constraint weights using SGD.

        Args:
            candidates (np.array[np.array[str]]): candidates considered for each UR.
            violations (np.array[np.array[int]]): the violation profiles of a list of tableaux.
            srs (np.array[np.array[float]]): probability of a surface form observed for a given UR.
            *regs: a variable number of regularization terms. Can be None.
            iters (int): how many iterations to perform. Defaults to 10000.
            eta (float): learning rate. Defaults to 0.05.

        Returns:
            _type_: _description_
        """
        # Initialization: begins with the weights that are currently stored.
        w = self.w

        # Keeps a history of the change in weights.
        history = [w.squeeze()]

        # Passes through all the training data in each iteration,
        for _ in range(iters):
            # Generates a random array of indices of the shape (1, length |URs|).
            training_order = np.random.choice(
                a=range(len(violations)), size=len(violations), replace=False
            )

            dL_dw = 0

            # Loops through each tableau in the dataset by random order.
            for t in training_order:
                # Skips the tableaux without a winning candidate.
                if srs[t].count(None) == len(srs[t]):
                    continue

                # Violation profile of the winning candidate(s).
                winner_violations = np.zeros(len(self.const))
                for i, val in enumerate(srs[t]):
                    if val:
                        winner_violations = np.array([violations[t][i]]) * val

                # Computes probability for each candidate given the current weights.
                P = self.compute_probabilities(violations[t], w)

                # Computes the gradient.
                dL_dw += np.transpose(
                    winner_violations - (violations[t] * P).sum(axis=0, keepdims=True)
                )

            # Incorporates regularization if specified.
            if regs:
                for reg in regs:
                    dL_dw += reg

            # Updates the weights.
            w = w - eta * dL_dw

            # Replaces negative weights with 0.
            w[w < 0] = 0

            # Store history.
            history.append(w.squeeze())

        return w, history
