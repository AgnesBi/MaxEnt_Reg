## MaxEnt
# Base objective: Maximize the probability of the observed distribution.
# Learning proceeds by looping through all the tableaux:
#   Step 1: Calculates the predicted probability distribution over the candidates based on the current weights;
#   Step 2: Compares it to the given probability distribution;
#   Step 3: Updates the weights
#       if stochastic Gradient Descent (SGD): by going in the direction towards the minimum of the objective function.
#
# Last updated: 04/17/2023

import numpy as np
import random


# ==================== Helper functions =========================================
#
def compute_probabilities(weights, violations):
    """Predicts probabilities for a particular violation profile.

    Args:
        weights (list[float]): Constraint weights.
        violations (list[np.ndarray[int]]): the violation profiles of a list of tableaux.

    Returns:
        list[np.array[float]]: predicts probability for each candidate of given URs.
    """
    predictions = []

    for vio in violations:
        harmonies = np.dot(vio, weights)
        exp = np.exp(-harmonies)
        marginalization = np.sum(exp, axis=0)
        prob = exp / marginalization
        predictions.append(prob)

    return predictions


def get_weighted_winner_violations(observed_prob, violations):
    """_summary_

    Args:
        observed_prob (list[np.ndarray[float]]): probability of candidates observed for the given URs.
        violations (list[np.ndarray[int]]): the violation profiles of a list of tableaux.
    Returns:
        list[np.ndarray[float]]: the (weighted) violation profile of the winner candidates.
    """
    if len(observed_prob) != len(violations):
        raise ValueError(
            "Input error: the length of the two input lists should be the same."
        )

    rst = []
    for i in range(len(observed_prob)):
        weighted_tab = observed_prob[i].T * violations[i]
        # Removes the rows with only 0's (i.e., loser candidates).
        rst.append(weighted_tab[~np.all(weighted_tab == 0, axis=1)])

    return rst


#
# ==================== MaxEnt class =========================================
#
class MaxEnt:
    """A general-purpose MaxEnt class.

    Objects:
        constraints (list[str]): Names of constraints.
        weights (list[float]): Corresponding weight for each constraint.

    Methods:
        sorted_by_weights: Returns sorted (constraint, weight) tuples.
        SGD_learn:
    """

    def __init__(self, constraints, weights):
        self._cns = constraints
        self._cws = weights

    # TODO: figure out how we want the string representation to be.
    def __repr__(self):
        pass

    @property
    def cns(self):
        """Accesses constraint names."""
        return self._cns

    @property
    def cws(self):
        """Accesses constraint weights."""
        return self._cws

    def sorted_by_weights(self):
        """Sorts constraints from the highest- to lowest-weighted.

        Returns:
            list[tuple[str, float]]: a list of (constraint, weight) tuple sorted from highest to lowest weighted.
        """
        return sorted(zip(self.cns, self.cws), key=lambda x: x[1], reverse=True)

    def SGD_learn(
        self, violations, observed_prob, batch_size=1, iters=10000, eta=0.05, *regs
    ):
        """Updates constraint weights using SGD, with the option of Mini-batch SGD.

        Args:
            violations (list[np.ndarray[int]]): the violation profiles of a list of tableaux.
            observed_prob (list[np.ndarray[float]]): probability of candidates observed for the given URs.
            batch_size (int): batch size. Defaults to 1.
            iters (int): how many iterations to perform. Defaults to 10000.
            eta (float): learning rate. Defaults to 0.05.
            *regs: a variable number of regularization terms. Passed in as a tuple. Can be None.

        Returns:
            _type_: _description_
        """
        # Initialization: begins with the weights that are currently stored.
        weights = self.cws

        # Keeps a history of the change in weights.
        history = [weights.squeeze()]

        # Calculates the total number of tableaux in the dataset.
        tableaux_count = len(violations)

        # Sets the number of iterations.
        for _ in range(iters):
            # Generates a list of random indices of batch_size.
            training_set = random.sample(range(tableaux_count), batch_size)

            # Tracks gradient.
            dL_dw = 0

            for i in training_set:
                winner_indication_col = observed_prob[i]
                vio = violations[i]

                # Skips the tableaux without a winning candidate.
                if not winner_indication_col.any():
                    continue

                # Extracts the violation profile of the winning candidate(s).
                # Note there might be more than one winning candidate in the case of variation.
                winner_violations = get_weighted_winner_violations(
                    winner_indication_col, vio
                )

                # Computes probability for each candidate given the current weights.
                P = compute_probabilities(weights, vio)

                # Computes the gradient.
                dL_dw += np.transpose(
                    winner_violations - (vio * P).sum(axis=0, keepdims=True)
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
