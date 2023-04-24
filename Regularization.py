## Type hint and labelling
from typing import Union

## Matrix manipulation
import numpy as np


""" =================================================================================== 
                    REGULARIZATION CLASS DEFINITIONS
=================================================================================== """


class Grad:
    """Baseline Gradient Object. This is a subclass that will be used in superclasses
    in order to generate the gradient for different distributions. It takes in two
    arguments:

    (1) ``cns``: np.ndarray -- the constraint names
    (2) ``grs``: dict -- the group assignments for the constraints and their mu and sigma

    These base objects will then be used by the superclasses to generate the sums
    needed in each gradient calculation. This object stores the following
    instance variables:

    (A) ``cg``: dict
    (B) ``ms``: np.ndarray -- m [groups] x 2 [mu, sigma]
    """

    def __init__(self, cns, grs):
        ## Store the constraint names
        self._cns = cns

        ##
        self._cg = {g: i for g, (i, _, _) in grs.values()}
        self._ms = np.array([[m, s] for _, m, s in grs.values()])

    @property
    def cns(self):
        return self._cns

    @property
    def grs(self):
        return self._grs


class NrmGrad:
    def __init__(self, cns, grs):
        pass


class SumGrad:
    def __init__(self):
        pass


class Regularization:

    """==================== GRADIENTS ========================================="""

    """ Static methods for computing the gradient given a set of weights.
    Unless specified by the paramemter `type`, each method expects there 
    only two be two constraint types: markedness constraints ("M") and 
    faithfulness constraints ("F")
    """

    @staticmethod
    def grad_uni():
        """Returns the gradient given a flat prior, or in other words, 0"""
        return 0

    @staticmethod
    def grad_nrm(
        weights: list[float],
        names: list[str],
        mu: Union[float, dict[str, float]],
        sigma: Union[float, dict[str, float]],
    ):
        """Returns the gradient of a prior towards a target weight.
        Generality for both individual types -- e.g. M > F -- versus
        over all weights -- e.g. M = F = mu
        """

        ## Retrieve the type of the mu and sigma
        tMu = type(mu)
        tSigma = type(sigma)

        ## Check that the mu and sigma are of the same type
        assert (
            tMu == tSigma
        ), "Type mismatch: type of mu and sigma are {tMu} and {tSigma}"

        ## Initialize the weight gradient
        wGrad = np.zeros(weights.shape)

        ## If mu is a float, calculate the same prior over all weights
        if tMu == float:
            wGrad = (weights - mu) / (sigma**2)

        ## Otherwise, if mu is a dictionary, calculate the prior over
        ## each type. Assume only markedness and faithfulness constraints
        elif tMu == dict:
            ## Get the indices of the markedness and faithfulness constraints
            mIdx = Regularization.get_mIdx(names)
            fIdx = ~mIdx

            ## Update the gradients of the weights
            wGrad[mIdx] = (weights[mIdx] - mu["M"]) / (sigma["M"] ** 2)
            wGrad[fIdx] = (weights[fIdx] - mu["F"]) / (sigma["F"] ** 2)

        ## Return the gradient
        return wGrad

    @staticmethod
    def grad_sum(
        weights: list[float],
        names: list[str],
        mu: float,
        sigma: float,
    ):
        """Returns the gradient of a prior towards a target difference
        over constraint types -- e.g. markedness.sum() > faithfulness.sum()
        """

        ## Retrieve the type of the mu and sigma
        tMu = type(mu)
        tSigma = type(sigma)

        ## Under this regularization, we expect only a single mu and sigma
        assert (
            tMu == float and tSigma == float
        ), f"Got type {tMu} and {tSigma} instead of float"

        ## Check to make sure the constraint weights and shapes are the same shape
        assert (
            weights.shape == names.shape
        ), f"Weights {weights.shape} and names {names.shape} not of equal shape"

        ## Get the indices of the markedness and faithfulness constraints
        mIdx = Regularization.get_mIdx(names)
        fIdx = ~mIdx

        ## Sum the constraint weights of the markedness vs faithfulness constraints
        mSummed = weights[mIdx].sum()
        fSummed = weights[fIdx].sum()

        ## Calculate the gradient of the markedness vs faithfulness constraints
        mGrad = (mSummed - fSummed - mu) / (sigma**2)
        fGrad = (fSummed - mSummed + mu) / (sigma**2)

        ## Update the gradients of the weights
        wGrad = np.zeros(weights.shape)
        wGrad[mIdx] += mGrad
        wGrad[fIdx] += fGrad

        ## Return the gradient
        return wGrad

    @staticmethod
    def grad_ind(
        weights: list[float],
        names: list[str],
        types: dict[str, str],
        mu: float,
        sigma: float,
    ):
        """Returns the gradient of a prior towards a target difference
        between the markedness constraints and the faithfulness
        constraint of greatest weight
        """
        pass

    @staticmethod
    def grad_hrc(
        weights: list[float],
        names: list[str],
        types: dict[str, str],
        mu: dict[str, float],
        sigma: dict[str, float],
    ):
        """Returns the gradient of a prior towards a target difference
        between a hierarchy of constraints -- e.g. C1 > C2 > C3
        """
        pass

    """ ==================== CLASS METHODS ===================================== """

    def get_mIdx(cls, names: list[str]):
        return np.char.find(names, "*") != -1

    def get_fIdx(cls, names: list[str]):
        return np.char.find(names, "*") == -1


if __name__ == "__main__":
    ## Sample Constraints
    cns = np.asarray(["A", "B", "C"])

    ## Sample Groups
    grp = [""]

    ## Sample Distributions

    nrm = NrmGrad
    pass
