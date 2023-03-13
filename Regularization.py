## Type hint and labelling
from typing import Union

## Matrix manipulation
import numpy as np


""" =============================================================================== 
                    REGULARIZATION CLASS DEFINITION 
=============================================================================== """


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
            wGrad = (weights - mu) / (sigma ** 2)
        
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
        mu: Union[float, dict[str, float]],
        sigma: Union[float, dict[str, float]],
    ):
        """Returns the gradient of the differences between the markedness constraints
        and the faithfulness constraint of greatest weight
        """
        pass

    """ ==================== CLASS METHODS ===================================== """

    def get_mIdx(cls, names: list[str]):
        return np.char.find(names, "*") != -1

    def get_fIdx(cls, names: list[str]):
        return np.char.find(names, "*") == -1