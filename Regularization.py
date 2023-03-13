import numpy as np

class Regularization:

    """ ==================== INITIALIZATION ==================================== """
    def __init__(self, sigma: float, mu: float):
        self.sigma = sigma
        self.mu = mu

    """ ==================== GRADIENTS ========================================= """
    def grad_uni(self, weights: list[float], names: list[str]):
        """Returns the gradient given a flat prior; otherwise, 0
        """
        return 0

    def grad_nrm(self, weights: list[float], names: list[str]):
        """Returns the gradient of the normal objective function
        """
        pass

    def grad_sum(self, weights: list[float], names: list[str]):
        """Returns the gradient of the summed differences objective function
        """

        ## Check to make sure the constraint weights and shapes are the same shape
        assert weights.shape == names.shape, "Constraint weights and names not equal shape"

        ## Get the indices of the markedness and faithfulness constraints
        mIdx = self.get_mIdx(names)
        fIdx = ~mIdx

        ## Sum the constraint weights of the markedness vs faithfulness constraints
        mSummed = weights[mIdx].sum()
        fSummed = weights[fIdx].sum()

        ## Calculate the gradient of the markedness vs faithfulness constraints
        mGrad = (mSummed - fSummed - self.mu) / (self.sigma ** 2)
        fGrad = (fSummed - mSummed + self.mu) / (self.sigma ** 2)

        ## Return the expected gradients for each weight
        wGrad = np.zeros(weights.shape)
        wGrad[mIdx] += mGrad
        wGrad[fIdx] += fGrad

        return wGrad

    def grad_ind(self, weights, names):
        """Returns the gradient of the differences between the markedness constraints
        and the faithfulness constraint of greatest weight
        """
        pass 
    
    """ ==================== CLASS METHODS ===================================== """
    def get_mIdx(cls, names: list[str]):
        return np.char.find(names, "*") != -1

    def get_fIdx(cls, names: list[str]):
        return np.char.find(names, "*") == -1