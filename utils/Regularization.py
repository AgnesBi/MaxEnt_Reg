import numpy as np


""" ===================================================================================
                CONSTRAINT CATEGORIES CLASS DEFINITION 
=================================================================================== """


class Categories:
    def __init__(self, cns: list, grs: list, mus: list, sms: list):
        """Categories implementation. This class takes in a four arguments:

            (1) cns: the list of constraint names
            (2) grs: the list of constraint names belonging to a group
            (3) mus: the mus of each target distribution for each group (grs, )
            (4) sms: the sigmas of each target distribution for each group (grs, )

        The class checks to make sure that any given constraint only belongs
        to a single group. If it succeeds, both the constraint names as well
        as the groupings (converted from constraint names to indices) are
        saved as instance variables.
        """

        ## Check to make sure that the constraint groupings are disjoint
        i = set.intersection(*[set(gr) for gr in grs])
        assert i == set(), f"Groupings are not disjoint: {i} found in multiple groups."

        ## Store the constraints and grouping as instance variables
        self._cns, self._ncs = np.array(cns), len(cns)
        self._grs, self._ngs = np.array([np.isin(self.cns, gr) for gr in grs]), len(grs)

        ## Store the mus and sigmas as instance variables
        self._mus = np.array(mus)
        self._sms = np.array(sms)

    @property
    def cns(self):
        return self._cns

    @property
    def ncs(self):
        return self._ncs

    @property
    def grs(self):
        return self._grs

    @property
    def ngs(self):
        return self._ngs

    @property
    def mus(self):
        return self._mus

    @property
    def sms(self):
        return self._sms


""" ===================================================================================
                PRIOR CLASS DEFINITIONS
=================================================================================== """


class TGTPrior(Categories):
    def __init__(
        self, cns: list[str], grs: list[list[str]], mus: list[float], sms: list[float]
    ):
        """Target gradient class. Each group is assumed to have individual target
        weights. This class takes in four arguments:

            (1) cns: the list of constraint names
            (2) grs: the list of constraint names belonging to a group
            (3) mus: the mus of each target distribution for each group (grs, )
            (4) sms: the sigmas of each target distribution for each group (grs, )

        The class checks to make sure that the number of groups and parameters are
        identical. If it succeeds, all of the values are saved as instance variables.
        """

        ## Check to make sure that the number of groups and parameters are the same
        assert len(grs) == len(mus) == len(sms), "Unequal groups and parameters!"

        ## Inherit from superclass
        super().__init__(cns, grs, mus, sms)

        ## Initialize the mus and sigmas from the list of lists
        self._vmu = np.full(self.ncs, np.nan)
        self._vsm = np.full(self.ncs, np.nan)
        for g, m, s in zip(self.grs, self.mus, self.sms):
            self._vmu[g] = m
            self._vsm[g] = s

        ## Get the indices of the relevant constraints. The below code is largely
        ## redundant, but is included for completeness
        self._cid = ~np.isnan(self.vmu) & ~np.isnan(self.vsm)

    def gradient(self, cws):
        """Calculates the gradient of the constraint weights given the assigned groups
        and parameterization of mus and sigmas. Checks to make sure that the constraint
        weights and constraint parameters are the same size.
        """

        ## Check to make sure that the number of constraint weights align with the
        ## number of constraints
        assert self.ncs == cws.size, f"cns ({self.ncs}) != cws ({cws.size})"

        ## Calculate the gradient for each relevant constraint
        grd = (self.vmu[self.cid] - cws[self.cid]) / (self.vsm**2)

        return grd

    @property
    def vmu(self):
        return self._vmu

    @property
    def vsm(self):
        return self._vsm

    @property
    def cid(self):
        return self._cid


class DIFPrior(Categories):
    def __init__(
        self,
        cns: list[str],
        grs: list[list[str]],
        mus: list[float],
        sms: list[float],
        cmp: list[list[int]],
    ):
        """Gradient class. Each group is assumed to have summed target weight differences.
        This class takes in five arguments:

            (1) cns: the list of constraint names
            (2) grs: the list of constraint names belonging to a group
            (3) mus: the mus of each target distribution for each group (gr - 1, )
            (4) sms: the sigmas of each target distribution for each group (gr - 1, )
            (5) cmp: the indices of each group to take difference (gr - 1, 2)

        The class checks to  make sure that the number of groups and parameters are
        identical. If it succeeds, all of the values are saved as instance variables.
        """

        ## Check to make sure that the number of groups and parameters are the same
        assert len(grs) - 1 == len(mus) == len(sms), "Unequal groups and parameters!"

        ## Inherit from superclass
        super().__init__(cns, grs, mus, sms)

        ## Initialize the group comparisons instance variable
        self._cmp = cmp

    def gradient(self, cws):
        """Calculates the gradient of the constraint weights given the assigned groups
        and parameterization of mus and sigmas. Checks to make sure that the constraint
        weights and constraint parameters are the same size.
        """

        ## Check to make sure that the number of constraint weights align with the
        ## number of constraints
        assert self.ncs == cws.size, f"cns ({self.ncs}) != cws ({cws.size})"

        ## For each group, sum the constraint weights
        scw = np.asarray([cws[self.grs[x]].sum() for x in range(self.ngs)])

        ## Calculate the gradient for each relevant pair of groups
        grd = np.zeros(self.ncs)
        for i, c in enumerate(self.cmp):
            ## Retrieve the sums for the group comparison
            x, y = scw[c]

            ## Compute the difference between the groups
            dxy = x - y
            dyx = y - x

            ## Retrieve the mus and sigmas for the group comparison
            m = self.mus[i]
            s = self.sms[i]

            ## Update the gradient
            grd[self.grs[c[0]]] += (dxy - m) / (s**2)
            grd[self.grs[c[1]]] += (dyx + m) / (s**2)

        return grd

    @property
    def cmp(self):
        return self._cmp


if __name__ == "__main__":
    cns = ["A", "B", "C"]
    grs = [["A"], ["B"], ["C"]]
    mus = [10, 10, 1]
    sms = [3, 3, 3]

    x = TGTPrior(cns, grs, mus, sms)
    print(x.gradient(np.array([100, 50, 0])))

    mus = [5, 5]
    sms = [3, 3]
    msg = [[0, 1], [1, 2]]
    x = DIFPrior(cns, grs, mus, sms, msg)
    print(x.gradient(np.array([100, 50, 0])))
