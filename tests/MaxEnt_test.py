# Last updated: 04/17/2023

import sys
import os
from pathlib import Path

# Gets the absolute path of the directory containing the module.
module_dir = os.path.join(Path(__file__).parent.parent, "utils")

# Adds the directory to the Python path.
sys.path.append(module_dir)

import pytest
import MaxEnt as MaxEntUtil

from collections import namedtuple
import numpy as np
import math

Dataset = namedtuple("Dataset", ["constraints", "weights"])


@pytest.fixture
def ex_const_1():
    constraints = ["M1", "M2", "F"]
    weights = [7, 10, 0]
    return Dataset(constraints, weights)


@pytest.fixture
def ex_vio_1():
    return [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, 1, 0], [0, 1, 1]]),
    ]


@pytest.fixture
def ex_observed_prob_1():
    return [np.array([[1 / 3, 2 / 3, 0]]), np.array([[0, 1]])]


def test_compute_probabilities(ex_const_1, ex_vio_1):
    output = MaxEntUtil.compute_probabilities(ex_const_1.weights, ex_vio_1)

    exp0 = np.array([math.exp(-7), math.exp(-10), math.exp(0)])
    mar0 = math.exp(-7) + math.exp(-10) + math.exp(0)
    pred0 = exp0 / mar0
    assert output[0].any() == pred0.any()

    exp1 = np.array([math.exp(-17), math.exp(-7)])
    mar1 = math.exp(-17) + math.exp(-7)
    pred1 = exp1 / mar1
    assert output[1].any() == pred1.any()


def test_get_weighted_winner_violations(ex_observed_prob_1, ex_vio_1):
    output = MaxEntUtil.get_weighted_winner_violations(ex_observed_prob_1, ex_vio_1)
    expected = [np.array([[1 / 3, 2 / 3, 0]]), np.array([[0, 1, 1]])]
    assert output[0].any() == expected[0].any()
    assert output[1].any() == expected[1].any()


def test_accessors(ex_const_1):
    model = MaxEntUtil.MaxEnt(ex_const_1.constraints, ex_const_1.weights)

    assert ex_const_1.constraints == model.cns
    assert ex_const_1.weights == model.cws


def test_sorted_by_weights(ex_const_1):
    model = MaxEntUtil.MaxEnt(ex_const_1.constraints, ex_const_1.weights)
    output = model.sorted_by_weights()
    expected = [("M2", 10), ("M1", 7), ("F", 0)]
    assert output == expected
