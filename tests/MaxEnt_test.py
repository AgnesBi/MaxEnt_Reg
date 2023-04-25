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

############################ Test cases ###############################


@pytest.fixture
def test_const():
    constraints = ["M1", "M2", "F"]
    weights = np.array([[7], [10], [0]])
    return Dataset(constraints, weights)


@pytest.fixture
def test_vio():
    return [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, 1, 0], [0, 1, 1]]),
        np.array([[0, 0, 0], [0, 2, 1]]),
        np.array([[1, 0, 0], [0, 1, 1]]),
    ]


@pytest.fixture
def test_probs():
    return [
        np.array([[1 / 3], [2 / 3], [0]]),
        np.array([[0], [1]]),
        np.array([[1], [0]]),
        None,
    ]


############################ Test functions ###############################


def test_compute_probabilities(test_const, test_vio):
    exp0 = np.array([math.exp(-7), math.exp(-10), math.exp(0)])
    mar0 = math.exp(-7) + math.exp(-10) + math.exp(0)
    pred0 = exp0 / mar0
    output0 = (
        MaxEntUtil.compute_probabilities(test_const.weights, test_vio[0])
        .reshape((1, 3))
        .squeeze()
    )

    exp1 = np.array([math.exp(-17), math.exp(-10)])
    mar1 = math.exp(-17) + math.exp(-10)
    pred1 = exp1 / mar1
    output1 = (
        MaxEntUtil.compute_probabilities(test_const.weights, test_vio[1])
        .reshape((1, 2))
        .squeeze()
    )

    exp2 = np.array([math.exp(0), math.exp(-20)])
    mar2 = math.exp(0) + math.exp(-20)
    pred2 = exp2 / mar2
    output2 = (
        MaxEntUtil.compute_probabilities(test_const.weights, test_vio[2])
        .reshape((1, 2))
        .squeeze()
    )

    exp3 = np.array([math.exp(-7), math.exp(-10)])
    mar3 = math.exp(-7) + math.exp(-10)
    pred3 = exp3 / mar3
    output3 = (
        MaxEntUtil.compute_probabilities(test_const.weights, test_vio[3])
        .reshape((1, 2))
        .squeeze()
    )

    np.testing.assert_array_equal(output0, pred0)
    np.testing.assert_array_equal(output1, pred1)
    np.testing.assert_array_equal(output2, pred2)
    np.testing.assert_array_equal(output3, pred3)


def test_get_weighted_winner_violations(test_probs, test_vio):
    expected = [
        np.array([[1 / 3, 2 / 3, 0]]),
        np.array([[0, 1, 1]]),
        np.array([[0, 0, 0]]),
    ]

    output0 = MaxEntUtil.get_weighted_winner_violations(test_probs[0], test_vio[0])
    output1 = MaxEntUtil.get_weighted_winner_violations(test_probs[1], test_vio[1])
    output2 = MaxEntUtil.get_weighted_winner_violations(test_probs[2], test_vio[2])

    np.testing.assert_array_equal(output0, expected[0])
    np.testing.assert_array_equal(output1, expected[1])
    np.testing.assert_array_equal(output2, expected[2])


def test_accessors(test_const):
    model = MaxEntUtil.MaxEnt(test_const.constraints, test_const.weights)

    assert test_const.constraints == model.cns
    np.testing.assert_array_equal(test_const.weights, model.cws)


def test_sorted_by_weights(test_const):
    model = MaxEntUtil.MaxEnt(test_const.constraints, test_const.weights)
    output = model.sorted_by_weights()
    expected = [("M2", 10), ("M1", 7), ("F", 0)]
    assert output == expected
