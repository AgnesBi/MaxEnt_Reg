import sys
import os
from pathlib import Path

# Gets the absolute path of the directory containing the module.
module_dir = os.path.join(Path(__file__).parent.parent, "utils")

# Adds the directory to the Python path.
sys.path.append(module_dir)

import pytest
import MaxEnt as MaxEntClass

from collections import namedtuple
import numpy as np

Dataset = namedtuple('Dataset', ['constraints', 'weights', 'model'])

@pytest.fixture
def dataset_1():
    constraints = ["M1", "M2", "F"]
    weights = [7, 10, 0]
    model = MaxEntClass.MaxEnt(constraints, weights)
    return Dataset(constraints, weights, model)

def test_accessors(dataset_1):
    assert dataset_1.constraints == dataset_1.model.cns
    assert dataset_1.weights == dataset_1.model.cws


def test_sorted_by_weights(dataset_1):
    output = dataset_1.model.sorted_by_weights()
    expected = [("M2", 10), ("M1", 7), ("F", 0)]
    assert output == expected