import pytest
import main
import numpy as np

def test_magnitization():
    config = np.array([1,-1, 1])
    assert main.magnetization(config) == pytest.approx(0.3, abs=5e-2)

def test_ai(): pass

def test_aiwithmc(): pass
