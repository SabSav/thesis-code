"""Test suit for `ising` module"""

from pytest import *
from ising import *

def test_Chain():
    """Test basic functionality of `ising.Chain`"""
    assert len(Chain(spins = [1, -1, 1, 1]).spins) == 4
    assert len(Chain(size=5).spins) == 5

    # For 3 spins we have just two energy levels which differ by 2 J
    chain = Chain(coupling=3, spins = [1, 1, 1]);
    assert chain.deltaE(0) == 6
    assert chain.deltaE(1) == 6
    assert chain.deltaE(2) == 6

    chain.spins[1] *= -1
    assert chain.deltaE(1) == -6
    assert chain.deltaE(0) == 0
