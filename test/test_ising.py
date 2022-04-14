"""Test suit for `ising` module"""

from pytest import *
from ising import *

def test_Chain():
    """Test basic functionality of `ising.Chain`"""
    assert len(Chain(spins = [1, -1, 1, 1]).spins) == 4
    assert len(Chain(size=5).spins) == 5
    raises(TypeError, Chain, uknown_keyword=True)

    # For 3 spins we have just two energy levels which differ by 2 J
    chain = Chain(coupling=3, spins = [1, 1, 1]);
    assert chain.deltaE(0) == 6
    assert chain.deltaE(1) == 6
    assert chain.deltaE(2) == 6

    chain.spins[1] *= -1
    assert chain.deltaE(1) == -6
    assert chain.deltaE(0) == 0

def test_DynamicChain():
    """Test basic functionality of `ising.DynamicChain`"""
    chain = DynamicChain()
    assert (chain.action_rates == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).all()
    assert chain.action_rate(1, 1) == 1.0
    assert chain.action_rate(1, -1) == 1.0