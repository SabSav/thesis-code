"""Test suit for `ising` module"""
from pytest import *
from ising import *

def test_Chain():
    """Test basic functionality of `ising.Chain`"""
    assert len(Chain(spins=[1, -1, 1, 1]).spins) == 4
    assert len(Chain(size=5).spins) == 5
    raises(TypeError, Chain, uknown_keyword=True)

    # For 3 spins we have just two energy levels which differ by 4J
    chain = Chain(coupling=3, spins=[1, 1, 1])
    assert chain.deltaE(0) == 12
    assert chain.deltaE(1) == 12
    assert chain.deltaE(2) == 12
    assert chain.energy() == -9

    chain.spins[1] *= -1
    assert chain.deltaE(1) == -12
    assert chain.deltaE(0) == 0
    assert chain.energy() == 3


def test_DynamicChain():
    """Test basic functionality of `ising.DynamicChain`"""
    chain = DynamicChain()
    assert (chain.action_rates == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).all()
    assert chain.action_rate(1, 1) == 1.0
    assert chain.action_rate(1, -1) == 1.0

def test_theoretical_distributions():
    """Test `theoretical_distributions()`"""
    eng, mgn = theoretical_distributions(Chain())

    assert len(eng) == 2
    assert len(mgn) == 4

    assert np.all(eng[:, 0] == [-3, 1])
    assert mgn[0, 0] == -1.0
    assert mgn[1, 0] == -1/3
    assert mgn[2, 0] == 1/3
    assert mgn[3, 0] == 1.0

    assert eng[0, 1] == approx(0.948, abs=0.005)
    assert eng[1, 1] == approx(0.052, abs=0.005)

def test_typical_delta():
    chain = Chain(coupling=1, spins=[1, 1, 1])

    assert chain.typical_delta() == 4.0
    for s in chain.spins: assert s == 1
