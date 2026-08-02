"""Tests de l'interface Python.

Ils couvrent ce que les tests Rust ne peuvent pas atteindre : la fidelite de
la conversion numpy, le comportement des operateurs, et la traduction des
erreurs en exceptions Python lisibles.

Lancer avec :  pytest tests/ -v
"""

import numpy as np
import pytest

import FPMsBMF as bmf

SHAPES = [(1, 1), (3, 70), (65, 129), (100, 3), (2, 127), (64, 64), (128, 65)]


def rand(rows, cols, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols)) < density

@pytest.mark.parametrize("shape", SHAPES)
def test_numpy_roundtrip(shape):
    a = rand(*shape, seed=1)
    m = bmf.BitMatrix.from_numpy(a)
    assert m.shape == shape
    assert np.array_equal(m.to_numpy(), a)


@pytest.mark.parametrize("shape", SHAPES)
def test_count_ones_matches_numpy(shape):
    a = rand(*shape, seed=2)
    assert bmf.BitMatrix.from_numpy(a).count_ones() == int(a.sum())


@pytest.mark.parametrize("shape", SHAPES)
def test_constructors(shape):
    rows, cols = shape
    assert bmf.BitMatrix.zeros(rows, cols).count_ones() == 0
    assert bmf.BitMatrix.ones(rows, cols).count_ones() == rows * cols
    assert bmf.BitMatrix(rows, cols).count_ones() == 0
    assert bmf.BitMatrix(rows, cols, fill=True).count_ones() == rows * cols


@pytest.mark.parametrize("shape", SHAPES)
def test_transpose(shape):
    a = rand(*shape, seed=3)
    m = bmf.BitMatrix.from_numpy(a)
    assert np.array_equal(m.transpose().to_numpy(), a.T)
    assert m.transpose().transpose() == m


def test_getitem_setitem():
    m = bmf.BitMatrix.zeros(5, 70)
    assert not m[2, 65]
    m[2, 65] = True
    assert m[2, 65]
    assert m.count_ones() == 1
    m[2, 65] = False
    assert m.count_ones() == 0

@pytest.mark.parametrize("shape", SHAPES)
def test_bitwise_operators_match_numpy(shape):
    a, b = rand(*shape, seed=4), rand(*shape, seed=5)
    ma, mb = bmf.BitMatrix.from_numpy(a), bmf.BitMatrix.from_numpy(b)

    assert np.array_equal((ma | mb).to_numpy(), a | b)
    assert np.array_equal((ma & mb).to_numpy(), a & b)
    assert np.array_equal((ma ^ mb).to_numpy(), a ^ b)
    assert np.array_equal((ma - mb).to_numpy(), a & ~b)
    assert np.array_equal((~ma).to_numpy(), ~a)


@pytest.mark.parametrize("shape", SHAPES)
def test_inplace_operators(shape):
    a, b = rand(*shape, seed=6), rand(*shape, seed=7)
    ma, mb = bmf.BitMatrix.from_numpy(a), bmf.BitMatrix.from_numpy(b)

    acc = ma.copy()
    acc |= mb
    assert acc == (ma | mb)

    acc = ma.copy()
    acc &= mb
    assert acc == (ma & mb)

    acc = ma.copy()
    acc ^= mb
    assert acc == (ma ^ mb)

    assert np.array_equal(ma.to_numpy(), a)


def test_copy_is_independent():
    m = bmf.BitMatrix.zeros(10, 10)
    c = m.copy()
    c[0, 0] = True
    assert m.count_ones() == 0
    assert c.count_ones() == 1


@pytest.mark.parametrize("m,k,n", [(3, 5, 7), (10, 65, 4), (65, 3, 129), (1, 1, 1)])
def test_matmul_is_boolean_product(m, k, n):
    a, b = rand(m, k, seed=8), rand(k, n, seed=9)
    expected = (a.astype(np.uint8) @ b.astype(np.uint8)) > 0
    got = (bmf.BitMatrix.from_numpy(a) @ bmf.BitMatrix.from_numpy(b)).to_numpy()
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("shape", SHAPES)
def test_hamming_and_count_andnot(shape):
    a, b = rand(*shape, seed=10), rand(*shape, seed=11)
    ma, mb = bmf.BitMatrix.from_numpy(a), bmf.BitMatrix.from_numpy(b)

    assert ma.hamming(mb) == int((a != b).sum())
    assert ma.hamming(ma) == 0
    assert ma.count_andnot(mb) == int((a & ~b).sum())


def test_equality_and_hash():
    a = rand(20, 30, seed=12)
    m1, m2 = bmf.BitMatrix.from_numpy(a), bmf.BitMatrix.from_numpy(a)
    assert m1 == m2
    assert hash(m1) == hash(m2)
    assert len({m1, m2}) == 1

    m2[0, 0] = not m2[0, 0]
    assert m1 != m2

def test_dimension_errors_are_value_errors():
    a = bmf.BitMatrix.zeros(3, 4)
    b = bmf.BitMatrix.zeros(4, 3)

    with pytest.raises(ValueError, match="incompatibles"):
        _ = a | b
    with pytest.raises(ValueError, match="incompatibles"):
        _ = a.hamming(b)
    with pytest.raises(ValueError, match="incompatibles"):
        _ = a @ a


def test_zero_dimensions_rejected():
    with pytest.raises(ValueError, match="au moins 1"):
        bmf.BitMatrix.zeros(0, 5)
    with pytest.raises(ValueError, match="au moins 1"):
        bmf.BitMatrix.zeros(5, 0)


def test_index_errors():
    m = bmf.BitMatrix.zeros(3, 4)
    with pytest.raises(IndexError):
        _ = m[3, 0]
    with pytest.raises(IndexError):
        _ = m[0, 4]


def test_unknown_method_lists_the_valid_ones():
    x = bmf.BitMatrix.from_numpy(rand(20, 15, seed=13))
    with pytest.raises(ValueError) as e:
        bmf.ao_bmf(x, 3, method="zeta2")
    for name in bmf.methods():
        assert name in str(e.value)


def test_invalid_rank():
    x = bmf.BitMatrix.from_numpy(rand(20, 15, seed=14))
    with pytest.raises(ValueError, match="au moins 1"):
        bmf.ao_bmf(x, 0)
    with pytest.raises(ValueError, match="trop grand"):
        bmf.ao_bmf(x, 40)

@pytest.mark.parametrize("method", ["naive", "zeta", "greedy", "greedy-ls"])
def test_boolls_cost_is_consistent(method):
    w_np, x_np = rand(50, 6, seed=15), rand(50, 1, 0.4, seed=16)[:, 0]
    w = bmf.BitMatrix.from_numpy(w_np)

    h, cost = bmf.boolls(w, x_np, method=method)
    assert 0 <= h < 2 ** w.cols

    selected = np.zeros(w.cols, dtype=bool)
    for k in range(w.cols):
        selected[k] = bool((h >> k) & 1)
    predicted = w_np[:, selected].any(axis=1)
    assert cost == int((predicted != x_np).sum())


def test_exact_methods_agree_and_bound_the_heuristics():
    w_np, x_np = rand(80, 8, seed=17), rand(80, 1, 0.4, seed=18)[:, 0]
    w = bmf.BitMatrix.from_numpy(w_np)
    solver = bmf.BoolLs(w)

    naive = solver.solve(x_np, method="naive")
    zeta = solver.solve(x_np, method="zeta")
    assert naive == zeta

    exact = zeta[1]
    greedy = solver.solve(x_np, method="greedy")[1]
    greedy_ls = solver.solve(x_np, method="greedy-ls")[1]

    assert exact <= greedy_ls <= greedy <= int(x_np.sum())


def test_boolls_seed_is_reproducible():
    w_np, x_np = rand(60, 8, seed=19), rand(60, 1, 0.4, seed=20)[:, 0]
    solver = bmf.BoolLs(bmf.BitMatrix.from_numpy(w_np))
    for seed in (0, 1, 42):
        a = solver.solve(x_np, method="greedy-ls", seed=seed)
        b = solver.solve(x_np, method="greedy-ls", seed=seed)
        assert a == b


def test_solve_all_matches_column_by_column():
    x_np = rand(50, 20, seed=21)
    w = bmf.BitMatrix.from_numpy(rand(50, 5, seed=22))
    x = bmf.BitMatrix.from_numpy(x_np)
    solver = bmf.BoolLs(w)

    h, total = solver.solve_all(x, method="zeta")
    assert h.shape == (5, 20)

    one_by_one = sum(solver.solve(x_np[:, j], method="zeta")[1] for j in range(20))
    assert total == one_by_one
    assert (w @ h).hamming(x) == total


@pytest.mark.parametrize("shape,r", [((30, 20), 3), ((65, 40), 5), ((100, 129), 4)])
def test_ao_bmf_error_matches_reconstruction(shape, r):
    x = bmf.BitMatrix.from_numpy(rand(*shape, seed=23))
    res = bmf.ao_bmf(x, r, method="zeta")

    assert res.w.shape == (shape[0], r)
    assert res.h.shape == (r, shape[1])
    assert res.error == x.hamming(res.reconstruct())
    assert res.error == x.hamming(res.w @ res.h)
    assert res.error <= x.count_ones()


def test_ao_bmf_is_monotone_in_rank():
    x = bmf.BitMatrix.from_numpy(rand(60, 45, seed=24))
    best = [min(bmf.ao_bmf(x, r, seed=s).error for s in range(3)) for r in range(1, 7)]
    assert best == sorted(best, reverse=True)


def test_ao_bmf_seed_is_reproducible():
    x = bmf.BitMatrix.from_numpy(rand(50, 40, seed=25))
    for seed in (0, 1, 99):
        a, b = bmf.ao_bmf(x, 4, seed=seed), bmf.ao_bmf(x, 4, seed=seed)
        assert a.error == b.error
        assert a.w == b.w and a.h == b.h


def test_ao_bmf_seeds_explore_different_optima():
    x = bmf.BitMatrix.from_numpy(rand(80, 60, seed=26))
    errors = {bmf.ao_bmf(x, 5, seed=s).error for s in range(8)}
    assert len(errors) > 1


def test_planted_instance_is_well_approximated():
    w0, h0 = rand(40, 3, 0.4, seed=27), rand(3, 30, 0.4, seed=28)
    x_np = (w0.astype(np.uint8) @ h0.astype(np.uint8)) > 0
    x = bmf.BitMatrix.from_numpy(x_np)

    best = min(bmf.ao_bmf(x, 3, max_iter=100, seed=s).error for s in range(10))
    assert best < x.count_ones() / 4


def test_degenerate_instances():
    zero = bmf.BitMatrix.zeros(20, 15)
    assert bmf.ao_bmf(zero, 3).error == 0

    full = bmf.BitMatrix.ones(25, 18)
    assert bmf.ao_bmf(full, 1).error == 0

    single = bmf.BitMatrix.from_numpy(rand(40, 1, 0.5, seed=29))
    assert bmf.ao_bmf(single, 1).error == 0

def test_from_list_basic():
    m = bmf.BitMatrix.from_list([[1, 0, 1], [0, 1, 1]])
    assert m.shape == (2, 3)
    assert m.count_ones() == 4
    assert m[0, 0] and not m[0, 1] and m[0, 2]
    assert not m[1, 0] and m[1, 1] and m[1, 2]


def test_from_list_accepts_python_bools():
    a = bmf.BitMatrix.from_list([[1, 0], [0, 1]])
    b = bmf.BitMatrix.from_list([[True, False], [False, True]])
    assert a == b


@pytest.mark.parametrize("shape", SHAPES)
def test_from_list_roundtrip(shape):
    a = rand(*shape, seed=30)
    m = bmf.BitMatrix.from_numpy(a)
    lst = m.to_list()

    assert isinstance(lst, list)
    assert isinstance(lst[0], list)
    assert all(isinstance(v, int) for v in lst[0])
    assert bmf.BitMatrix.from_list(lst) == m


@pytest.mark.parametrize("shape", SHAPES)
def test_to_list_matches_numpy(shape):
    a = rand(*shape, seed=31)
    m = bmf.BitMatrix.from_numpy(a)
    assert np.array_equal(np.array(m.to_list(), dtype=bool), a)


def test_from_list_rejects_invalid_input():
    with pytest.raises(ValueError, match="aucune ligne"):
        bmf.BitMatrix.from_list([])
    with pytest.raises(ValueError, match="lignes sont vides"):
        bmf.BitMatrix.from_list([[]])
    with pytest.raises(ValueError, match="la ligne 1"):
        bmf.BitMatrix.from_list([[1, 0, 1], [0, 1]])
    with pytest.raises(ValueError, match="seuls 0 et 1"):
        bmf.BitMatrix.from_list([[1, 2], [0, 1]])
    with pytest.raises(ValueError, match="seuls 0 et 1"):
        bmf.BitMatrix.from_list([[1, -1], [0, 1]])


def test_from_list_solves():
    w = bmf.BitMatrix.from_list(
        [[1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [0, 1], [0, 1]]
    )
    x = np.array([0, 0, 0, 1, 1, 1, 1], dtype=bool)
    solver = bmf.BoolLs(w)

    assert solver.solve(x, method="zeta") == (3, 3)
    assert solver.solve(x, method="greedy") == (0, 4)