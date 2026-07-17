"""Numerical parity between the numpy AR-HMM oracle and the numba backend.

The pure-numpy functions/methods in ``arhmm_model`` are the reference; the
``@njit`` backend must reproduce them (tight parity on the primitives for fixed
params; soft high-agreement on a full EM fit, whose local optimum can drift on
marginal data with tiny float differences).
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.behavior.feature_library.arhmm_model import (
    ARHMM,
    _REG,
    _ar_log_likelihoods,
    _ar_log_likelihoods_nb,
    _backward,
    _backward_nb,
    _build_design,
    _cholesky_all,
    _cholesky_stack,
    _forward,
    _forward_nb,
    _m_step_nb,
    fit_numba,
    predict_numba,
)


def _random_params(
    K: int, D: int, nlags: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Valid random AR-HMM parameters (A, Q SPD, uniform trans/start)."""
    rng = np.random.default_rng(seed)
    dim_in = D * nlags + 1
    A = rng.standard_normal((K, D, dim_in)) * 0.3
    Q = np.empty((K, D, D))
    for k in range(K):
        M = rng.standard_normal((D, D))
        Q[k] = M @ M.T + np.eye(D)
    log_trans = np.log(np.full((K, K), 1.0 / K))
    log_start = np.log(np.full(K, 1.0 / K))
    return A, Q, log_trans, log_start


def _random_walk(T: int, D: int, seed: int, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((T, D)) * 0.1, axis=0) + shift


# --------------------------------------------------------------------------
# Primitive parity (fixed params → tight tolerance)
# --------------------------------------------------------------------------


def test_emission_loglik_parity() -> None:
    K, D, nlags = 4, 3, 1
    A, Q, _, _ = _random_params(K, D, nlags, seed=0)
    X = _random_walk(200, D, seed=1)
    x_in = np.ascontiguousarray(_build_design(X, nlags))
    x_out = np.ascontiguousarray(X[nlags:])

    Q_cho, Q_logdet = _cholesky_all(Q)
    L, logdet = _cholesky_stack(Q)

    ll_np = _ar_log_likelihoods(X, A, Q_cho, Q_logdet, nlags, K)
    ll_nb = _ar_log_likelihoods_nb(x_in, x_out, A, L, logdet)
    assert np.allclose(ll_np, ll_nb, rtol=1e-9, atol=1e-9)


def test_forward_backward_parity() -> None:
    K, D, nlags = 5, 3, 1
    A, Q, log_trans, log_start = _random_params(K, D, nlags, seed=2)
    X = _random_walk(250, D, seed=3)
    Q_cho, Q_logdet = _cholesky_all(Q)
    ll = _ar_log_likelihoods(X, A, Q_cho, Q_logdet, nlags, K)

    assert np.allclose(
        _forward(ll, log_trans, log_start),
        _forward_nb(ll, log_trans, log_start),
        rtol=1e-8,
        atol=1e-8,
    )
    assert np.allclose(
        _backward(ll, log_trans),
        _backward_nb(ll, log_trans),
        rtol=1e-8,
        atol=1e-8,
    )


def test_estep_suffstats_and_xi_parity() -> None:
    """The numba per-sequence E-step suff-stats + xi match the numpy _e_step."""
    from mosaic.behavior.feature_library.arhmm_model import _e_step_seq_nb

    K, D, nlags = 5, 3, 1
    dim_in = D * nlags + 1
    A, Q, log_trans, log_start = _random_params(K, D, nlags, seed=4)
    X = _random_walk(300, D, seed=5)
    x_in = np.ascontiguousarray(_build_design(X, nlags))
    x_out = np.ascontiguousarray(X[nlags:])
    Q_cho, Q_logdet = _cholesky_all(Q)
    L, logdet = _cholesky_stack(Q)

    model = ARHMM(n_states=K, n_lags=nlags, sticky_weight=100.0, random_state=1)
    ll_np, suff = model._e_step([X], A, Q, Q_cho, Q_logdet, log_trans, log_start, K, D)
    ll_nb, soi, sii, soo, gs, xs, sts = _e_step_seq_nb(
        x_in, x_out, A, L, logdet, log_trans, log_start, K, D, dim_in
    )

    assert ll_np == pytest.approx(ll_nb, rel=1e-8)
    assert np.allclose(suff["S_out_in"], soi, rtol=1e-8, atol=1e-8)
    assert np.allclose(suff["S_in_in"], sii, rtol=1e-8, atol=1e-8)
    assert np.allclose(suff["S_out_out"], soo, rtol=1e-8, atol=1e-8)
    assert np.allclose(suff["gamma_sum"], gs, rtol=1e-8, atol=1e-8)
    assert np.allclose(suff["xi_sum"], xs, rtol=1e-8, atol=1e-8)
    assert np.allclose(suff["start_sum"], sts, rtol=1e-8, atol=1e-8)


def test_mstep_parity() -> None:
    K, D, nlags = 5, 3, 1
    dim_in = D * nlags + 1
    A, Q, log_trans, log_start = _random_params(K, D, nlags, seed=6)
    X = _random_walk(300, D, seed=7)
    Q_cho, Q_logdet = _cholesky_all(Q)

    model = ARHMM(n_states=K, n_lags=nlags, sticky_weight=100.0, random_state=1)
    _, suff = model._e_step([X], A, Q, Q_cho, Q_logdet, log_trans, log_start, K, D)
    A_np, Q_np, lt_np, ls_np = model._m_step(suff, K, D)
    A_nb, Q_nb, lt_nb, ls_nb = _m_step_nb(
        suff["S_out_in"],
        suff["S_in_in"],
        suff["S_out_out"],
        suff["gamma_sum"],
        suff["xi_sum"],
        suff["start_sum"],
        K,
        D,
        dim_in,
        100.0,
        _REG,
    )
    assert np.allclose(A_np, A_nb, rtol=1e-8, atol=1e-8)
    assert np.allclose(Q_np, Q_nb, rtol=1e-8, atol=1e-8)
    assert np.allclose(lt_np, lt_nb, rtol=1e-8, atol=1e-8)
    assert np.allclose(ls_np, ls_nb, rtol=1e-8, atol=1e-8)


def test_viterbi_exact_match_on_identical_params() -> None:
    """Viterbi is exact-arg, so identical params → identical integer states."""
    K, D, nlags = 6, 3, 1
    A, Q, log_trans, log_start = _random_params(K, D, nlags, seed=8)
    X = _random_walk(400, D, seed=9)
    Q_cho, Q_logdet = _cholesky_all(Q)

    # numpy predict (mirrors its internal Viterbi + lag padding)
    model = ARHMM(n_states=K, n_lags=nlags, random_state=1)
    model.A_, model.Q_ = A, Q
    model.Q_cho_, model.Q_logdet_ = Q_cho, Q_logdet
    model.log_transmat_, model.log_startprob_ = log_trans, log_start
    model.n_features_ = D
    states_np = model.predict(X)
    states_nb = predict_numba(model, X)
    assert np.array_equal(states_np, states_nb)


# --------------------------------------------------------------------------
# Full-fit equivalence (soft agreement)
# --------------------------------------------------------------------------


def test_full_fit_equivalence() -> None:
    D = 3
    seqs = [
        _random_walk(400, D, seed=10),
        _random_walk(350, D, seed=11, shift=4.0),
    ]
    kw = dict(n_states=6, n_lags=1, n_iter=40, n_restarts=1, random_state=42)

    m_np = ARHMM(**kw).fit([s.copy() for s in seqs])
    m_nb = fit_numba([s.copy() for s in seqs], **kw)

    # Fitted AR params should match closely (identical KMeans init + math).
    assert m_np.A_ is not None and m_nb.A_ is not None
    assert np.allclose(m_np.A_, m_nb.A_, rtol=1e-5, atol=1e-6)

    # Decoded syllables should agree on held-out-style decoding.
    agree = (m_np.predict(seqs[0]) == predict_numba(m_nb, seqs[0])).mean()
    assert agree >= 0.99
