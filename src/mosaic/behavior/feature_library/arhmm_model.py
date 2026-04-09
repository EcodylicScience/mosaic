"""Autoregressive Hidden Markov Model (AR-HMM) with EM fitting.

A standalone implementation using numpy/scipy — no external HMM library
required.  Fits switching autoregressive dynamics with sticky transitions
via Expectation–Maximisation and decodes the most-likely state sequence
with the Viterbi algorithm.

This module has **no** mosaic imports and can be tested independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

log = logging.getLogger(__name__)

# Tiny constant added to covariance diagonals for numerical stability.
_REG = 1e-6


# ---------------------------------------------------------------------------
# AR-HMM model
# ---------------------------------------------------------------------------


@dataclass
class ARHMM:
    """Autoregressive Hidden Markov Model.

    Each of the *K* discrete states owns an AR(``n_lags``) linear model:

        x_t = A_k @ [x_{t-1}; ...; x_{t-nlags}; 1] + ε,   ε ~ N(0, Q_k)

    Transitions between states are governed by a *K × K* matrix with a
    *sticky* prior that encourages self-transitions (controlled by
    ``sticky_weight``).

    Parameters
    ----------
    n_states : int
        Maximum number of hidden states.
    n_lags : int
        AR order (number of lagged frames used as regressors).
    sticky_weight : float
        Extra pseudo-count added to the diagonal of the transition matrix
        during M-step updates.  Larger values → states persist longer.
    n_iter : int
        Maximum EM iterations per restart.
    tol : float
        Convergence threshold on relative change in log-likelihood.
    n_restarts : int
        Number of random restarts; the best (highest LL) is kept.
    random_state : int | None
        Seed for reproducibility.
    """

    n_states: int = 50
    n_lags: int = 1
    sticky_weight: float = 100.0
    n_iter: int = 200
    tol: float = 1e-4
    n_restarts: int = 1
    random_state: int | None = None

    # Fitted parameters (set by .fit())
    A_: np.ndarray | None = field(default=None, repr=False)  # (K, D, D*nlags+1)
    Q_: np.ndarray | None = field(default=None, repr=False)  # (K, D, D)
    Q_cho_: list | None = field(default=None, repr=False)  # Cholesky factors
    Q_logdet_: np.ndarray | None = field(default=None, repr=False)  # (K,)
    log_transmat_: np.ndarray | None = field(default=None, repr=False)  # (K, K)
    log_startprob_: np.ndarray | None = field(default=None, repr=False)  # (K,)
    n_features_: int | None = field(default=None, repr=False)
    active_states_: np.ndarray | None = field(default=None, repr=False)

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(self, sequences: list[np.ndarray]) -> ARHMM:
        """Fit the AR-HMM via EM on *sequences*.

        Parameters
        ----------
        sequences : list of ndarray, each shape (T_i, D)
            Feature matrices for each sequence.

        Returns
        -------
        self
        """
        if not sequences:
            msg = "No sequences provided for fitting."
            raise ValueError(msg)

        D = sequences[0].shape[1]
        for s in sequences:
            if s.shape[1] != D:
                msg = f"Feature dim mismatch: expected {D}, got {s.shape[1]}"
                raise ValueError(msg)
            if s.shape[0] <= self.n_lags:
                msg = (
                    f"Sequence length {s.shape[0]} must exceed n_lags={self.n_lags}"
                )
                raise ValueError(msg)

        self.n_features_ = D
        K = self.n_states
        rng = np.random.default_rng(self.random_state)

        best_ll = -np.inf
        best_params: tuple | None = None

        for restart in range(self.n_restarts):
            seed = int(rng.integers(0, 2**31))
            A, Q, log_trans, log_start = self._init_params(sequences, K, D, seed)
            Q_cho, Q_logdet = _cholesky_all(Q)

            prev_ll = -np.inf
            for it in range(self.n_iter):
                # ----- E-step -----
                total_ll, suff = self._e_step(
                    sequences, A, Q, Q_cho, Q_logdet, log_trans, log_start, K, D,
                )

                # Convergence check
                rel_change = abs(total_ll - prev_ll) / max(abs(prev_ll), 1.0)
                if it > 0 and rel_change < self.tol:
                    log.info(
                        "restart %d converged at iter %d  LL=%.4f",
                        restart, it, total_ll,
                    )
                    break
                prev_ll = total_ll

                # ----- M-step -----
                A, Q, log_trans, log_start = self._m_step(suff, K, D)
                Q_cho, Q_logdet = _cholesky_all(Q)
            else:
                log.info(
                    "restart %d reached max iter %d  LL=%.4f",
                    restart, self.n_iter, total_ll,
                )

            if total_ll > best_ll:
                best_ll = total_ll
                best_params = (A, Q, Q_cho, Q_logdet, log_trans, log_start)

        assert best_params is not None
        self.A_, self.Q_, self.Q_cho_, self.Q_logdet_, self.log_transmat_, self.log_startprob_ = best_params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding → per-frame state labels.

        Parameters
        ----------
        X : ndarray, shape (T, D)

        Returns
        -------
        labels : ndarray of int32, shape (T,)
            State assignments.  The first ``n_lags`` frames are assigned
            the same state as frame ``n_lags`` (the earliest decodable
            frame).
        """
        self._check_fitted()
        assert self.A_ is not None  # for type-checker
        K = self.A_.shape[0]
        log_lik = _ar_log_likelihoods(
            X, self.A_, self.Q_cho_, self.Q_logdet_, self.n_lags, K,
        )
        T_eff = log_lik.shape[0]

        # Viterbi (log-space)
        V = np.empty((T_eff, K), dtype=np.float64)
        ptr = np.empty((T_eff, K), dtype=np.intp)
        V[0] = self.log_startprob_ + log_lik[0]

        for t in range(1, T_eff):
            scores = V[t - 1, :, None] + self.log_transmat_  # (K, K)
            ptr[t] = scores.argmax(axis=0)
            V[t] = scores[ptr[t], np.arange(K)] + log_lik[t]

        # Back-trace
        states = np.empty(T_eff, dtype=np.int32)
        states[-1] = int(V[-1].argmax())
        for t in range(T_eff - 2, -1, -1):
            states[t] = ptr[t + 1, states[t + 1]]

        # Pad leading lags with the first decoded state
        full = np.empty(X.shape[0], dtype=np.int32)
        full[: self.n_lags] = states[0]
        full[self.n_lags :] = states
        return full

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood of *X* under the fitted model."""
        self._check_fitted()
        assert self.A_ is not None
        K = self.A_.shape[0]
        log_lik = _ar_log_likelihoods(
            X, self.A_, self.Q_cho_, self.Q_logdet_, self.n_lags, K,
        )
        log_alpha = _forward(log_lik, self.log_transmat_, self.log_startprob_)
        return float(logsumexp(log_alpha[-1]))

    def prune_states(self, sequences: list[np.ndarray], threshold: float = 0.01) -> None:
        """Drop states whose posterior mass is below *threshold*.

        Re-indexes the remaining states to 0..K'-1.
        """
        self._check_fitted()
        assert self.A_ is not None
        K = self.A_.shape[0]

        # Accumulate posterior mass per state
        state_mass = np.zeros(K, dtype=np.float64)
        total_frames = 0
        for X in sequences:
            log_lik = _ar_log_likelihoods(
                X, self.A_, self.Q_cho_, self.Q_logdet_, self.n_lags, K,
            )
            log_alpha = _forward(log_lik, self.log_transmat_, self.log_startprob_)
            log_beta = _backward(log_lik, self.log_transmat_)
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            state_mass += gamma.sum(axis=0)
            total_frames += gamma.shape[0]

        usage = state_mass / total_frames
        keep = np.where(usage >= threshold)[0]

        if len(keep) == K:
            self.active_states_ = np.arange(K, dtype=np.int32)
            return

        log.info(
            "Pruning: keeping %d of %d states (threshold=%.3f)",
            len(keep), K, threshold,
        )
        self.A_ = self.A_[keep]
        self.Q_ = self.Q_[keep]
        self.Q_cho_ = [self.Q_cho_[i] for i in keep]
        self.Q_logdet_ = self.Q_logdet_[keep]

        trans = np.exp(self.log_transmat_)[np.ix_(keep, keep)]
        trans /= trans.sum(axis=1, keepdims=True)
        self.log_transmat_ = np.log(np.clip(trans, 1e-300, None))

        start = np.exp(self.log_startprob_)[keep]
        start /= start.sum()
        self.log_startprob_ = np.log(np.clip(start, 1e-300, None))

        self.n_states = len(keep)
        self.active_states_ = keep.astype(np.int32)

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.A_ is None:
            msg = "Model not fitted — call .fit() first."
            raise RuntimeError(msg)

    def _init_params(
        self,
        sequences: list[np.ndarray],
        K: int,
        D: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """K-means initialisation of AR parameters."""
        from sklearn.cluster import KMeans

        all_data = np.vstack(sequences)
        n_samples = all_data.shape[0]
        effective_k = min(K, n_samples)

        km = KMeans(n_clusters=effective_k, n_init=1, random_state=seed).fit(all_data)
        labels_all = km.predict(all_data)

        nlags = self.n_lags
        dim_in = D * nlags + 1  # AR input dimension (lags + bias)

        A = np.zeros((effective_k, D, dim_in), dtype=np.float64)
        Q = np.zeros((effective_k, D, D), dtype=np.float64)

        # Build per-sequence lagged design matrices
        x_outs: list[np.ndarray] = []
        x_ins: list[np.ndarray] = []
        label_effs: list[np.ndarray] = []
        offset = 0
        for X in sequences:
            T = X.shape[0]
            x_out = X[nlags:]  # (T-nlags, D)
            x_in = _build_design(X, nlags)  # (T-nlags, D*nlags+1)
            x_outs.append(x_out)
            x_ins.append(x_in)
            label_effs.append(labels_all[offset + nlags : offset + T])
            offset += T

        x_out_all = np.vstack(x_outs)
        x_in_all = np.vstack(x_ins)
        lbl_all = np.concatenate(label_effs)

        for k in range(effective_k):
            mask = lbl_all == k
            n_k = int(mask.sum())
            if n_k < dim_in + 1:
                # Not enough data for OLS; use identity AR with large noise
                A[k, :, :D] = 0.9 * np.eye(D)
                Q[k] = np.eye(D)
                continue

            xo = x_out_all[mask]  # (n_k, D)
            xi = x_in_all[mask]  # (n_k, dim_in)
            # Regularised OLS: A_k = (X_out^T X_in) @ inv(X_in^T X_in + λI)
            S_in = xi.T @ xi + _REG * np.eye(dim_in)
            S_out_in = xo.T @ xi
            A[k] = cho_solve(cho_factor(S_in), S_out_in.T).T

            resid = xo - xi @ A[k].T
            Q[k] = (resid.T @ resid) / n_k + _REG * np.eye(D)

        # Empirical transition matrix + sticky prior
        trans_counts = np.zeros((effective_k, effective_k), dtype=np.float64)
        for lbl_seq in label_effs:
            for t in range(len(lbl_seq) - 1):
                trans_counts[lbl_seq[t], lbl_seq[t + 1]] += 1
        trans_counts += 1.0  # Laplace smoothing
        trans_counts[np.diag_indices(effective_k)] += self.sticky_weight
        trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        log_trans = np.log(np.clip(trans, 1e-300, None))

        # Uniform start
        log_start = np.full(effective_k, -np.log(effective_k))

        return A, Q, log_trans, log_start

    def _e_step(
        self,
        sequences: list[np.ndarray],
        A: np.ndarray,
        Q: np.ndarray,
        Q_cho: list,
        Q_logdet: np.ndarray,
        log_trans: np.ndarray,
        log_start: np.ndarray,
        K: int,
        D: int,
    ) -> tuple[float, dict]:
        """E-step: forward-backward on each sequence, accumulate sufficient
        statistics."""
        nlags = self.n_lags
        dim_in = D * nlags + 1

        # Sufficient statistics
        S_out_in = np.zeros((K, D, dim_in), dtype=np.float64)
        S_in_in = np.zeros((K, dim_in, dim_in), dtype=np.float64)
        S_out_out = np.zeros((K, D, D), dtype=np.float64)
        gamma_sum = np.zeros(K, dtype=np.float64)
        xi_sum = np.zeros((K, K), dtype=np.float64)
        start_sum = np.zeros(K, dtype=np.float64)
        total_ll = 0.0

        for X in sequences:
            log_lik = _ar_log_likelihoods(X, A, Q_cho, Q_logdet, nlags, K)
            T_eff = log_lik.shape[0]

            log_alpha = _forward(log_lik, log_trans, log_start)
            log_beta = _backward(log_lik, log_trans)

            # Evidence (log-likelihood of this sequence)
            ll_seq = float(logsumexp(log_alpha[-1]))
            total_ll += ll_seq

            # State posteriors γ
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)  # (T_eff, K)

            # Transition posteriors ξ
            for t in range(T_eff - 1):
                # ξ[t, j, k] ∝ α[t,j] * A[j,k] * lik[t+1,k] * β[t+1,k]
                log_xi_t = (
                    log_alpha[t, :, None]
                    + log_trans
                    + log_lik[t + 1, None, :]
                    + log_beta[t + 1, None, :]
                )
                log_xi_t -= logsumexp(log_xi_t)
                xi_sum += np.exp(log_xi_t)

            start_sum += gamma[0]

            # AR sufficient statistics
            x_out = X[nlags:]  # (T_eff, D)
            x_in = _build_design(X, nlags)  # (T_eff, dim_in)

            for k in range(K):
                w = gamma[:, k]  # (T_eff,)
                wx_out = x_out * w[:, None]  # (T_eff, D)
                S_out_in[k] += wx_out.T @ x_in
                S_in_in[k] += (x_in * w[:, None]).T @ x_in
                S_out_out[k] += wx_out.T @ x_out
                gamma_sum[k] += w.sum()

        suff = {
            "S_out_in": S_out_in,
            "S_in_in": S_in_in,
            "S_out_out": S_out_out,
            "gamma_sum": gamma_sum,
            "xi_sum": xi_sum,
            "start_sum": start_sum,
        }
        return total_ll, suff

    def _m_step(
        self,
        suff: dict,
        K: int,
        D: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """M-step: update parameters from sufficient statistics."""
        dim_in = D * self.n_lags + 1
        A = np.zeros((K, D, dim_in), dtype=np.float64)
        Q = np.zeros((K, D, D), dtype=np.float64)

        for k in range(K):
            n_k = suff["gamma_sum"][k]
            if n_k < 1e-10:
                # Dead state: keep identity-ish AR with large noise
                A[k, :, :D] = 0.9 * np.eye(D)
                Q[k] = np.eye(D)
                continue

            S_in = suff["S_in_in"][k] + _REG * np.eye(dim_in)
            A[k] = cho_solve(cho_factor(S_in), suff["S_out_in"][k].T).T

            # Q_k = (1/n_k) * (S_out_out - A S_out_in^T)
            Q[k] = (suff["S_out_out"][k] - A[k] @ suff["S_out_in"][k].T) / n_k
            Q[k] = 0.5 * (Q[k] + Q[k].T) + _REG * np.eye(D)

            # Ensure positive-definite
            eigvals = np.linalg.eigvalsh(Q[k])
            if eigvals.min() < _REG:
                Q[k] += (abs(eigvals.min()) + _REG) * np.eye(D)

        # Transition matrix with sticky prior
        xi = suff["xi_sum"] + 1e-10  # avoid zeros
        xi[np.diag_indices(K)] += self.sticky_weight
        trans = xi / xi.sum(axis=1, keepdims=True)
        log_trans = np.log(np.clip(trans, 1e-300, None))

        # Start probabilities
        start = suff["start_sum"] + 1e-10
        start /= start.sum()
        log_start = np.log(np.clip(start, 1e-300, None))

        return A, Q, log_trans, log_start


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no class state)
# ---------------------------------------------------------------------------


def _build_design(X: np.ndarray, nlags: int) -> np.ndarray:
    """Build the lagged design matrix for AR regression.

    Returns shape (T - nlags, D * nlags + 1).
    """
    T, D = X.shape
    T_eff = T - nlags
    dim_in = D * nlags + 1
    x_in = np.empty((T_eff, dim_in), dtype=np.float64)
    for lag in range(nlags):
        start = nlags - lag - 1
        x_in[:, lag * D : (lag + 1) * D] = X[start : start + T_eff]
    x_in[:, -1] = 1.0  # bias
    return x_in


def _cholesky_all(
    Q: np.ndarray,
) -> tuple[list, np.ndarray]:
    """Cholesky-factorise each (D, D) noise covariance matrix in *Q*.

    Returns a list of cho_factor tuples and log-determinant array.
    """
    K = Q.shape[0]
    cho_list: list = []
    logdets = np.empty(K, dtype=np.float64)
    for k in range(K):
        cf = cho_factor(Q[k])
        cho_list.append(cf)
        logdets[k] = 2.0 * np.sum(np.log(np.diag(cf[0])))
    return cho_list, logdets


def _ar_log_likelihoods(
    X: np.ndarray,
    A: np.ndarray,
    Q_cho: list,
    Q_logdet: np.ndarray,
    nlags: int,
    K: int,
) -> np.ndarray:
    """Compute AR emission log-likelihoods.

    Returns shape (T - nlags, K).
    """
    D = X.shape[1]
    x_in = _build_design(X, nlags)  # (T_eff, dim_in)
    x_out = X[nlags:]  # (T_eff, D)
    T_eff = x_out.shape[0]
    log_const = -0.5 * D * np.log(2.0 * np.pi)

    log_lik = np.empty((T_eff, K), dtype=np.float64)
    for k in range(K):
        resid = x_out - x_in @ A[k].T  # (T_eff, D)
        # Mahalanobis: resid @ Q^{-1} @ resid^T, per row
        solved = cho_solve(Q_cho[k], resid.T).T  # (T_eff, D)
        mahal = np.sum(resid * solved, axis=1)  # (T_eff,)
        log_lik[:, k] = log_const - 0.5 * Q_logdet[k] - 0.5 * mahal

    return log_lik


def _forward(
    log_lik: np.ndarray,
    log_trans: np.ndarray,
    log_start: np.ndarray,
) -> np.ndarray:
    """Forward pass (log-space).  Returns log_alpha, shape (T, K)."""
    T, K = log_lik.shape
    log_alpha = np.empty((T, K), dtype=np.float64)
    log_alpha[0] = log_start + log_lik[0]
    for t in range(1, T):
        # log_alpha[t, k] = logsumexp_j(log_alpha[t-1, j] + log_trans[j, k]) + log_lik[t, k]
        log_alpha[t] = logsumexp(
            log_alpha[t - 1, :, None] + log_trans, axis=0,
        ) + log_lik[t]
    return log_alpha


def _backward(
    log_lik: np.ndarray,
    log_trans: np.ndarray,
) -> np.ndarray:
    """Backward pass (log-space).  Returns log_beta, shape (T, K)."""
    T, K = log_lik.shape
    log_beta = np.empty((T, K), dtype=np.float64)
    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t, k] = logsumexp_j(log_trans[k, j] + log_lik[t+1, j] + log_beta[t+1, j])
        log_beta[t] = logsumexp(
            log_trans + log_lik[t + 1, None, :] + log_beta[t + 1, None, :],
            axis=1,
        )
    return log_beta
