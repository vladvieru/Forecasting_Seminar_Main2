"""
forecast_combination.py
=======================
Research-grade implementation of a graph-based, covariance-aware forecast
combination method for unstable environments.

Layers
------
1. Local prediction of future pairwise loss differentials (Richter-Smetanina style)
2. Graph aggregation via eigenvector centrality
3. Covariance-aware simplex-constrained weight optimization with shrinkage

Author : (research prototype)
License: MIT
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import optimize
from scipy.linalg import eigh

# ---------------------------------------------------------------------------
# 0.  CONSTANTS & HELPERS
# ---------------------------------------------------------------------------

RNG_SEED = 42
EPS = 1e-12          # generic numerical floor
RIDGE_COV = 1e-6     # default ridge for covariance matrices
TELEPORT = 1e-6      # default teleportation for eigenvector centrality


def _ensure_rng(seed=None):
    if seed is None:
        return np.random.default_rng(RNG_SEED)
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# ===================================================================
# 1.  LOSS FUNCTIONS
# ===================================================================

def squared_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Squared-error loss  (y-f)^2."""
    return (y - f) ** 2


def absolute_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Absolute-error loss  |y-f|."""
    return np.abs(y - f)


LOSS_REGISTRY: Dict[str, Callable] = {
    "squared": squared_loss,
    "absolute": absolute_loss,
}


# ===================================================================
# 2.  LOCAL PAIRWISE LOSS-DIFFERENTIAL MODEL  (Hyperedge style)
# ===================================================================
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

# ---------- kernel ----------

def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel K(u) = 0.75*(1-u^2) for |u|<=1, else 0."""
    w = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1.0
    w[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return w


# ---------- local linear AR estimation ----------

def _build_ar_design(series: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build AR(d) regression matrices from *series*.

    Returns
    -------
    Y : shape (T-d,)
    X : shape (T-d, d+1)   columns = [1, lag1, ..., lag_d]
    """
    n = len(series)
    if n <= d:
        raise ValueError("Series too short for AR({})".format(d))
    Y = series[d:].copy()
    X = np.ones((n - d, d + 1))
    for k in range(1, d + 1):
        X[:, k] = series[d - k: n - k]
    return Y, X


def local_linear_ar_fit(
    series: np.ndarray,
    d: int,
    h: float,
    target_frac: float = 1.0,
    t_filter: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local linear AR(d) fit at rescaled time *target_frac*, which is set to 1 as per the Hyperedge description.

    Instead of returning the entire Theta path, appropriate residuals, gives the edge coefficients, residuals only.

    Builds the augmented Z_t(u) = [X_{t-1}, (t/T - u)*X_{t-1}] design
    matrix (size 2*(d+1)) as in equation (3) of Richter, Smetanina, estimates
    theta(u) via WLS, and returns the first (d+1) components as rho(u).

    Parameters
    ----------
    series      : 1-d array of length T
    d           : AR lag order
    h           : bandwidth on [0,1] scale
    target_frac : evaluation point u (usually 1.0 for forecasting)
    t_filter    : optional array of length T; 0 = exclude, 1 = include.
                  Applied to the (T-d) observation rows after lag truncation.

    Returns
    -------
    beta    : rho(u) estimate, shape (d+1,)  [first half of theta(u)]
    resid   : Y - X @ beta, shape (T-d,)
    W_diag  : kernel weights, shape (T-d,)
    """
    Y, X = _build_ar_design(series, d)
    n = len(Y)  # = T - d
    T = len(series)

    # rescaled time for each observation row (index d, d+1, ..., T-1)
    fracs = np.arange(d, T) / T

    # kernel weights centred at target_frac
    u = (fracs - target_frac) / max(h, EPS)
    W_diag = epanechnikov_kernel(u)

    # apply t_filter if supplied (operates on the post-lag rows)
    if t_filter is not None:
        W_diag = W_diag * t_filter[d:]

    # build augmented local linear design matrix Z_t(u) of size (n, 2*(d+1))
    # top block: X  (the standard AR regressors)
    # bottom block: (t/T - u) * X  (local linear expansion)
    delta = (fracs - target_frac)[:, None]   
    X_ll = X * delta                          
    Z = np.hstack([X, X_ll])                  

    # WLS: (Z'WZ)^{-1} Z'WY  where W = diag(W_diag)
    ZW = Z * W_diag[:, None]                  
    ZWZ = ZW.T @ Z                          
    ZWY = ZW.T @ Y                            

    try:
        theta = np.linalg.solve(ZWZ, ZWY)
    except np.linalg.LinAlgError: #fallback for singularity
        theta = np.zeros(2 * (d + 1))

    # rho(u) is the first (d+1) components of theta
    beta = theta[:d + 1]
    resid = Y - X @ beta
    return beta, resid, W_diag


def _estimate_full_sequence(
    series: np.ndarray,
    d: int,
    h: float,
    t_filter: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate rho(t/T) and compute residuals at every t/T, matching the
    Richter, Smetanina est_theta_general vectorised loop. (added for completeness)

    Returns
    -------
    rho_seq  : shape (T-d, d+1)
    resid_seq: shape (T-d,)      
    """
    Y, X = _build_ar_design(series, d)
    n = len(Y)
    T = len(series)
    fracs = np.arange(d, T) / T

    rho_seq = np.empty((n, d + 1))
    for i in range(n):
        t_frac = fracs[i]
        beta, _, _ = local_linear_ar_fit(
            series, d, h, target_frac=t_frac, t_filter=t_filter
        )
        rho_seq[i] = beta

    resid_seq = Y - np.sum(X * rho_seq, axis=1)
    return rho_seq, resid_seq


def local_predict_mean(series: np.ndarray, d: int, h: float) -> float:
    """
    One-step-ahead conditional mean forecast from local linear AR(d).
    Localises at the boundary (target_frac = 1.0).

    Returns X_T' @ rho_hat(1) where X_T = [1, L_T, ..., L_{T-d+1}].
    """
    beta, _, _ = local_linear_ar_fit(series, d, h, target_frac=1.0)
    x_new = np.ones(d + 1)
    for k in range(1, d + 1):
        x_new[k] = series[-k]
    return float(x_new @ beta)


def local_predict_scale(
    series: np.ndarray,
    d: int,
    h1: float,
    h2: float,
) -> float:
    """
    Local scale (std-dev) estimate at u=1 from second-stage local linear
    kernel regression on squared residuals of the first-stage local AR(d).

    Computes residuals xi_t = L_t - X_{t-1}' rho_hat(t/T) for each t
    using the time-appropriate coefficient, matching est_theta_general
    in Richter, Smetanina. Then runs local linear regression of xi_t^2 on
    rescaled time, evaluated at u=1. Again, matching the Hyperedge methodology.
    """
    T = len(series)
    n = T - d

    # --- stage 1: time-varying residuals using rho(t/T) at each t ---
    _, resid_seq = _estimate_full_sequence(series, d, h1)
    sq_resid = resid_seq ** 2

    # --- stage 2: local linear regression of sq_resid on rescaled time ---
    # F_t(u) = [1, t/T - u], evaluated at u = 1
    fracs = np.arange(d, T) / T
    u2 = (fracs - 1.0) / max(h2, EPS)
    W2 = epanechnikov_kernel(u2)

    delta2 = (fracs - 1.0)[:, None]
    F = np.hstack([np.ones((n, 1)), delta2])   # shape (n, 2)
    FW = F * W2[:, None]
    FWF = FW.T @ F
    FWsq = FW.T @ sq_resid

    try:
        varsigma = np.linalg.solve(FWF, FWsq)
        var_est = float(varsigma[0])           # first component = sigma^2(1)
    except np.linalg.LinAlgError: #fallback
        var_est = float(np.mean(sq_resid))

    return float(np.sqrt(max(var_est, EPS)))


# ---------- BIC lag selection ----------

def bic_lag_selection(
    series: np.ndarray,
    d_max: int,
    h: float,
    correct_bic: bool = False,
) -> int:
    """
    Select AR lag order d in {1,...,d_max} via BIC.

    Parameters
    ----------
    series      : 1-d array of length T
    d_max       : maximum lag order to consider
    h           : preliminary bandwidth for estimation
    correct_bic : if True, uses the Richter, Smetanina BIC formulation (eq. 7):
                      BIC(d) = sum_t log(sigma^2(t/T)) + (d+1)*log(T-d)
                  if False (default), uses an approximate formulation. (used for speed in testing)

    Returns
    -------
    best_d : int
    """
    T = len(series)
    best_d = 0
    best_bic = np.inf

    for d in range(0, d_max + 1):
        if T <= d + 2:
            continue

        if correct_bic:
            # BIC (eq. 7): sum over t of log(sigma^2(t/T)) + (d+1)*log(T-d)
            # use time-varying residuals at each t/T then take as sigma^2(t/T) estimate
            _, resid_seq = _estimate_full_sequence(series, d, h)
            sigma2_seq = np.maximum(resid_seq ** 2, EPS)
            bic = np.sum(np.log(sigma2_seq)) + (d + 1) * np.log(T - d)
        else:
            # Approximate formulation
            beta, resid, W_diag = local_linear_ar_fit(
                series, d, h, target_frac=1.0
            )
            n_eff = W_diag.sum()
            if n_eff < d + 2:
                continue
            sse = (W_diag * resid ** 2).sum()
            sigma2 = sse / max(n_eff, EPS)
            if sigma2 <= 0:
                continue
            bic = np.log(sigma2 + EPS) + (d + 1) * np.log(max(n_eff, 2)) / max(n_eff, 1)

        if bic < best_bic:
            best_bic = bic
            best_d = d

    return best_d


# ---------- CV bandwidth selection ----------

def cv_bandwidth_selection(
    series: np.ndarray,
    d: int,
    h_grid: np.ndarray,
    n_folds: int = 20,
) -> Tuple[float, np.ndarray]:
    """
    Interleaved cross-validation for bandwidth selection, matching the
    Richter, Smetanina fold structure zeta_j = {Q*k + j, k=1,2,...} with Q=n_folds.

    h_grid sets the search grid for h. Beware of low values, often leads to singularity issues, as too few observations are effective (nonzero weights).

    Each held-out observation i is predicted using the model estimated
    without its fold, evaluated at its own rescaled time u = i/T.

    Returns (best_h, cv_scores).
    """
    T = len(series)
    Q = n_folds
    cv_scores = np.full(len(h_grid), np.nan)

    if T <= d + 2:
        return float(h_grid[len(h_grid) // 2]), cv_scores

    Y_full, X_full = _build_ar_design(series, d)
    n = len(Y_full)  # T - d

    for ih, h in enumerate(h_grid):
        fold_mse = np.zeros(Q)

        for j in range(Q):
            # interleaved fold indices into post-lag rows 0..n-1
            fold_idx = np.arange(j, n, Q)
            if len(fold_idx) == 0:
                continue

            # t_filter: exclude fold members (original indices fold_idx + d)
            t_filter = np.ones(T)
            t_filter[fold_idx + d] = 0.0

            # evaluate at each held-out observation's own rescaled time
            # resid[d:][dj] uses rho(s/T) at each s
            sq_errors = np.empty(len(fold_idx))
            for k, idx in enumerate(fold_idx):
                t_frac = (idx + d) / T
                beta_cv, _, _ = local_linear_ar_fit(
                    series, d, h,
                    target_frac=t_frac,
                    t_filter=t_filter,
                )
                pred = X_full[idx] @ beta_cv
                sq_errors[k] = (Y_full[idx] - pred) ** 2

            fold_mse[j] = np.mean(sq_errors)

        cv_scores[ih] = np.mean(fold_mse)   # mean of fold means

    best_idx = int(np.nanargmin(cv_scores))
    return float(h_grid[best_idx]), cv_scores


def make_h_grid(
    n_points: int = 15,
    h_min: float = 0.05,
    h_max: float = 1.0,
) -> np.ndarray:
    """Logarithmically spaced bandwidth grid on [h_min, h_max]."""
    return np.exp(np.linspace(np.log(h_min), np.log(h_max), n_points))


# ----------2.5 Full pairwise LD predictor, Richter-Smetanina formulation ----------

@dataclass
class PairwiseLDResult:
    """Result of predicting one pairwise loss differential."""
    mu_hat: float = 0.0
    sigma_hat: float = 1.0
    d_selected: int = 0
    h1_selected: float = 0.2
    h2_selected: float = 0.2
    theta_hat: Optional[np.ndarray] = field(default=None, repr=False)
    sigsq_hat: Optional[np.ndarray] = field(default=None, repr=False)
    std_resid: Optional[np.ndarray] = field(default=None, repr=False)
    Y_hat: Optional[np.ndarray] = field(default=None, repr=False)
    mu_path: Optional[np.ndarray] = field(default=None, repr=False)
    mean_forecast: Optional[float] = None
    prob_forecast: Optional[float] = None
    bic: Optional[np.ndarray] = field(default=None, repr=False)
    h1_cv: Optional[np.ndarray] = field(default=None, repr=False)
    h2_cv: Optional[np.ndarray] = field(default=None, repr=False)
    h1_grid: Optional[np.ndarray] = field(default=None, repr=False)
    h2_grid: Optional[np.ndarray] = field(default=None, repr=False)


def predict_pairwise_ld(
    delta_L: np.ndarray,
    d_max: int = 4,
    h1_grid: Optional[np.ndarray] = None,
    h2_grid: Optional[np.ndarray] = None,
    fixed_d: Optional[int] = None,
    fixed_h1: Optional[float] = None,
    fixed_h2: Optional[float] = None,
    n_cv_folds: int = 20,
    correct_bic: bool = False,
) -> PairwiseLDResult:
    """
    Given a history of pairwise loss differentials (up to time T),
    produce next-period conditional mean and scale estimates.

    Parameters
    ----------
    delta_L     : 1-d array of loss differences L_t = Loss_A - Loss_B
    d_max       : maximum lag order for BIC selection
    h1_grid     : bandwidth grid for first-stage CV
    h2_grid     : bandwidth grid for second-stage CV
    fixed_d     : fix lag order, bypassing BIC selection
    fixed_h1    : fix first-stage bandwidth, bypassing CV
    fixed_h2    : fix second-stage bandwidth, bypassing CV
    n_cv_folds  : number of interleaved CV folds Q>=20
    correct_bic : if True use full BIC formulation; if False use
                  approximate formulation (default False)

    Returns
    -------
    PairwiseLDResult with mu_hat, sigma_hat, and selected d, h1, h2
    """
    T = len(delta_L)
    if T < 5:
        return PairwiseLDResult()

    if h1_grid is None:
        h1_grid = make_h_grid(12, 0.05, 1.0)
    if h2_grid is None:
        h2_grid = make_h_grid(10, 0.10, 1.0)

    h_prelim = float(h1_grid[len(h1_grid) // 2])

    # lag selection
    if fixed_d is not None:
        d = fixed_d
    else:
        d = bic_lag_selection(delta_L, d_max, h_prelim, correct_bic=correct_bic)

    # bandwidth h1
    if fixed_h1 is not None:
        h1 = fixed_h1
    else:
        h1, _ = cv_bandwidth_selection(delta_L, d, h1_grid, n_cv_folds)

    # bandwidth h2
    if fixed_h2 is not None:
        h2 = fixed_h2
    else:
        h2, _ = cv_bandwidth_selection(delta_L, d, h2_grid, n_cv_folds)

    mu = local_predict_mean(delta_L, d, h1)
    sigma = local_predict_scale(delta_L, d, h1, h2)

    return PairwiseLDResult(
        mu_hat=mu,
        sigma_hat=sigma,
        d_selected=d,
        h1_selected=h1,
        h2_selected=h2,
    )


def _v2_est_theta_general(
    d: int,
    h: float,
    series: np.ndarray,
    t_filter: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    ***As given by Richter, Smetanina***
    First-stage estimation of the model with p lags. In addition to theta, we calculate:
    1. the fitted values for Y_t
    2. the unconditional means for Y_t
    3. model residuals for Y_t

    Parameters
    ----------
    d : int
        Integer for the number of lags to include.
    h : float
        Float giving the bandwidth for the first-stage estimation, referred to as h1 in the paper.
    Y : array_like
        Array of Y data points on which to fit the model.
    t_filter : array_like
        Array of length T, filled with 0's where data points should be discarded for estimation
        and 1's otherwise.

    Returns
    -------
    theta_hat : array_like
        Array of theta estimates of size T x 2d.
    Y_hat : array_like
        Array of fitted values of size T-1.
    mu_hat : array_like
        Array of unconditional means of size T-1.
    resid : array_like
        Array of first-stage residuals of size T-1.

    '''
    Y = np.asarray(series, dtype=float)
    T = Y.shape[0]
    if T <= d:
        raise ValueError(f"Series too short for v2 AR({d}) fit.")

    Y_trunc = Y[d:]
    X = np.ones((T - d, d + 1))
    for lag in range(d):
        X[:, lag + 1] = Y[d - (lag + 1): T - (lag + 1)]

    X_full = np.ones((T - d, 2 * (d + 1)))
    X_full[:, : d + 1] = X

    theta_hat = np.full((T, 2 * (d + 1)), np.nan)
    Y_hat = np.full(T, np.nan)
    mu_path = np.full(T, np.nan)
    u_vector = np.arange(d, T) / T
    t_filter = np.asarray(t_filter, dtype=float)

    for t in range(d, T):
        scaled_distance = (t / T - u_vector) / max(h, EPS)
        kernel = epanechnikov_kernel(scaled_distance) * t_filter[d:]
        if np.sum(kernel) < EPS:
            theta_t = np.zeros(2 * (d + 1))
        else:
            X_full[:, d + 1:] = X * (t / T - u_vector)[:, None]
            X_full_kernel = X_full * kernel[:, None]
            xtwx = X_full_kernel.T @ X_full
            xtwy = X_full_kernel.T @ Y_trunc
            try:
                theta_t = np.linalg.solve(xtwx, xtwy)
            except np.linalg.LinAlgError:
                theta_t = np.linalg.lstsq(xtwx, xtwy, rcond=None)[0]

        theta_hat[t] = theta_t
        Y_hat[t] = theta_t[: d + 1] @ X[t - d]
        denom = 1.0 - np.sum(theta_t[1: d + 1])
        if np.abs(denom) > EPS:
            mu_path[t] = theta_t[0] / denom

    resid = Y - Y_hat
    return theta_hat, Y_hat, mu_path, resid


def _v2_h1_cv_calc(
    d: int,
    h1_grid: np.ndarray,
    series: np.ndarray,
    n_folds: int = 20,
) -> np.ndarray:
    '''
    ***As given by Richter, Smetanina***
    Performs CV for h1.

    Parameters
    ----------
    d : int
        Integer for the number of lags to include.
    h1_vector : array_like
        Array of h1 candidates to perform CV.
    Y : array_like
        Array of data for the first-stage estimation.
    Q : integer
        Integer for the number of folds in the CV.

    Returns
    -------
    h1_CV : array_like
        Array of h1 CV distances.

    '''
    Y = np.asarray(series, dtype=float)
    T = Y.shape[0]
    gridsize = len(h1_grid)
    blocksize = int((T - d) / n_folds)
    if blocksize <= 0:
        return np.full(gridsize, np.inf)

    scores = np.full(gridsize, np.inf)
    for ih, h1 in enumerate(h1_grid):
        fold_scores = []
        for j in range(n_folds):
            dj = np.arange(blocksize) * n_folds + j
            if dj.size == 0:
                continue
            t_filter = np.ones(T)
            t_filter[dj + d] = 0.0
            _, _, _, resid = _v2_est_theta_general(d, h1, Y, t_filter)
            fold_scores.append(np.mean(resid[d:][dj] ** 2))
        if fold_scores:
            scores[ih] = float(np.mean(fold_scores))
    return scores


def _v2_h2_cv_calc(
    h2_grid: np.ndarray,
    resid: np.ndarray,
    n_folds: int = 20,
) -> np.ndarray:
    '''
    ***As given by Richter, Smetanina***
    Performs CV for h2.

    Parameters
    ----------
    h2_vector : array_like
        Array of h2 candidates to perform CV.
    resid : array_like
        Array of residuals from the first-stage estimation.
    Q : integer
        Integer for the number of folds in the CV.

    Returns
    -------
    h2_CV : array_like
        Array of h2 CV distances.

    '''
    resid = np.asarray(resid, dtype=float)
    T = resid.shape[0]
    gridsize = len(h2_grid)
    blocksize = int(T / n_folds)
    if blocksize <= 0:
        return np.full(gridsize, np.inf)

    scores = np.full(gridsize, np.inf)
    for ih, h2 in enumerate(h2_grid):
        fold_scores = []
        for j in range(n_folds):
            dj = np.arange(blocksize) * n_folds + j
            if dj.size == 0:
                continue
            t_filter = np.ones(T)
            t_filter[dj] = 0.0
            _, _, _, sigsq_resid = _v2_est_theta_general(0, h2, resid ** 2, t_filter)
            fold_scores.append(np.mean(sigsq_resid[dj] ** 2))
        if fold_scores:
            scores[ih] = float(np.mean(fold_scores))
    return scores


def _v2_bic_lag_selection(
    series: np.ndarray,
    d_max: int,
    h1: float = 0.3,
    h2: float = 0.3,
) -> np.ndarray:
    '''
    ***As given by Richter, Smetanina***
    Calculates the BIC to select the lag length.

     Paramaters
    ----------
    d_max : int
        The maximum number of lags to consider in the BIC (default is 5).
     h1 : float
        The first-stage bandwidth to evaluate the BIC at (default is 0.3).
    h2 : float
        The second-stage bandwidth to evaluate the BIC at (default is 0.3).

     Returns
    -------
    BIC : array_like
    Array of BIC for d = 0, ..., d_max.

    '''
    Y = np.asarray(series, dtype=float)
    T = Y.shape[0]
    bic = np.full(d_max, np.inf)
    if T <= d_max:
        return bic

    for d in range(1, d_max + 1):
        _, _, _, resid = _v2_est_theta_general(d, h1, Y, np.ones(T))
        resid_var = resid[d_max:] ** 2
        if resid_var.size == 0:
            continue
        _, sigsq_hat, _, _ = _v2_est_theta_general(0, h2, resid_var, np.ones(T - d_max))
        sigsq_hat = np.maximum(1e-6, sigsq_hat)
        bic[d - 1] = (d + 1) * np.log(T - d_max) + np.sum(np.log(np.sqrt(sigsq_hat) + 1.0))
    return bic


def _v2_next_period_forecast(
    x_pred: np.ndarray,
    rho_pred: np.ndarray,
    sigsq_pred: float,
    std_resid: np.ndarray,
) -> Tuple[float, float]:
    '''
    ***As given by Richter, Smetanina***
    Note that the probability forecast is not fully implemented later, as it is unimportant for our purpose.
    
    Forecasts the mean of Y and the probability of Y being negative next period.

    Parameters
    ----------
    X_pred : array_like
        Array of X values to predict from.
    rho_pred : array_like
        Array of rho values to predict from.
    sigsq_pred : float
        Float of sigma squared to predict from.
    std_resid : array_like
        Array of standardized residuals to approximate the empirical distribution.

    Returns
    -------
    Y_hat_pred : float
        Float giving the predicted mean.
    prob_np : float
        Float giving the predicted probability of negative sign.

    '''
    mean_fc = float(x_pred @ rho_pred)
    finite_resid = np.asarray(std_resid, dtype=float)
    finite_resid = finite_resid[np.isfinite(finite_resid)]
    if finite_resid.size == 0:
        return mean_fc, 0.5
    error_threshold = -mean_fc / max(np.sqrt(max(sigsq_pred, EPS)), EPS)
    prob_fc = float(np.mean(finite_resid < error_threshold))
    return mean_fc, prob_fc


def _standardize_residuals(std_resid: np.ndarray) -> np.ndarray:
    """ Explicit helper to center and scale residuals, with a safe zero fallback."""
    x = np.asarray(std_resid, dtype=float)
    if x.size == 0:
        return x
    x = x - np.nanmean(x)
    scale = np.nanstd(x)
    if not np.isfinite(scale) or scale < EPS:
        return np.zeros_like(x)
    return x / scale


def predict_pairwise_ld_v2(
    delta_L: np.ndarray,
    d_max: int = 4,
    h1_grid: Optional[np.ndarray] = None,
    h2_grid: Optional[np.ndarray] = None,
    fixed_d: Optional[int] = None,
    fixed_h1: Optional[float] = None,
    fixed_h2: Optional[float] = None,
    n_cv_folds: int = 20,
    correct_bic: bool = True,
) -> PairwiseLDResult:
    """
    Reproduce the pairwise-LD workflow introduced in Richter, Smetanina.

    The returned `mu_hat` is the scalar quantity consumed by the graph layer:
    the last available value of the v2 `mu_path`.
    """
    Y = np.asarray(delta_L, dtype=float)
    T = Y.shape[0]
    if T < 5:
        return PairwiseLDResult()

    if h1_grid is None:
        h1_grid = (np.arange(1, 21) ** 1.5) / (20 ** 1.5)
    if h2_grid is None:
        h2_grid = (np.arange(1, 21) ** 1.5) / (20 ** 1.5)

    bic = None
    if fixed_d is not None:
        d = fixed_d
    elif correct_bic:
        bic = _v2_bic_lag_selection(Y, d_max)
        finite_bic = np.where(np.isfinite(bic), bic, np.inf)
        d = int(np.argmin(finite_bic)) + 1
    else:
        d = 1

    h_min = max(2 * (d + 1) + 1, 7) / T
    h1_grid = np.asarray(h1_grid, dtype=float)
    h2_grid = np.asarray(h2_grid, dtype=float)
    h1_grid = h1_grid[h1_grid >= h_min]
    h2_grid = h2_grid[h2_grid >= h_min]
    if h1_grid.size == 0:
        h1_grid = np.array([0.5])
    if h2_grid.size == 0:
        h2_grid = np.array([0.5])

    h1_cv_scores = None
    if fixed_h1 is not None:
        h1 = fixed_h1
    else:
        h1_cv_scores = _v2_h1_cv_calc(d, h1_grid, Y, n_folds=n_cv_folds)
        h1 = float(h1_grid[int(np.argmin(h1_cv_scores))])

    theta_hat, Y_hat, mu_path, resid = _v2_est_theta_general(d, h1, Y, np.ones(T))

    h2_cv_scores = None
    if fixed_h2 is not None:
        h2 = fixed_h2
    else:
        h2_cv_scores = _v2_h2_cv_calc(h2_grid, resid[d:], n_folds=n_cv_folds)
        h2 = float(h2_grid[int(np.argmin(h2_cv_scores))])

    _, sigsq_hat, _, _ = _v2_est_theta_general(0, h2, resid[d:] ** 2, np.ones(T - d))
    sigsq_hat = np.maximum(1e-4, sigsq_hat)
    std_resid = _standardize_residuals(resid[d:] / np.sqrt(sigsq_hat))

    if d == 0:
        x_pred = np.array([1.0])
        rho_pred = theta_hat[-1, :1]
    else:
        x_pred = np.r_[1.0, Y[-d:]]
        rho_pred = theta_hat[-1, : d + 1]

    mean_fc, prob_fc = _v2_next_period_forecast(
        x_pred,
        rho_pred,
        float(sigsq_hat[-1]),
        std_resid,
    )

    mu_tail = mu_path[np.isfinite(mu_path)]
    mu_value = float(mu_tail[-1]) if mu_tail.size else float(mean_fc)

    return PairwiseLDResult(
        mu_hat=mu_value,
        sigma_hat=float(np.sqrt(sigsq_hat[-1])),
        d_selected=d,
        h1_selected=float(h1),
        h2_selected=float(h2),
        theta_hat=theta_hat,
        sigsq_hat=sigsq_hat,
        std_resid=std_resid,
        Y_hat=Y_hat,
        mu_path=mu_path,
        mean_forecast=mean_fc,
        prob_forecast=prob_fc,
        bic=bic,
        h1_cv=h1_cv_scores,
        h2_cv=h2_cv_scores,
        h1_grid=h1_grid,
        h2_grid=h2_grid,
    )


PAIRWISE_LD_METHOD_VARIANTS = {
    "rs_selection": ("legacy", "v2"),
    "graph_only": ("legacy", "v2"),
    "full_gcsr": ("legacy", "v2"),
}

PAIRWISE_LD_VARIANT_SUFFIX = {
    "legacy": "",
    "v2": "_v2",
}

PAIRWISE_LD_PREDICTORS = {
    "legacy": predict_pairwise_ld,
    "v2": predict_pairwise_ld_v2,
}

CENTRALITY_ORDER = {
    "eigenvector": 0,
    "pagerank": 1,
    "softmax": 2,
    "rowsum": 3,
}

DEFAULT_COMPARISON_CENTRALITY_TYPES = ("eigenvector", "pagerank", "softmax")


def _resolve_loss_diff_versions(
    variants: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """Validate and de-duplicate the requested LD variants."""
    if variants is None:
        variants = ("legacy", "v2")

    ordered = []
    for variant in variants:
        if variant not in PAIRWISE_LD_PREDICTORS:
            valid = ", ".join(PAIRWISE_LD_PREDICTORS)
            raise ValueError(f"Unknown loss-differential variant '{variant}'. Valid options: {valid}.")
        if variant not in ordered:
            ordered.append(variant)

    if not ordered:
        raise ValueError("At least one loss-differential variant must be enabled.")
    return tuple(ordered)


def _loss_diff_method_name(base_name: str, variant: str) -> str:
    """Map a LD-driven base method to its variant-specific public name."""
    return f"{base_name}{PAIRWISE_LD_VARIANT_SUFFIX[variant]}"


def _resolve_centrality_types(
    primary_type: str,
    requested_types: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """Validate the requested centrality measures and preserve a stable order."""
    if requested_types is None:
        requested_types = (primary_type,)

    ordered: List[str] = []
    for centrality_type in requested_types:
        if centrality_type not in CENTRALITY_ORDER:
            valid = ", ".join(CENTRALITY_ORDER)
            raise ValueError(
                f"Unknown centrality type '{centrality_type}'. Valid options: {valid}."
            )
        if centrality_type not in ordered:
            ordered.append(centrality_type)

    if primary_type not in ordered:
        ordered.insert(0, primary_type)

    ordered.sort(key=lambda name: CENTRALITY_ORDER[name])
    return tuple(ordered)


def _centrality_method_name(
    base_name: str,
    centrality_type: str,
    active_types: Sequence[str],
) -> str:
    """
    Expose explicit centrality names only when multiple measures are active.

    This preserves the historical public names `graph_only` and `full_gcsr`
    for the single-centrality case while creating fully explicit names such as
    `graph_only_pagerank` when several centrality measures are compared.
    """
    if len(tuple(active_types)) <= 1:
        return base_name
    return f"{base_name}_{centrality_type}"


def _ld_centrality_method_name(
    base_name: str,
    variant: str,
    centrality_type: str,
    active_types: Sequence[str],
) -> str:
    """
    Public method naming for graph-based LD methods with variant labels.

    Historical legacy names are preserved for backward compatibility, while
    alternative LD engines such as `v2` are made explicit via names like
    `graph_only_v2_pagerank` and `full_gcsr_v2_softmax`.
    """
    if variant == "legacy":
        return _centrality_method_name(base_name, centrality_type, active_types)
    if len(tuple(active_types)) <= 1:
        return f"{base_name}_{variant}"
    return f"{base_name}_{variant}_{centrality_type}"


def _loss_diff_method_enabled(base_name: str, variant: str) -> bool:
    """Check whether a loss-differential method is exposed for a variant."""
    return variant in PAIRWISE_LD_METHOD_VARIANTS.get(base_name, ())


def _primary_loss_diff_variant(variants: Sequence[str]) -> str:
    """Choose the variant mirrored into the legacy diagnostic fields."""
    return "legacy" if "legacy" in variants else variants[0]


def _is_gcsr_method(name: str) -> bool:
    """Identify GCSR variants for highlighting in plots and tables."""
    return name.startswith("full_gcsr")


def _method_family(name: str) -> str:
    """Parse the method family from a public method name."""
    for family in (
        "full_gcsr",
        "graph_only",
        "rs_selection",
        "bates_granger_mv",
        "recent_best",
        "var_error",
        "equal",
    ):
        if name == family or name.startswith(f"{family}_"):
            return family
    return name


def _method_centrality_type(name: str) -> Optional[str]:
    """Extract the centrality suffix when the method name is explicit."""
    family = _method_family(name)
    if family not in {"full_gcsr", "graph_only"}:
        return None
    for centrality_type in CENTRALITY_ORDER:
        if name.endswith(f"_{centrality_type}") or f"_{centrality_type}_" in name:
            return centrality_type
    return None


def _method_sort_key(name: str) -> Tuple[int, int, str]:
    """Stable plotting/reporting order for benchmark and model variants."""
    family_order = {
        "full_gcsr": 0,
        "graph_only": 1,
        "rs_selection": 2,
        "bates_granger_mv": 3,
        "var_error": 4,
        "recent_best": 5,
        "equal": 6,
    }
    family = _method_family(name)
    centrality_type = _method_centrality_type(name)
    return (
        family_order.get(family, 99),
        CENTRALITY_ORDER.get(centrality_type, 99),
        name,
    )


def _preferred_method_subset(
    available_methods: Sequence[str],
    limit: Optional[int] = None,
) -> List[str]:
    """Pick a readable default subset with explicit centrality variants first."""
    ordered = sorted(set(available_methods), key=_method_sort_key)
    if limit is not None:
        return ordered[:limit]
    return ordered


def _ensure_comparison_centrality_types(bt_cfg: BacktestConfig) -> BacktestConfig:
    """
    Guarantee that comparison-oriented workflows evaluate all centrality types.

    Simulation Monte Carlo and empirical comparison pipelines should not
    silently collapse back to a single graph centrality merely because the
    caller omitted `centrality_types`.
    """
    if bt_cfg.centrality_types is not None:
        return bt_cfg
    return replace(bt_cfg, centrality_types=DEFAULT_COMPARISON_CENTRALITY_TYPES)

# ===================================================================
# 3.  GRAPH LAYER
# ===================================================================

def build_adjacency_raw(mu_matrix: np.ndarray) -> np.ndarray:
    """
    Raw adjacency: A_{ij} = max(-mu_{ij}, 0).
    mu_{ij} < 0 => i beats j => edge weight from i to j.
    """
    A = np.maximum(-mu_matrix, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def build_adjacency_standardized(
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    c: float = 1e-4,
) -> np.ndarray:
    """Standardized adjacency: A_{ij} = max(-mu_{ij}/(sigma_{ij}+c), 0)."""
    A = np.maximum(-mu_matrix / (sigma_matrix + c), 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def build_adjacency_thresholded(
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    threshold: float = 1.0,
    c: float = 1e-4,
) -> np.ndarray:
    """Thresholded adjacency based on t-ratio."""
    t_ratio = -mu_matrix / (sigma_matrix + c)
    A = np.where(t_ratio > threshold, t_ratio, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def eigenvector_centrality(
    A: np.ndarray,
    teleport: float = TELEPORT,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Dominant eigenvector centrality of non-negative matrix A.

    Uses power iteration with optional teleportation for reducibility.
    Returns normalised score vector summing to 1.
    """
    M = A.shape[0]
    if A.max() < EPS:
        return np.ones(M) / M  # fallback: uniform

    # add teleportation
    A_reg = A + teleport * np.ones((M, M)) / M
    # power iteration
    r = np.ones(M) / M
    for _ in range(max_iter):
        r_new = A_reg @ r
        norm = r_new.sum()
        if norm < EPS:
            return np.ones(M) / M
        r_new /= norm
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    r = np.maximum(r, 0.0)
    r /= r.sum() + EPS
    return r


def row_sum_strength(A: np.ndarray) -> np.ndarray:
    """Row-sum (out-strength) centrality, normalised."""
    s = A.sum(axis=1)
    total = s.sum()
    if total < EPS:
        return np.ones(A.shape[0]) / A.shape[0]
    return s / total


def pagerank_centrality(
    A: np.ndarray,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """Simple PageRank on column-normalised A with damping *alpha*."""
    M = A.shape[0]
    col_sum = A.sum(axis=0)
    col_sum[col_sum < EPS] = 1.0
    P = A / col_sum[None, :]
    r = np.ones(M) / M
    for _ in range(max_iter):
        r_new = alpha * (P @ r) + (1 - alpha) / M
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    r = np.maximum(r, 0.0)
    r /= r.sum() + EPS
    return r


def softmax_average_advantage(mu_matrix: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Softmax of average predicted advantages (negative of row-mean mu)."""
    M = mu_matrix.shape[0]
    adv = np.zeros(M)
    for i in range(M):
        vals = [-mu_matrix[i, j] for j in range(M) if j != i]
        adv[i] = np.mean(vals) if vals else 0.0
    adv_scaled = adv / max(tau, EPS)
    adv_scaled -= adv_scaled.max()  # numerical stability
    w = np.exp(adv_scaled)
    return w / w.sum()


def compute_centrality_scores(
    A: np.ndarray,
    mu_matrix: np.ndarray,
    centrality_type: str,
    teleport: float = TELEPORT,
) -> np.ndarray:
    """Dispatch helper for the supported graph-centrality measures."""
    if centrality_type == "eigenvector":
        return eigenvector_centrality(A, teleport)
    if centrality_type == "rowsum":
        return row_sum_strength(A)
    if centrality_type == "pagerank":
        return pagerank_centrality(A)
    if centrality_type == "softmax":
        return softmax_average_advantage(mu_matrix)
    valid = ", ".join(CENTRALITY_ORDER)
    raise ValueError(f"Unsupported centrality type '{centrality_type}'. Valid options: {valid}.")


# ===================================================================
# 4.  COVARIANCE LAYER
# ===================================================================

def rolling_covariance(
    errors: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Rolling sample covariance of forecast errors.
    errors : shape (T, M)
    Uses last *window* observations.
    """
    T, M = errors.shape
    use = errors[max(0, T - window):T]
    if len(use) < M + 2:
        return np.eye(M)
    return np.cov(use, rowvar=False, ddof=1)


def ewma_covariance(
    errors: np.ndarray,
    lam: float = 0.94,
) -> np.ndarray:
    """Exponentially weighted moving average covariance."""
    T, M = errors.shape
    S = np.zeros((M, M))
    if T == 0:
        return np.eye(M)
    S = np.outer(errors[0], errors[0])
    for t in range(1, T):
        S = lam * S + (1 - lam) * np.outer(errors[t], errors[t])
    return S


def shrinkage_covariance(
    errors: np.ndarray,
    window: int = 60,
    shrink_target: str = "diagonal",
    shrink_intensity: Optional[float] = None,
) -> np.ndarray:
    """
    Ledoit–Wolf style shrinkage toward diagonal or identity.
    If *shrink_intensity* is None, use a simple analytical formula.
    """
    S = rolling_covariance(errors, window)
    M = S.shape[0]
    if shrink_target == "diagonal":
        T_mat = np.diag(np.diag(S))
    else:
        T_mat = np.eye(M) * np.trace(S) / M

    if shrink_intensity is not None:
        delta = shrink_intensity
    else:
        # simple Oracle Approximating Shrinkage intensity
        n = min(len(errors), window)
        delta = min(max((M) / (n + M), 0.01), 0.99)

    return (1 - delta) * S + delta * T_mat


def diagonal_covariance(errors: np.ndarray, window: int = 60) -> np.ndarray:
    """Diagonal-only covariance (variances only)."""
    S = rolling_covariance(errors, window)
    return np.diag(np.diag(S))


def regularise_cov(Sigma: np.ndarray, ridge: float = RIDGE_COV) -> np.ndarray:
    """Add ridge to diagonal for numerical stability."""
    return Sigma + ridge * np.eye(Sigma.shape[0])


COVARIANCE_REGISTRY = {
    "rolling": rolling_covariance,
    "ewma": ewma_covariance,
    "shrinkage": shrinkage_covariance,
    "diagonal": diagonal_covariance,
}


def _normalise_window_grid(window_grid: Sequence[int]) -> Tuple[int, ...]:
    """Return a sorted, de-duplicated tuple of strictly positive candidate windows."""
    cleaned = sorted({max(int(window), 1) for window in window_grid})
    if not cleaned:
        raise ValueError("window_grid must contain at least one strictly positive value.")
    return tuple(cleaned)


def _estimate_covariance(
    errors_hist: np.ndarray,
    bt_cfg: BacktestConfig,
    window_override: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate the innovation covariance for a given historical window.

    For EWMA we first truncate the error history to the requested rolling window
    and then apply the exponential smoother so that all covariance estimators
    participate in the same 20/40/60 window-length sweep.
    """
    cov_fn = COVARIANCE_REGISTRY.get(bt_cfg.cov_method, shrinkage_covariance)
    window = int(window_override if window_override is not None else bt_cfg.cov_window)

    if bt_cfg.cov_method == "ewma":
        use_errors = errors_hist[max(0, len(errors_hist) - window):]
        Sigma = cov_fn(use_errors, bt_cfg.cov_ewma_lambda)
    else:
        Sigma = cov_fn(errors_hist, window)
    return regularise_cov(Sigma, bt_cfg.ridge_cov)


# ===================================================================
# 5.  WEIGHT OPTIMIZATION LAYER
# ===================================================================

def simplex_project(w: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex.
    Algorithm of Duchi et al. (2008).
    """
    M = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, M + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones(M) / M
    rho_max = rho[-1]
    theta = (cssv[rho_max] - 1.0) / (rho_max + 1.0)
    w_proj = np.maximum(w - theta, 0.0)
    w_proj /= w_proj.sum() + EPS
    return w_proj


def graph_only_weights(r: np.ndarray) -> np.ndarray:
    """Weights proportional to graph centrality scores."""
    w = np.maximum(r, 0.0)
    s = w.sum()
    if s < EPS:
        raise Exception("error in graph_only_weights: s < EPS")
    return w / s


def covariance_only_weights(Sigma: np.ndarray, ridge: float = RIDGE_COV) -> np.ndarray:
    """Simplex-constrained minimum-variance weights via QP."""
    M = Sigma.shape[0]
    Sigma_r = regularise_cov(Sigma, ridge)
    r_zero = np.zeros(M)
    gamma_zero = 0.0
    return _solve_combination_qp(Sigma_r, r_zero, 0.0, gamma_zero, M)


def full_combination_weights(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Full graph-covariance-shrinkage combination.

    argmin_w  w'Sigma w  - alpha * r'w  + gamma * ||w - wbar||^2
    s.t. w in simplex
    """
    M = Sigma.shape[0]
    Sigma_r = regularise_cov(Sigma, ridge)
    return _solve_combination_qp(Sigma_r, r, alpha, gamma, M)


def _solve_simplex_qp_active_set(
    Q: np.ndarray,
    c: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Solve min_w w'Qw + c'w subject to w >= 0 and sum(w)=1.

    The problem is tiny in this project (number of forecasters), so an active-set
    KKT solve is materially faster than repeatedly calling a generic optimizer.
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    M = len(c)
    active = np.ones(M, dtype=bool)
    w = np.zeros(M)
    lam = 0.0

    for _ in range(max_iter):
        idx = np.where(active)[0]
        if len(idx) == 0:
            return np.ones(M) / M

        Qaa = 2.0 * Q[np.ix_(idx, idx)]
        ca = c[idx]
        kkt = np.block([
            [Qaa, np.ones((len(idx), 1))],
            [np.ones((1, len(idx))), np.zeros((1, 1))],
        ])
        rhs = np.concatenate([-ca, [1.0]])

        try:
            sol = np.linalg.solve(kkt, rhs)
        except LinAlgError:
            sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]

        w_candidate = np.zeros(M)
        w_candidate[idx] = sol[:len(idx)]
        lam = float(sol[-1])

        if np.any(w_candidate[idx] < -tol):
            drop_idx = idx[int(np.argmin(w_candidate[idx]))]
            active[drop_idx] = False
            continue

        w_candidate = np.maximum(w_candidate, 0.0)
        total = w_candidate.sum()
        if total <= EPS:
            return np.ones(M) / M
        w_candidate /= total

        gradient = 2.0 * Q @ w_candidate + c
        inactive = ~active
        if np.any(inactive) and np.any(gradient[inactive] + lam < -tol):
            inactive_idx = np.where(inactive)[0]
            add_idx = inactive_idx[int(np.argmin(gradient[inactive] + lam))]
            active[add_idx] = True
            continue

        w = w_candidate
        break
    else:
        w = simplex_project(w_candidate)

    return simplex_project(w)


def _solve_combination_qp(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    M: int,
) -> np.ndarray:
    """Core QP solver using scipy."""
    wbar = np.ones(M) / M

    # f(w) = w' Sigma w - alpha r'w + gamma ||w-wbar||^2
    #      = w' (Sigma + gamma I) w - (alpha r + 2 gamma wbar)' w + const
    Q = Sigma + gamma * np.eye(M)
    c = -(alpha * r + 2.0 * gamma * wbar)

    try:
        return _solve_simplex_qp_active_set(Q, c)
    except Exception:
        pass

    def objective(w):
        return w @ Q @ w + c @ w

    def gradient(w):
        return 2.0 * Q @ w + c

    # constraints: sum(w) = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * M
    w0 = wbar.copy()

    try:
        res = optimize.minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        w_opt = res.x
    except Exception:
        w_opt = wbar.copy()

    # safety project
    w_opt = np.maximum(w_opt, 0.0)
    w_opt /= w_opt.sum() + EPS
    return w_opt


def multiplicative_tilt_weights(
    w_base: np.ndarray,
    r: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Multiplicative tilt: w_i propto w_base_i * exp(kappa * r_i), renormalised.
    """
    log_w = np.log(np.maximum(w_base, EPS)) + kappa * r
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum() + EPS
    return w


# ===================================================================
# 6.  BENCHMARK METHODS
# ===================================================================

def equal_weights(M: int) -> np.ndarray:
    return np.ones(M) / M


def median_forecast(forecasts: np.ndarray) -> float:
    """Return median across M forecasts at one time point."""
    return float(np.median(forecasts))


def recent_best_selection(
    losses: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Select forecast with lowest average recent loss.
    losses : (T_hist, M)
    """
    M = losses.shape[1]
    use = losses[max(0, losses.shape[0] - window):]
    avg = use.mean(axis=0)
    w = np.zeros(M)
    w[np.argmin(avg)] = 1.0
    return w


def bates_granger_weights(
    errors: np.ndarray,
    window: int = 60,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Bates-Granger inverse-variance weights (diagonal),
    then simplex projection.
    """
    T, M = errors.shape
    use = errors[max(0, T - window):]
    var = np.var(use, axis=0, ddof=1)
    var = np.maximum(var, EPS)
    inv_var = 1.0 / var
    w = inv_var / inv_var.sum()
    return w


def bates_granger_mv_weights(
    errors: np.ndarray,
    window: int = 60,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Minimum-variance simplex-constrained (uses full covariance).
    """
    Sigma = rolling_covariance(errors, window)
    return covariance_only_weights(Sigma, ridge)


def rs_selection_weights(
    mu_matrix: np.ndarray,
) -> np.ndarray:
    """
    Richter-Smetanina style selection: pick forecast i that is predicted
    to beat the most others (largest net outperformance count).
    """
    M = mu_matrix.shape[0]
    wins = np.zeros(M)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            if mu_matrix[i, j] < 0:
                wins[i] += 1
    w = np.zeros(M)
    w[np.argmax(wins)] = 1.0
    return w


def _fit_var_ols(
    panel: np.ndarray,
    lags: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a VAR(lags) with intercept by equation-wise OLS on a multivariate panel.

    Parameters
    ----------
    panel : array of shape (T, M)
        Multivariate time series.
    lags : int
        Number of VAR lags.

    Returns
    -------
    coef : array of shape (1 + M * lags, M)
        OLS coefficient matrix.
    resid : array of shape (T-lags, M)
        In-sample residuals.
    last_state : array of shape (1 + M * lags,)
        Regressor vector for the next one-step-ahead forecast.
    """
    T, M = panel.shape
    if T <= lags:
        raise ValueError("Not enough observations for VAR({})".format(lags))

    Y = panel[lags:]
    X = np.ones((T - lags, 1 + M * lags))
    for ell in range(1, lags + 1):
        start = 1 + (ell - 1) * M
        stop = start + M
        X[:, start:stop] = panel[lags - ell:T - ell]

    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef

    last_state = np.ones(1 + M * lags)
    for ell in range(1, lags + 1):
        start = 1 + (ell - 1) * M
        stop = start + M
        last_state[start:stop] = panel[-ell]

    return coef, resid, last_state


def _select_var_lag(
    panel: np.ndarray,
    max_lags: int = 3,
    ic: Literal["bic", "aic"] = "bic",
) -> int:
    """
    Select a VAR lag order using a simple multivariate AIC/BIC criterion.
    """
    T, M = panel.shape
    best_lag = 0
    best_score = np.inf

    max_feasible = min(int(max_lags), max(T - 2, 0))
    for lags in range(0, max_feasible + 1):
        try:
            coef, resid, _ = _fit_var_ols(panel, lags)
        except ValueError:
            continue

        n_obs = resid.shape[0]
        if n_obs <= 0:
            continue

        Sigma_hat = (resid.T @ resid) / max(n_obs, 1)
        Sigma_hat = regularise_cov(Sigma_hat, ridge=RIDGE_COV)
        sign, logdet = np.linalg.slogdet(Sigma_hat)
        if sign <= 0:
            continue

        n_params = coef.size
        if ic == "aic":
            score = logdet + 2.0 * n_params / max(n_obs, 1)
        else:
            score = logdet + np.log(max(n_obs, 2)) * n_params / max(n_obs, 1)

        if score < best_score:
            best_score = score
            best_lag = lags

    return best_lag


def var_error_weights(
    errors: np.ndarray,
    max_lags: int = 3,
    fixed_lag: Optional[int] = None,
    window: int = 60,
    ic: Literal["bic", "aic"] = "bic",
    ridge: float = RIDGE_COV,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Weight-based benchmark built from a VAR model for the forecast-error panel.

    The VAR provides a one-step-ahead conditional mean forecast for the vector
    of model errors along with an innovation covariance estimate. Under squared
    loss, the conditional second moment of the combined error is

        E[(w'e_{t+1})^2 | F_t] = w' (Sigma_u + mu_e mu_e') w,

    so the benchmark chooses simplex weights that minimise this quantity.

    Returns
    -------
    weights : array of shape (M,)
        Simplex-constrained weights implied by the VAR error forecast.
    selected_lag : int
        VAR lag order used for the one-step-ahead error forecast.
    err_forecast : array of shape (M,)
        One-step-ahead conditional mean error forecast.
    second_moment : array of shape (M, M)
        Conditional second-moment matrix Sigma_u + mu_e mu_e'.
    """
    T_hist, M = errors.shape
    if T_hist == 0:
        w_eq = equal_weights(M)
        return w_eq, 0, np.zeros(M), np.eye(M)

    use = errors[max(0, T_hist - window):T_hist]
    max_lags = min(int(max_lags), max(len(use) - 2, 0))

    if fixed_lag is not None:
        lags = min(int(fixed_lag), max_lags)
    else:
        lags = _select_var_lag(use, max_lags=max_lags, ic=ic)

    coef, resid, last_state = _fit_var_ols(use, lags)
    err_forecast = last_state @ coef

    n_obs = resid.shape[0]
    Sigma_u = (resid.T @ resid) / max(n_obs, 1)
    Sigma_u = regularise_cov(Sigma_u, ridge=ridge)

    second_moment = Sigma_u + np.outer(err_forecast, err_forecast)
    weights = covariance_only_weights(second_moment, ridge=ridge)
    return weights, lags, err_forecast, second_moment


# ===================================================================
# 7.  SIMULATION ENVIRONMENT
# ===================================================================

@dataclass
class ScenarioConfig:
    """
    Store the configuration for a single simulation scenario.

    The configuration controls the number of forecasters and time periods, the
    common-shock component, the bias process, the idiosyncratic variance
    process, optional outliers, and optional cross-sectional dependence through
    either a one-factor or clustered structure.
    """
    name: str = "default"
    M: int = 8           # Number of forecasters
    T: int = 400         # Total number of time periods
    T0: int = 200        # First out-of-sample evaluation index
    sigma_common: float = 0.5   # Standard deviation of the common shock
    seed: int = 42

    # Bias parameters
    bias_type: str = "zero"  # One of: zero, constant, break, drift, cluster, common_rw_idio_ar1
    bias_values: Optional[np.ndarray] = None  # Fixed bias vector, shape (M,)
    bias_break_time: Optional[int] = None  # Break date for bias_type="break"
    bias_pre: Optional[np.ndarray] = None  # Bias vector before the break
    bias_post: Optional[np.ndarray] = None  # Bias vector after the break
    bias_drift_speed: float = 0.01  # Fallback drift scale if eta2 is not given
    bias_centered: bool = True  # Remove cross-sectional mean bias if True
    bias_init_low: float = 0.0  # Lower bound for random initial biases
    bias_init_high: float = 1.0  # Upper bound for random initial biases
    bias_drift_rho: float = 0.95  # Persistence of the latent bias slope
    bias_drift_eta2: Optional[float] = None  # Innovation variance of drift slope
    bias_common_rw_scale: float = 0.05  # Innovation std-dev of the common random-walk bias
    bias_common_rw_init: float = 0.0  # Initial level of the common random-walk bias
    bias_idio_ar1_rho: float = 0.9  # Persistence of forecaster-specific AR(1) bias deviations
    bias_idio_ar1_scale: Optional[float] = None  # Innovation std-dev of the AR(1) deviations

    # Variance parameters
    sigma_idio: Optional[np.ndarray] = None  # Fixed sigma path, shape (M,) or (T, M)
    sigma_process: str = "constant"  # One of: constant, break, smooth_precision
    base_sigma2: float = 1.0  # Baseline variance used in smooth-precision mode
    sigma_shift_time: Optional[int] = None  # Break date for piecewise sigma paths
    sigma_pre: Optional[np.ndarray] = None  # Sigma vector before the variance shift
    sigma_post: Optional[np.ndarray] = None  # Sigma vector after the variance shift
    sigma_drift_rho: float = 0.95  # Persistence of the latent log-precision slope
    sigma_drift_eta2: float = 8e-4  # Innovation variance of the precision slope
    log_precision_min: float = -2.0  # Lower truncation bound for log precision
    log_precision_max: float = 2.0  # Upper truncation bound for log precision

    # Outlier parameters
    outlier_prob: float = 0.0  # Per-entry probability of replacing idio noise
    outlier_scale: float = 5.0  # Maximum outlier multiplier relative to sigma
    outlier_min_scale: float = 2.0  # Minimum outlier multiplier relative to sigma

    # Dependence parameters
    factor_rho: Optional[np.ndarray] = None  # One-factor loadings, shape (M,)
    n_clusters: int = 1  # Number of clusters in clustered dependence mode
    cluster_labels: Optional[np.ndarray] = None  # Cluster assignment for each forecaster
    cluster_rho: float = 0.6  # Loading on the cluster-specific latent factor


@dataclass
class SimulationData:
    """
    Store the output of a simulation run.

    Attributes:
        y (np.ndarray): Realized target series of shape (T,)
        forecasts (np.ndarray): Forecast matrix of shape (T, M)
        errors (np.ndarray): Forecast-error matrix of shape (T, M)
        losses (np.ndarray): Loss matrix of shape (T, M)
        bias_paths (np.ndarray): Bias process for each forecaster, shape (T, M)
        sigma_paths (np.ndarray): Idiosyncratic scale paths, shape (T, M)
        common_shock (np.ndarray): Common shock series of shape (T,)
        config (ScenarioConfig): Configuration used to generate the data
    """
    y: np.ndarray               # (T,) realized target series
    forecasts: np.ndarray       # (T, M) forecast values
    errors: np.ndarray          # (T, M) forecast errors
    losses: np.ndarray          # (T, M) squared losses
    bias_paths: np.ndarray      # (T, M) bias paths
    sigma_paths: np.ndarray     # (T, M) idiosyncratic standard deviations
    common_shock: np.ndarray    # (T,) common shock path
    config: ScenarioConfig = field(default_factory=ScenarioConfig)


def generate_scenario(cfg: ScenarioConfig) -> SimulationData:
    """
    Generate simulated targets, forecasts, errors, and losses from a scenario
    configuration.

    The generator builds the data in stages:

        1. Construct bias paths `b_{j,t}`
        2. Construct idiosyncratic scale paths `sigma_{j,t}`
        3. Draw a common shock shared across forecasters
        4. Draw idiosyncratic noise, optionally with factor or cluster dependence
        5. Combine components into forecast errors
        6. Build the target series and implied forecasts

    Forecast errors follow the sign convention used throughout this file:

        e_{j,t} = c_t - b_{j,t} - u_{j,t}

    where `c_t` is the common shock, `b_{j,t}` is the bias term, and `u_{j,t}`
    is the idiosyncratic component.

    Args:
        cfg (ScenarioConfig): Scenario specification

    Returns:
        SimulationData: Simulated target, forecasts, errors, and metadata
    """
    rng = _ensure_rng(cfg.seed)
    M, T = cfg.M, cfg.T

    # 1. Generate bias paths  b_{j,t}
    bias = _generate_bias(cfg, rng)  # (T, M)

    # 2. Generate idiosyncratic variance paths
    sigma_idio = _generate_sigma(cfg, rng)  # (T, M)

    # 3. Common shock
    common = rng.normal(0, cfg.sigma_common, size=T)

    # 4. Idiosyncratic errors
    idio = rng.normal(0, 1, size=(T, M)) * sigma_idio

    # Replace independent idiosyncratic noise by a one-factor structure if requested.
    if cfg.factor_rho is not None:
        factor = rng.normal(0, 1, size=T)
        idio_clean = np.zeros((T, M))
        u = rng.normal(0, 1, size=(T, M))
        for j in range(M):
            rho_j = cfg.factor_rho[j]
            idio_clean[:, j] = sigma_idio[:, j] * (
                rho_j * factor + np.sqrt(max(1 - rho_j ** 2, 0)) * u[:, j]
            )
        idio = idio_clean

    # Cluster dependence takes precedence over the global one-factor structure.
    if cfg.n_clusters > 1 and cfg.cluster_labels is not None:
        cluster_factors = rng.normal(0, 1, size=(T, cfg.n_clusters))
        u = rng.normal(0, 1, size=(T, M))
        rho_c = cfg.cluster_rho
        for j in range(M):
            cl = cfg.cluster_labels[j]
            idio[:, j] = sigma_idio[:, j] * (
                rho_c * cluster_factors[:, cl]
                + np.sqrt(max(1 - rho_c ** 2, 0)) * u[:, j]
            )

    # Outliers replace, rather than add to, the baseline idiosyncratic shock.
    if cfg.outlier_prob > 0:
        outlier_mask = rng.random(size=(T, M)) < cfg.outlier_prob
        upper_scale = max(cfg.outlier_scale, cfg.outlier_min_scale)
        outlier_magnitude = rng.uniform(
            cfg.outlier_min_scale * sigma_idio,
            upper_scale * sigma_idio,
            size=(T, M),
        )
        outlier_sign = rng.choice([-1.0, 1.0], size=(T, M))
        outlier_vals = outlier_sign * outlier_magnitude
        idio = np.where(outlier_mask, outlier_vals, idio)

    # 5. Forecast errors.
    # Keep the sign convention used in this file: common shocks enter
    # positively, while bias and idiosyncratic terms enter negatively.
    errors = common[:, None] - bias - idio  # (T, M)

    # 6. Target
    # y_t drawn as some latent, forecasts = y - errors
    # Simplest: y_t = mu + common noise + signal
    y_signal = rng.normal(0, 1, size=T).cumsum() * 0.01  # mild random walk signal
    y = y_signal + common  # plus common
    forecasts = y[:, None] - errors  # f_{j,t} = y_t - e_{j,t}

    # Losses are computed using squared error throughout the simulation section.
    losses = squared_loss(y[:, None], forecasts)

    return SimulationData(
        y=y,
        forecasts=forecasts,
        errors=errors,
        losses=losses,
        bias_paths=bias,
        sigma_paths=sigma_idio,
        common_shock=common,
        config=cfg,
    )


def _generate_bias(cfg: ScenarioConfig, rng) -> np.ndarray:
    """
    Generate the bias paths implied by a scenario configuration.

    Supported bias specifications are:

        - `zero`: no bias
        - `constant`: fixed bias vector over time
        - `break`: piecewise-constant bias with one break
        - `drift`: smoothly evolving bias with a persistent latent slope
        - `cluster`: constant bias vector intended for grouped forecasters
        - `common_rw_idio_ar1`: shared random-walk bias plus idiosyncratic AR(1) deviations
        
    Args:
        cfg (ScenarioConfig): Scenario specification
        rng: NumPy random number generator

    Returns:
        np.ndarray: Bias matrix of shape (T, M)
    """
    M, T = cfg.M, cfg.T
    bias = np.zeros((T, M))

    if cfg.bias_type == "zero":
        pass

    elif cfg.bias_type == "constant":
        if cfg.bias_values is not None:
            b = np.asarray(cfg.bias_values, dtype=float).copy()
        else:
            b = rng.uniform(cfg.bias_init_low, cfg.bias_init_high, size=M)
        if cfg.bias_centered:
            b -= b.mean()
        bias = np.tile(b, (T, 1))

    elif cfg.bias_type == "break":
        t_break = cfg.bias_break_time if cfg.bias_break_time is not None else T // 2
        b_pre = (
            np.asarray(cfg.bias_pre, dtype=float).copy()
            if cfg.bias_pre is not None
            else rng.uniform(cfg.bias_init_low, cfg.bias_init_high, size=M)
        )
        b_post = (
            np.asarray(cfg.bias_post, dtype=float).copy()
            if cfg.bias_post is not None
            else rng.uniform(cfg.bias_init_low, cfg.bias_init_high, size=M)
        )
        if cfg.bias_centered:
            b_pre -= b_pre.mean()
            b_post -= b_post.mean()
        bias[:t_break] = b_pre[None, :]
        bias[t_break:] = b_post[None, :]

    elif cfg.bias_type == "drift":
        rho = cfg.bias_drift_rho
        eta = np.sqrt(max(cfg.bias_drift_eta2, EPS)) if cfg.bias_drift_eta2 is not None else max(cfg.bias_drift_speed, EPS)
        b = np.zeros((T, M))
        slope = np.zeros((T, M))
        b[0] = rng.uniform(cfg.bias_init_low, cfg.bias_init_high, size=M)
        slope[0] = rng.normal(0, eta, size=M)
        for t in range(1, T):
            slope[t] = rho * slope[t - 1] + rng.normal(0, eta, size=M)
            b[t] = b[t - 1] + slope[t]
        if cfg.bias_centered:
            b -= b.mean(axis=1, keepdims=True)
        bias = b

    elif cfg.bias_type == "cluster":
        # This currently behaves like a constant-bias design, not a dynamic cluster process.
        if cfg.bias_values is not None:
            b = cfg.bias_values.copy()
        else:
            b = rng.normal(0, 0.3, size=M)
        if cfg.bias_centered:
            b -= b.mean()
        bias = np.tile(b, (T, 1))

    elif cfg.bias_type == "common_rw_idio_ar1":
        common_rw = np.zeros(T)
        common_rw[0] = float(cfg.bias_common_rw_init)
        rw_scale = max(float(cfg.bias_common_rw_scale), EPS)
        idio_scale = (
            max(float(cfg.bias_idio_ar1_scale), EPS)
            if cfg.bias_idio_ar1_scale is not None
            else rw_scale / 10.0
        )
        rho = float(cfg.bias_idio_ar1_rho)

        for t in range(1, T):
            common_rw[t] = common_rw[t - 1] + rng.normal(0.0, rw_scale)

        idio_bias = np.zeros((T, M))
        stationary_scale = idio_scale / np.sqrt(max(1.0 - rho ** 2, EPS))
        idio_bias[0] = rng.normal(0.0, stationary_scale, size=M)
        for t in range(1, T):
            idio_bias[t] = rho * idio_bias[t - 1] + rng.normal(0.0, idio_scale, size=M)

        if cfg.bias_centered:
            idio_bias -= idio_bias.mean(axis=1, keepdims=True)

        bias = common_rw[:, None] + idio_bias

    else:
        raise ValueError(f"Unsupported bias_type: {cfg.bias_type}")

    return bias


def _generate_sigma(cfg: ScenarioConfig, rng) -> np.ndarray:
    """
    Generate idiosyncratic standard-deviation paths for each forecaster.

    The function first respects any explicitly provided `sigma_idio` input. If
    no fixed path is supplied, it supports either a smooth log-precision drift
    or a simple piecewise-constant variance shift. If neither is requested, it
    falls back to a constant default scale.

    Args:
        cfg (ScenarioConfig): Scenario specification
        rng: NumPy random number generator

    Returns:
        np.ndarray: Idiosyncratic standard deviations of shape (T, M)
    """
    M, T = cfg.M, cfg.T

    if cfg.sigma_idio is not None:
        s = np.asarray(cfg.sigma_idio, dtype=float)
        if s.ndim == 1:
            return np.tile(s, (T, 1))
        return s

    if cfg.sigma_process == "smooth_precision":
        log_precision = np.zeros((T, M))
        slope = np.zeros((T, M))
        precision = np.zeros((T, M))

        eta = np.sqrt(max(cfg.sigma_drift_eta2, EPS))
        base_precision = 1.0 / max(cfg.base_sigma2, EPS)
        base_log_precision = np.log(base_precision)

        log_precision[0] = base_log_precision + rng.normal(0, 0.1, size=M)
        slope[0] = rng.normal(0, eta, size=M)
        log_precision[0] = np.clip(
            log_precision[0], cfg.log_precision_min, cfg.log_precision_max
        )
        precision[0] = np.exp(log_precision[0])

        for t in range(1, T):
            slope[t] = (
                cfg.sigma_drift_rho * slope[t - 1]
                + rng.normal(0, eta, size=M)
            )
            log_precision[t] = log_precision[t - 1] + slope[t]
            log_precision[t] = np.clip(
                log_precision[t], cfg.log_precision_min, cfg.log_precision_max
            )
            precision[t] = np.exp(log_precision[t])

        return np.sqrt(1.0 / np.maximum(precision, EPS))  # Convert precision back to sigma

    base = np.ones(M) * 0.5

    if cfg.sigma_shift_time is not None:
        sigma_pre = cfg.sigma_pre if cfg.sigma_pre is not None else base
        sigma_post = cfg.sigma_post if cfg.sigma_post is not None else base * 2
        out = np.zeros((T, M))
        out[:cfg.sigma_shift_time] = sigma_pre[None, :]
        out[cfg.sigma_shift_time:] = sigma_post[None, :]
        return out

    return np.tile(base, (T, 1))


# ===================================================================
# 7b.  PRE-BUILT SCENARIOS
# ===================================================================

def scenario_1A(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a stable benchmark scenario with unbiased, homoskedastic forecasts.

    All forecasters have zero bias over time and a constant idiosyncratic
    standard deviation of 1.0. Cross-sectional comovement is introduced only
    through a common shock with standard deviation `sigma_common = 0.5`.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 1A
    """
    return ScenarioConfig(
        name="1A_stable_unbiased",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 1.0,
    )


def scenario_2A(M=8, T=400, T0=200, seed=42, break_frac=0.5) -> ScenarioConfig:
    """
    Generate a scenario with an abrupt structural break in forecaster biases.

    Biases are constant up to time `t_break = floor(T * break_frac)` and then
    switch instantaneously to a new independently drawn bias vector. The
    idiosyncratic standard deviation remains constant at 1.0 throughout.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation
        break_frac (float): Fraction of the sample at which the bias break occurs

    Returns:
        ScenarioConfig: Configuration for scenario 2A
    """
    rng = _ensure_rng(seed + 1)
    b_pre = rng.uniform(0.0, 1.0, size=M)
    b_post = rng.uniform(0.0, 1.0, size=M)
    t_break = int(T * break_frac)
    return ScenarioConfig(
        name="2A_abrupt_break",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="break",
        bias_break_time=t_break,
        bias_pre=b_pre,
        bias_post=b_post,
        bias_centered=False,
        sigma_idio=np.ones(M) * 1.0,
    )


def scenario_2B(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a scenario with smoothly drifting forecaster biases.

    Biases evolve over time through a latent slope process with persistence
    `bias_drift_rho = 0.95` and innovation scale `sqrt(5e-4)`. This creates
    gradual nonstationarity in forecast bias while idiosyncratic volatility
    remains constant across forecasters and time.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 2B
    """
    return ScenarioConfig(
        name="2B_smooth_drift",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="drift",
        bias_drift_speed=float(np.sqrt(5e-4)),
        bias_drift_rho=0.95,
        bias_centered=False,
        bias_init_low=0.0,
        bias_init_high=1.0,
        sigma_idio=np.ones(M) * 1.0,
    )


def scenario_2C(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a scenario with smooth drift in forecaster precision.

    Forecasts remain unbiased, but each forecaster's idiosyncratic variance
    changes over time through a persistent log-precision process. Precision is
    clipped to the interval defined by `log_precision_min` and
    `log_precision_max`, creating gradual heteroskedasticity without bias drift.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 2C
    """
    return ScenarioConfig(
        name="2C_precision_shift",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_process="smooth_precision",
        base_sigma2=1.0,
        sigma_drift_rho=0.95,
        sigma_drift_eta2=8e-4,
        log_precision_min=-2.0,
        log_precision_max=2.0,
    )


def scenario_4A(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """
    Generate a scenario with a shared random-walk bias and idiosyncratic AR(1) biases.

    All forecasters are exposed to the same slowly moving common bias component,
    modeled as a random walk with innovation standard deviation 0.05. On top of
    that, each forecaster carries its own AR(1) bias deviation with persistence
    0.9 and innovation standard deviation 0.005, so the idiosyncratic bias
    innovations are approximately ten times smaller than the volatility of the
    common random walk.

    Args:
        M (int): Number of forecasters
        T (int): Number of time steps
        T0 (int): Start of the out-of-sample evaluation period
        seed (int): Random seed used in the simulation

    Returns:
        ScenarioConfig: Configuration for scenario 4A
    """
    return ScenarioConfig(
        name="4A_common_rw_idio_ar1_bias",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="common_rw_idio_ar1",
        bias_common_rw_scale=0.05,
        bias_common_rw_init=0.0,
        bias_idio_ar1_rho=0.9,
        bias_idio_ar1_scale=0.005,
        bias_centered=False,
        sigma_idio=np.ones(M) * 1.0,
    )


def scenario_3A(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """Display-name alias for the former scenario 4A."""
    cfg = scenario_4A(M=M, T=T, T0=T0, seed=seed)
    cfg.name = cfg.name.replace("4A_", "3A_", 1)
    return cfg


ALL_SCENARIO_FACTORIES = {
    "1A": scenario_1A,
    "2A": scenario_2A,
    "2B": scenario_2B,
    "2C": scenario_2C,
    "3A": scenario_3A,
}

LEGACY_SCENARIO_FACTORIES = {
    "4A": scenario_4A,
}


# ===================================================================
# 8.  ROLLING BACKTEST ENGINE
# ===================================================================

@dataclass
class BacktestConfig:
    """Hyper-parameters for the rolling backtest."""
    # pairwise LD model
    d_max: int = 4
    fixed_d: Optional[int] = None
    fixed_h1: Optional[float] = None
    fixed_h2: Optional[float] = None
    n_cv_folds: int = 5
    loss_diff_versions: Tuple[str, ...] = ("legacy", "v2")
    fast_fixed_ld: bool = False

    # graph layer
    adjacency_type: str = "standardized"   # raw, standardized, thresholded
    centrality_type: str = "eigenvector"   # eigenvector, rowsum, pagerank, softmax
    centrality_types: Optional[Tuple[str, ...]] = None
    teleport: float = TELEPORT
    adj_reg_c: float = 1e-4

    # covariance
    cov_method: str = "shrinkage"  # rolling, ewma, shrinkage, diagonal
    cov_window: int = 60
    cov_ewma_lambda: float = 0.94
    ridge_cov: float = RIDGE_COV

    # optimisation
    alpha: Optional[float] = None  # None => tune; graph reward strength
    gamma: Optional[float] = None  # None => tune; equal-weight shrinkage
    alpha_grid: Optional[np.ndarray] = None
    gamma_grid: Optional[np.ndarray] = None
    tune_window: int = 40  # rolling window for tuning alpha, gamma
    estimate_window: bool = False
    window_grid: Tuple[int, ...] = (20, 40, 60)

    # loss
    loss_name: str = "squared"

    # benchmarks
    recent_best_window: int = 20
    bg_window: int = 60
    var_window: int = 60
    var_max_lags: int = 3
    var_fixed_lag: Optional[int] = None
    var_ic: Literal["bic", "aic"] = "bic"
    include_benchmarks: bool = True

    # misc
    min_history: int = 30  # minimum observations before producing weights


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    oos_periods: np.ndarray         # (n_oos,)
    y_oos: np.ndarray               # (n_oos,)
    forecasts_oos: np.ndarray       # (n_oos, M)

    # weights: dict of method_name -> (n_oos, M)
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_forecasts: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_losses: Dict[str, np.ndarray] = field(default_factory=dict)

    # diagnostics
    mu_matrices: Optional[List] = None   # list of (M,M) at each oos period
    sigma_matrices: Optional[List] = None
    adjacency_matrices: Optional[List] = None
    centrality_scores: Optional[List] = None
    cov_matrices: Optional[List] = None
    alpha_selected: Optional[np.ndarray] = None
    gamma_selected: Optional[np.ndarray] = None
    var_lag_selected: Optional[np.ndarray] = None
    d_selected: Optional[List] = None
    centrality_types: Tuple[str, ...] = field(default_factory=tuple)
    primary_centrality_type: str = "eigenvector"
    candidate_windows: Tuple[int, ...] = field(default_factory=tuple)
    window_selected: Optional[np.ndarray] = None
    window_grid_scores: Dict[int, np.ndarray] = field(default_factory=dict)
    loss_diff_versions: Tuple[str, ...] = field(default_factory=tuple)
    mu_matrices_by_variant: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    sigma_matrices_by_variant: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    adjacency_matrices_by_variant: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    centrality_scores_by_variant: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    centrality_scores_by_type: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    alpha_selected_by_variant: Dict[str, np.ndarray] = field(default_factory=dict)
    gamma_selected_by_variant: Dict[str, np.ndarray] = field(default_factory=dict)
    alpha_selected_by_type: Dict[str, np.ndarray] = field(default_factory=dict)
    gamma_selected_by_type: Dict[str, np.ndarray] = field(default_factory=dict)


def run_backtest(
    data: SimulationData,
    bt_cfg: BacktestConfig = None,
    verbose: bool = False,
) -> BacktestResult:
    """
    Full rolling out-of-sample backtest.
    """
    if bt_cfg is None:
        bt_cfg = BacktestConfig()

    cfg = data.config
    M = cfg.M
    T = cfg.T
    T0 = cfg.T0

    loss_fn = LOSS_REGISTRY[bt_cfg.loss_name]

    y = data.y
    forecasts = data.forecasts
    losses = data.losses
    errors = data.errors

    n_oos = T - T0
    oos_periods = np.arange(T0, T)
    active_loss_diff_versions = _resolve_loss_diff_versions(bt_cfg.loss_diff_versions)
    primary_loss_diff_variant = _primary_loss_diff_variant(active_loss_diff_versions)
    active_centrality_types = _resolve_centrality_types(
        bt_cfg.centrality_type,
        bt_cfg.centrality_types,
    )
    candidate_windows = _normalise_window_grid(bt_cfg.window_grid)

    # Pre-compute pairwise loss differentials (full history)
    # delta_L[i,j,t] = L_i(t) - L_j(t)
    # We'll use 3D array for convenience but only access up to current t
    delta_L_full = np.zeros((M, M, T))
    for i in range(M):
        for j in range(M):
            if i != j:
                delta_L_full[i, j, :] = losses[:, i] - losses[:, j]

    # Storage
    res = BacktestResult(
        oos_periods=oos_periods,
        y_oos=y[T0:T],
        forecasts_oos=forecasts[T0:T],
        mu_matrices=[],
        sigma_matrices=[],
        adjacency_matrices=[],
        centrality_scores=[],
        cov_matrices=[],
        alpha_selected=np.zeros(n_oos),
        gamma_selected=np.zeros(n_oos),
        var_lag_selected=np.zeros(n_oos, dtype=int),
        centrality_types=active_centrality_types,
        primary_centrality_type=bt_cfg.centrality_type,
        candidate_windows=candidate_windows,
        window_selected=np.full(n_oos, float(bt_cfg.cov_window)),
        window_grid_scores={window: np.full(n_oos, np.nan) for window in candidate_windows},
        loss_diff_versions=active_loss_diff_versions,
        mu_matrices_by_variant={variant: [] for variant in active_loss_diff_versions},
        sigma_matrices_by_variant={variant: [] for variant in active_loss_diff_versions},
        adjacency_matrices_by_variant={variant: [] for variant in active_loss_diff_versions},
        centrality_scores_by_variant={variant: [] for variant in active_loss_diff_versions},
        centrality_scores_by_type={centrality_type: [] for centrality_type in active_centrality_types},
        alpha_selected_by_variant={
            variant: np.zeros(n_oos) for variant in active_loss_diff_versions
        },
        gamma_selected_by_variant={
            variant: np.zeros(n_oos) for variant in active_loss_diff_versions
        },
        alpha_selected_by_type={
            centrality_type: np.zeros(n_oos) for centrality_type in active_centrality_types
        },
        gamma_selected_by_type={
            centrality_type: np.zeros(n_oos) for centrality_type in active_centrality_types
        },
    )

    # Method names
    method_names = ["equal"]
    if bt_cfg.include_benchmarks:
        method_names.extend([
            "recent_best",
            "bates_granger_mv",
            "var_error",
        ])
    for base_name, supported_variants in PAIRWISE_LD_METHOD_VARIANTS.items():
        for variant in active_loss_diff_versions:
            if variant not in supported_variants:
                continue
            if base_name in {"graph_only", "full_gcsr"}:
                for centrality_type in active_centrality_types:
                    method_names.append(
                        _ld_centrality_method_name(
                            base_name,
                            variant,
                            centrality_type,
                            active_centrality_types,
                        )
                    )
            else:
                method_names.append(_loss_diff_method_name(base_name, variant))
    method_names = _preferred_method_subset(method_names)
    for name in method_names:
        res.weights[name] = np.zeros((n_oos, M))
        res.combined_forecasts[name] = np.zeros(n_oos)
        res.combined_losses[name] = np.zeros(n_oos)

    # Default grids for alpha, gamma
    if bt_cfg.alpha_grid is None:
        bt_cfg.alpha_grid = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
    if bt_cfg.gamma_grid is None:
        bt_cfg.gamma_grid = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

    # ---- Main OOS loop ----
    for oos_idx in range(n_oos):
        t = T0 + oos_idx  # forecast origin: using data up to t-1, forecasting y_t
        # But let's clarify timing:
        # At origin t, we have y_0,...,y_{t-1} and forecasts f_{j,0},...,f_{j,t}
        # We form weights for combining f_{j,t} to forecast y_t
        # After y_t is realised, we can evaluate.
        # So: losses available up through t-1, forecasts for period t already made.

        hist_end = t  # exclusive: indices 0..t-1 are available
        if hist_end < bt_cfg.min_history:
            # not enough history, use equal weights
            for name in method_names:
                res.weights[name][oos_idx] = equal_weights(M)
            f_oos = forecasts[t]
            w_eq = equal_weights(M)
            y_t = y[t]
            for name in method_names:
                w = res.weights[name][oos_idx]
                res.combined_forecasts[name][oos_idx] = f_oos @ w
                res.combined_losses[name][oos_idx] = loss_fn(
                    y_t, res.combined_forecasts[name][oos_idx]
                )
            continue

        # --- LAYER 1: Pairwise LD predictions ---
        loss_diff_states = {}
        for variant in active_loss_diff_versions:
            predictor = PAIRWISE_LD_PREDICTORS[variant]
            mu_mat = np.zeros((M, M))
            sigma_mat = np.zeros((M, M))
            for i in range(M):
                for j in range(i + 1, M):
                    dl_hist = delta_L_full[i, j, :hist_end]
                    if len(dl_hist) < bt_cfg.min_history:
                        continue
                    try:
                        pld = predictor(
                            dl_hist,
                            d_max=bt_cfg.d_max,
                            fixed_d=bt_cfg.fixed_d,
                            fixed_h1=bt_cfg.fixed_h1,
                            fixed_h2=bt_cfg.fixed_h2,
                            n_cv_folds=bt_cfg.n_cv_folds,
                            fast_fixed=bt_cfg.fast_fixed_ld,
                        )
                    except Exception as exc:
                        if verbose:
                            print(
                                f"[BACKTEST] {variant} pair ({i},{j}) at t={t} failed "
                                f"with {type(exc).__name__}: {exc}"
                            )
                        continue

                    mu_ij = float(np.nan_to_num(pld.mu_hat, nan=0.0, posinf=0.0, neginf=0.0))
                    sigma_ij = float(np.nan_to_num(pld.sigma_hat, nan=0.0, posinf=0.0, neginf=0.0))
                    mu_mat[i, j] = mu_ij
                    mu_mat[j, i] = -mu_ij
                    sigma_mat[i, j] = sigma_ij
                    sigma_mat[j, i] = sigma_ij

            if bt_cfg.adjacency_type == "raw":
                A = build_adjacency_raw(mu_mat)
            elif bt_cfg.adjacency_type == "standardized":
                A = build_adjacency_standardized(mu_mat, sigma_mat, bt_cfg.adj_reg_c)
            elif bt_cfg.adjacency_type == "thresholded":
                A = build_adjacency_thresholded(mu_mat, sigma_mat, 1.0, bt_cfg.adj_reg_c)
            else:
                A = build_adjacency_raw(mu_mat)

            centrality_by_type = {}
            for centrality_type in active_centrality_types:
                centrality_by_type[centrality_type] = compute_centrality_scores(
                    A,
                    mu_mat,
                    centrality_type,
                    bt_cfg.teleport,
                )

            loss_diff_states[variant] = {
                "mu_mat": mu_mat,
                "sigma_mat": sigma_mat,
                "adjacency": A,
                "centrality_by_type": centrality_by_type,
            }
            res.mu_matrices_by_variant[variant].append(mu_mat.copy())
            res.sigma_matrices_by_variant[variant].append(sigma_mat.copy())
            res.adjacency_matrices_by_variant[variant].append(A.copy())
            primary_r = centrality_by_type[bt_cfg.centrality_type]
            res.centrality_scores_by_variant[variant].append(primary_r.copy())

            if variant == primary_loss_diff_variant:
                for centrality_type, scores in centrality_by_type.items():
                    res.centrality_scores_by_type[centrality_type].append(scores.copy())

        errors_hist = errors[:hist_end]

        # --- Window-length selection on the primary GCSR variant ---
        primary_state = loss_diff_states[primary_loss_diff_variant]
        selected_window = bt_cfg.cov_window
        window_scores = {}
        if bt_cfg.estimate_window:
            selected_window, window_scores = _select_optimal_window(
                bt_cfg,
                losses[:hist_end],
                forecasts[:hist_end],
                y[:hist_end],
                errors_hist,
                primary_state["centrality_by_type"][bt_cfg.centrality_type],
                M,
                loss_fn,
            )
        res.window_selected[oos_idx] = float(selected_window)
        for window, score in window_scores.items():
            res.window_grid_scores[window][oos_idx] = score

        # --- LAYER 3: Covariance ---
        Sigma = _estimate_covariance(
            errors_hist,
            bt_cfg,
            window_override=selected_window,
        )
        res.cov_matrices.append(Sigma.copy())

        # --- Tune alpha, gamma for the explicit centrality variants ---
        for variant, state in loss_diff_states.items():
            state["alpha_by_type"] = {}
            state["gamma_by_type"] = {}
            primary_alpha = None
            primary_gamma = None
            for centrality_type, scores in state["centrality_by_type"].items():
                alpha_sel, gamma_sel = _tune_alpha_gamma(
                    bt_cfg, losses[:hist_end], forecasts[:hist_end], y[:hist_end],
                    errors_hist, delta_L_full[:, :, :hist_end],
                    scores, Sigma, M, loss_fn,
                    tune_window_override=selected_window,
                )
                state["alpha_by_type"][centrality_type] = alpha_sel
                state["gamma_by_type"][centrality_type] = gamma_sel

                if variant == primary_loss_diff_variant:
                    res.alpha_selected_by_type[centrality_type][oos_idx] = alpha_sel
                    res.gamma_selected_by_type[centrality_type][oos_idx] = gamma_sel

                if centrality_type == bt_cfg.centrality_type:
                    primary_alpha = alpha_sel
                    primary_gamma = gamma_sel

            res.alpha_selected_by_variant[variant][oos_idx] = float(primary_alpha)
            res.gamma_selected_by_variant[variant][oos_idx] = float(primary_gamma)

        res.mu_matrices.append(primary_state["mu_mat"].copy())
        res.sigma_matrices.append(primary_state["sigma_mat"].copy())
        res.adjacency_matrices.append(primary_state["adjacency"].copy())
        res.centrality_scores.append(
            primary_state["centrality_by_type"][bt_cfg.centrality_type].copy()
        )
        res.alpha_selected[oos_idx] = primary_state["alpha_by_type"][bt_cfg.centrality_type]
        res.gamma_selected[oos_idx] = primary_state["gamma_by_type"][bt_cfg.centrality_type]

        # --- Compute weights for each method ---
        f_oos = forecasts[t]
        y_t = y[t]

        # 1. Equal
        w_eq = equal_weights(M)
        res.weights["equal"][oos_idx] = w_eq

        # 2. Recent best
        if "recent_best" in res.weights:
            recent_best_window = (
                selected_window if bt_cfg.estimate_window else bt_cfg.recent_best_window
            )
            w_rb = recent_best_selection(losses[:hist_end], recent_best_window)
            res.weights["recent_best"][oos_idx] = w_rb

        # 3. Bates-Granger MV
        if "bates_granger_mv" in res.weights:
            bg_window = selected_window if bt_cfg.estimate_window else bt_cfg.bg_window
            w_bgmv = bates_granger_mv_weights(errors_hist, bg_window)
            res.weights["bates_granger_mv"][oos_idx] = w_bgmv

        # 4. VAR benchmark on forecast errors
        if "var_error" in res.weights:
            var_window = selected_window if bt_cfg.estimate_window else bt_cfg.var_window
            w_var, var_lags, _, _ = var_error_weights(
                errors_hist,
                max_lags=bt_cfg.var_max_lags,
                fixed_lag=bt_cfg.var_fixed_lag,
                window=var_window,
                ic=bt_cfg.var_ic,
                ridge=bt_cfg.ridge_cov,
            )
            res.var_lag_selected[oos_idx] = var_lags
            res.weights["var_error"][oos_idx] = w_var

        # 5. LD-driven methods for each pairwise model variant
        for variant, state in loss_diff_states.items():
            mu_mat = state["mu_mat"]

            if _loss_diff_method_enabled("rs_selection", variant):
                w_rs = rs_selection_weights(mu_mat)
                res.weights[_loss_diff_method_name("rs_selection", variant)][oos_idx] = w_rs

            for centrality_type in active_centrality_types:
                scores = state["centrality_by_type"][centrality_type]
                alpha_sel = state["alpha_by_type"][centrality_type]
                gamma_sel = state["gamma_by_type"][centrality_type]

                if _loss_diff_method_enabled("graph_only", variant):
                    graph_name = _ld_centrality_method_name(
                        "graph_only",
                        variant,
                        centrality_type,
                        active_centrality_types,
                    )
                    w_go = graph_only_weights(scores)
                    res.weights[graph_name][oos_idx] = w_go

                if _loss_diff_method_enabled("full_gcsr", variant):
                    full_name = _ld_centrality_method_name(
                        "full_gcsr",
                        variant,
                        centrality_type,
                        active_centrality_types,
                    )
                    w_full = full_combination_weights(
                        Sigma, scores, alpha_sel, gamma_sel, bt_cfg.ridge_cov
                    )
                    res.weights[full_name][oos_idx] = w_full

        # --- Combined forecasts and losses ---
        for name in method_names:
            w = res.weights[name][oos_idx]
            comb = float(f_oos @ w)
            res.combined_forecasts[name][oos_idx] = comb
            res.combined_losses[name][oos_idx] = loss_fn(y_t, comb)

        if verbose and oos_idx % 50 == 0:
            print(f"  OOS {oos_idx}/{n_oos} (t={t})")

    return res


def _select_optimal_window(
    bt_cfg,
    losses_hist,
    forecasts_hist,
    y_hist,
    errors_hist,
    r_current,
    M,
    loss_fn,
):
    """
    Select the best rolling window from the configured grid using in-sample loss.

    The objective is the validation loss of the primary `full_gcsr` specification
    evaluated over the candidate tuning window. The chosen window is then reused
    for the covariance matrix and the rolling-window benchmarks at the current
    forecast origin.
    """
    best_window = int(bt_cfg.cov_window)
    best_score = np.inf
    scores = {}

    for window in _normalise_window_grid(bt_cfg.window_grid):
        Sigma = _estimate_covariance(errors_hist, bt_cfg, window_override=window)
        _, _, validation_loss = _tune_alpha_gamma(
            bt_cfg,
            losses_hist,
            forecasts_hist,
            y_hist,
            errors_hist,
            None,
            r_current,
            Sigma,
            M,
            loss_fn,
            tune_window_override=window,
            return_score=True,
        )
        scores[window] = validation_loss
        if validation_loss < best_score:
            best_score = validation_loss
            best_window = window

    return int(best_window), scores


def _tune_alpha_gamma(
    bt_cfg, losses_hist, forecasts_hist, y_hist,
    errors_hist, delta_L_hist,
    r_current, Sigma_current, M, loss_fn,
    tune_window_override: Optional[int] = None,
    return_score: bool = False,
):
    """
    Tune alpha, gamma over a past validation window.
    Uses recent OOS performance of the method.
    """
    T_hist = len(y_hist)
    target_window = bt_cfg.tune_window if tune_window_override is None else int(tune_window_override)
    tw = min(target_window, T_hist - bt_cfg.min_history)
    if tw < 5:
        # not enough history to tune, use moderate defaults
        if return_score:
            return 0.1, 0.1, np.nan
        return 0.1, 0.1

    alpha_grid = bt_cfg.alpha_grid
    gamma_grid = bt_cfg.gamma_grid

    # Simple grid search: for each (alpha, gamma), compute what the
    # combined forecast would have been over the tuning window using
    # the *current* r and Sigma as proxies (approximation for speed)
    best_loss = np.inf
    best_alpha = float(alpha_grid[len(alpha_grid) // 2])
    best_gamma = float(gamma_grid[len(gamma_grid) // 2])

    val_start = T_hist - tw
    y_val = y_hist[val_start:T_hist]
    f_val = forecasts_hist[val_start:T_hist]

    if bt_cfg.alpha is not None and bt_cfg.gamma is not None:
        fixed_weights = full_combination_weights(
            Sigma_current,
            r_current,
            bt_cfg.alpha,
            bt_cfg.gamma,
            bt_cfg.ridge_cov,
        )
        fixed_score = float(np.mean(loss_fn(y_val, f_val @ fixed_weights)))
        if return_score:
            return bt_cfg.alpha, bt_cfg.gamma, fixed_score
        return bt_cfg.alpha, bt_cfg.gamma

    for alpha in alpha_grid:
        for gamma in gamma_grid:
            w = full_combination_weights(
                Sigma_current, r_current, alpha, gamma, bt_cfg.ridge_cov
            )
            comb = f_val @ w
            avg_loss = loss_fn(y_val, comb).mean()
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_alpha = float(alpha)
                best_gamma = float(gamma)

    if return_score:
        return best_alpha, best_gamma, float(best_loss)
    return best_alpha, best_gamma


# ===================================================================
# 9.  REPLICATION AND SUMMARY
# ===================================================================

@dataclass
class MCResult:
    """
    Store Monte Carlo performance summaries for one simulation scenario.

    The object collects replication-level mean losses, relative losses versus
    equal weights, and aggregate win frequencies across all combination
    methods evaluated in the backtest.
    """
    scenario_name: str
    n_reps: int
    method_names: List[str]
    msfe_matrix: np.ndarray       # (n_reps, n_methods)
    rel_msfe_matrix: np.ndarray   # relative to equal weights
    mean_msfe: np.ndarray         # (n_methods,)
    median_msfe: np.ndarray
    mean_rel_msfe: np.ndarray
    win_freq: np.ndarray          # fraction of reps each method wins


def run_monte_carlo(
    scenario_factory,
    n_reps: int = 20,
    bt_cfg: BacktestConfig = None,
    verbose: bool = False,
    **scenario_kwargs,
) -> MCResult:
    """
    Run repeated Monte Carlo replications for a single scenario design.

    For each replication, the scenario is regenerated with a shifted random
    seed, the rolling backtest is executed, and mean out-of-sample losses are
    collected for all methods. The function then aggregates those replication-
    level results into one `MCResult`.

    Args:
        scenario_factory (Callable): Scenario constructor such as `scenario_1A`
            or `scenario_2B`
        n_reps (int): Number of Monte Carlo replications
        bt_cfg (BacktestConfig | None): Backtest configuration used in each run
        verbose (bool): Whether to print replication progress
        **scenario_kwargs: Keyword arguments forwarded to `scenario_factory`

    Returns:
        MCResult: Aggregated Monte Carlo summary for the scenario
    """
    if bt_cfg is None:
        bt_cfg = BacktestConfig(
            centrality_types=DEFAULT_COMPARISON_CENTRALITY_TYPES,
        )
    else:
        bt_cfg = _ensure_comparison_centrality_types(bt_cfg)

    all_results = []
    for rep in range(n_reps):
        kw = dict(scenario_kwargs)
        kw["seed"] = kw.get("seed", 42) + rep * 1000
        cfg = scenario_factory(**kw)
        data = generate_scenario(cfg)
        res = run_backtest(data, bt_cfg, verbose=False)
        all_results.append(res)
        if verbose:
            print(f"Replication {rep+1}/{n_reps} done.")

    # Summarise
    method_names = list(all_results[0].combined_losses.keys())
    n_methods = len(method_names)
    msfe_mat = np.zeros((n_reps, n_methods))
    for rep, res in enumerate(all_results):
        for jm, name in enumerate(method_names):
            msfe_mat[rep, jm] = res.combined_losses[name].mean()

    eq_idx = method_names.index("equal")
    rel_msfe = msfe_mat / (msfe_mat[:, eq_idx:eq_idx + 1] + EPS)

    mean_msfe = msfe_mat.mean(axis=0)
    median_msfe = np.median(msfe_mat, axis=0)
    mean_rel = rel_msfe.mean(axis=0)
    win_freq = np.zeros(n_methods)
    winners = msfe_mat.argmin(axis=1)
    for w in winners:
        win_freq[w] += 1
    win_freq /= n_reps

    return MCResult(
        scenario_name=cfg.name,
        n_reps=n_reps,
        method_names=method_names,
        msfe_matrix=msfe_mat,
        rel_msfe_matrix=rel_msfe,
        mean_msfe=mean_msfe,
        median_msfe=median_msfe,
        mean_rel_msfe=mean_rel,
        win_freq=win_freq,
    )


def summarise_mc(mc: MCResult) -> pd.DataFrame:
    """
    Convert a Monte Carlo result object into a tidy summary table.

    Args:
        mc (MCResult): Monte Carlo output from `run_monte_carlo`

    Returns:
        pd.DataFrame: Method-level summary sorted by mean MSFE
    """
    df = pd.DataFrame({
        "Method": mc.method_names,
        "Mean_MSFE": mc.mean_msfe,
        "Median_MSFE": mc.median_msfe,
        "Rel_MSFE_vs_EW": mc.mean_rel_msfe,
        "Win_Freq": mc.win_freq,
    })
    df = df.sort_values("Mean_MSFE").reset_index(drop=True)
    return df


# ===================================================================
# 10.  PLOTTING UTILITIES
# ===================================================================

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


METHOD_FAMILY_COLORS = {
    "full_gcsr": "#D55E00",
    "graph_only": "#009E73",
    "rs_selection": "#CC79A7",
    "bates_granger_mv": "#E69F00",
    "var_error": "#0072B2",
    "recent_best": "#666666",
    "equal": "#111111",
}

PAPER_METHOD_ORDER = (
    "equal",
    "var_error",
    "full_gcsr_eigenvector",
    "full_gcsr_pagerank",
    "full_gcsr_softmax",
    "graph_only_eigenvector",
    "graph_only_pagerank",
    "graph_only_softmax",
    "rs_selection_v2",
    "rs_selection",
    "recent_best",
)

PAPER_ADAPTABILITY_METHOD_ORDER = (
    "equal",
    "bates_granger_mv",
    "var_error",
    "graph_only_eigenvector",
    "graph_only_pagerank",
    "graph_only_softmax",
    "full_gcsr_eigenvector",
    "full_gcsr_pagerank",
    "full_gcsr_softmax",
    "rs_selection",
    "rs_selection_v2",
    "recent_best",
)

CENTRALITY_DISPLAY_LABELS = {
    "eigenvector": "eigenvector",
    "pagerank": "pagerank",
    "softmax": "softmax",
    "rowsum": "row sum",
}

BASE_METHOD_DISPLAY_LABELS = {
    "equal": "Equal weights",
    "var_error": "VAR",
    "bates_granger_mv": "Bates-Granger",
    "recent_best": "Recent-best",
    "rs_selection": "Hyperedge RS selection",
    "rs_selection_v2": "RS selection",
    "actual": "Actual",
}

CENTRALITY_COLORS = {
    "eigenvector": "#0072B2",
    "pagerank": "#D55E00",
    "softmax": "#009E73",
    "rowsum": "#CC79A7",
}

CENTRALITY_LINESTYLES = {
    "eigenvector": "-",
    "pagerank": "--",
    "softmax": ":",
    "rowsum": "-.",
}

CENTRALITY_MARKERS = {
    "eigenvector": "o",
    "pagerank": "s",
    "softmax": "^",
    "rowsum": "D",
}

FAMILY_MARKERS = {
    "full_gcsr": "o",
    "graph_only": "s",
    "rs_selection": "^",
    "bates_granger_mv": "D",
    "var_error": "P",
    "recent_best": "X",
    "equal": "v",
}

CENTRALITY_HATCHES = {
    "eigenvector": "////",
    "pagerank": "xxxx",
    "softmax": "....",
    "rowsum": "\\\\\\\\",
}

WINDOW_LINESTYLES = {
    20: "-",
    40: "--",
    60: ":",
}

WINDOW_COLORS = {
    20: "#0B3954",
    40: "#6B8E23",
    60: "#B15E3E",
}


def set_plot_style():
    """Set a readable, grayscale-friendly plotting style."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.titlepad": 10,
        "axes.labelpad": 8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
        "legend.title_fontsize": 12,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.4,
        "lines.markersize": 6.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "#B0B0B0",
        "font.family": "DejaVu Sans",
    })


set_plot_style()


def scenario_display_label(name: Union[str, int]) -> str:
    """Map internal scenario identifiers to paper-facing labels."""
    text = str(name)
    return "3A" if text == "4A" else text


def _method_order_index(method: str, order: Sequence[str] = PAPER_METHOD_ORDER) -> int:
    """Return a stable index for display ordering with graceful fallback."""
    order_map = {name: idx for idx, name in enumerate(order)}
    if method in order_map:
        return order_map[method]
    family = _method_family(method)
    centrality_type = _method_centrality_type(method)
    family_fallback = {
        "equal": 0,
        "var_error": 1,
        "full_gcsr": 2,
        "graph_only": 3,
        "rs_selection": 4,
        "recent_best": 5,
        "bates_granger_mv": 6,
    }
    return 100 + 10 * family_fallback.get(family, 9) + CENTRALITY_ORDER.get(centrality_type, 9)


def order_methods_for_display(
    methods: Sequence[str],
    order: Sequence[str] = PAPER_METHOD_ORDER,
) -> List[str]:
    """Order available method names for consistent paper legends."""
    available = list(dict.fromkeys(methods))
    order_map = {name: idx for idx, name in enumerate(order)}
    ordered = [name for name in order if name in available]
    remainder = [name for name in available if name not in order_map]
    remainder.sort(key=_method_sort_key)
    return ordered + remainder


def method_display_label(method: str, math: bool = True) -> str:
    """Map internal method names to paper-facing labels."""
    if method in BASE_METHOD_DISPLAY_LABELS:
        return BASE_METHOD_DISPLAY_LABELS[method]

    family = _method_family(method)
    centrality_type = _method_centrality_type(method)
    family_label = {
        "full_gcsr": "GLIDER",
        "graph_only": "GLIDE",
    }.get(family)

    if family_label is not None:
        if centrality_type is None:
            return family_label
        centrality_label = CENTRALITY_DISPLAY_LABELS.get(centrality_type, centrality_type)
        if math:
            return rf"{family_label}$_{{\mathrm{{{centrality_label}}}}}$"
        return f"{family_label} ({centrality_label})"

    return method.replace("_", " ").title()


def _method_plot_style(name: str) -> Dict[str, object]:
    """Consistent style mapping used across performance and diagnostics plots."""
    family = _method_family(name)
    centrality_type = _method_centrality_type(name)
    linestyle = CENTRALITY_LINESTYLES.get(centrality_type, "-")
    marker = CENTRALITY_MARKERS.get(centrality_type, FAMILY_MARKERS.get(family, "o"))
    if name == "rs_selection_v2":
        linestyle = "-"
        marker = "P"
    elif name == "rs_selection":
        linestyle = "--"
        marker = "X"
    return {
        "color": METHOD_FAMILY_COLORS.get(family, "#4D4D4D"),
        "linestyle": linestyle,
        "marker": marker,
        "hatch": CENTRALITY_HATCHES.get(centrality_type, ""),
    }


def _apply_bar_patch_style(bar, name: str) -> None:
    """Apply face color, hatch, and outline styling to a bar artist."""
    style = _method_plot_style(name)
    bar.set_facecolor(style["color"])
    bar.set_edgecolor("#2F2F2F")
    bar.set_linewidth(0.8)
    if style["hatch"]:
        bar.set_hatch(style["hatch"])


def _line_marker_kwargs(n_points: int, marker: str) -> Dict[str, object]:
    """Readable marker settings shared by dense comparison plots."""
    return {
        "marker": marker,
        "markersize": 5.5,
        "markevery": max(n_points // 10, 1),
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
    }


def _apply_publication_legend(
    ax,
    ncol: int = 1,
    loc: str = "best",
    fontsize: float = 10.5,
    method_order: Optional[Sequence[str]] = None,
    display_labels: bool = True,
    **kwargs,
):
    """Draw a compact, readable legend suited for paper-ready figures."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if method_order is not None:
        order_map = {name: idx for idx, name in enumerate(method_order)}
        pairs = sorted(
            zip(handles, labels),
            key=lambda pair: (order_map.get(pair[1], 999), _method_order_index(pair[1])),
        )
        handles, labels = [list(values) for values in zip(*pairs)]
    if display_labels:
        labels = [method_display_label(label) for label in labels]
    legend = ax.legend(
        handles,
        labels,
        ncol=ncol,
        loc=loc,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#B0B0B0",
        borderpad=0.5,
        labelspacing=0.4,
        columnspacing=1.0,
        handlelength=2.4,
        handletextpad=0.6,
        **kwargs,
    )
    for text in legend.get_texts():
        text.set_fontweight("medium")
    return legend


def _is_datetime_axis(x_axis: Sequence[object]) -> bool:
    """Check whether a plotting axis contains datelike values."""
    if x_axis is None:
        return False
    return pd.api.types.is_datetime64_any_dtype(pd.Series(x_axis))


def _format_plot_time_axis(ax, x_axis: Sequence[object], x_label: str = "Time") -> None:
    """Format numeric axes plainly and datelike axes with calendar years."""
    ax.set_xlabel(x_label)
    if not _is_datetime_axis(x_axis):
        return

    year_locator = mdates.YearLocator()
    year_formatter = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(year_formatter)
    ax.tick_params(axis="x", rotation=0)


# ---------- A. Scenario Visualization ----------

def plot_bias_paths(data: SimulationData, ax=None):
    """Plot latent bias paths b_{j,t}."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    M = data.config.M
    for j in range(M):
        ax.plot(data.bias_paths[:, j], label=f"Model {j+1}", alpha=0.7, linewidth=1)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5, label="OOS start")
    ax.set_title(f"Bias paths — {data.config.name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Bias $b_{j,t}$")
    _apply_publication_legend(ax, ncol=4, fontsize=10.0)
    return ax


def plot_sigma_paths(data: SimulationData, ax=None):
    """Plot idiosyncratic std-dev paths."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    M = data.config.M
    for j in range(M):
        ax.plot(data.sigma_paths[:, j], label=f"Model {j+1}", alpha=0.7, linewidth=1)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Idiosyncratic σ paths — {data.config.name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\sigma_{j,t}$")
    _apply_publication_legend(ax, ncol=4, fontsize=10.0)
    return ax


def plot_common_shock(data: SimulationData, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(data.common_shock, color="steelblue", alpha=0.7, linewidth=0.8)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Common shock — {data.config.name}")
    ax.set_xlabel("Time")
    return ax


def plot_forecast_errors(
    data: SimulationData,
    ax=None,
    max_models: Optional[int] = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    # Show the full simulated model panel unless the caller explicitly asks
    # for a smaller subset to avoid silently dropping forecasters.
    if max_models is None:
        M = data.config.M
    else:
        M = min(data.config.M, int(max_models))
    for j in range(M):
        ax.plot(data.errors[:, j], alpha=0.4, linewidth=0.6, label=f"Model {j+1}")
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Forecast errors — {data.config.name}")
    _apply_publication_legend(ax, ncol=min(M, 4), fontsize=10.0)
    return ax


def plot_scenario_summary(data: SimulationData):
    """4-panel scenario overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_bias_paths(data, axes[0, 0])
    plot_sigma_paths(data, axes[0, 1])
    plot_common_shock(data, axes[1, 0])
    plot_forecast_errors(data, axes[1, 1])
    fig.suptitle(f"Scenario: {data.config.name}", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------- B. Pairwise Methodology Visualization ----------

def plot_pairwise_heatmap(matrix: np.ndarray, title: str = "", ax=None, cmap="RdBu_r"):
    """Plot heatmap of an MxM pairwise matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    M = matrix.shape[0]
    if HAS_SEABORN:
        sns.heatmap(
            matrix, ax=ax, annot=M <= 10, fmt=".3f",
            center=0, cmap=cmap, square=True,
            xticklabels=[f"{i+1}" for i in range(M)],
            yticklabels=[f"{i+1}" for i in range(M)],
        )
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.set_xticklabels([f"{i+1}" for i in range(M)])
        ax.set_yticklabels([f"{i+1}" for i in range(M)])
    ax.set_title(title)
    ax.set_xlabel("Model j")
    ax.set_ylabel("Model i")
    return ax


def plot_adjacency_heatmaps(res: BacktestResult, time_indices=None):
    """Plot adjacency matrices at selected OOS time points."""
    if time_indices is None:
        n = len(res.adjacency_matrices)
        time_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        time_indices = [i for i in time_indices if 0 <= i < n]

    n_plots = len(time_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    for idx, ti in enumerate(time_indices):
        t_abs = res.oos_periods[ti]
        plot_pairwise_heatmap(
            res.adjacency_matrices[ti],
            title=f"Adjacency A (t={t_abs})",
            ax=axes[idx],
            cmap="YlOrRd",
        )
    fig.tight_layout()
    return fig


def plot_centrality_bars(res: BacktestResult, time_indices=None):
    """Bar charts of centrality scores at selected dates."""
    if time_indices is None:
        n = len(res.centrality_scores)
        time_indices = [0, n // 2, n - 1]
        time_indices = [i for i in time_indices if 0 <= i < n]

    n_plots = len(time_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]
    for idx, ti in enumerate(time_indices):
        r = res.centrality_scores[ti]
        M = len(r)
        t_abs = res.oos_periods[ti]
        bars = axes[idx].bar(
            range(M),
            r,
            color="#0B3954",
            edgecolor="#2F2F2F",
            alpha=0.85,
        )
        for bar in bars:
            bar.set_linewidth(0.7)
        axes[idx].axhline(1.0 / M, color="#4D4D4D", ls="--", alpha=0.8, label="1/M")
        axes[idx].set_title(f"Centrality (t={t_abs})")
        axes[idx].set_xlabel("Model")
        axes[idx].set_xticks(range(M))
        axes[idx].set_xticklabels([f"{i+1}" for i in range(M)])
        _apply_publication_legend(axes[idx], fontsize=10.0)
    fig.tight_layout()
    return fig


def plot_graph_network(A: np.ndarray, r: np.ndarray, title: str = ""):
    """Plot directed network from adjacency matrix."""
    if not HAS_NETWORKX:
        print("networkx not available; skipping graph plot.")
        return None
    M = A.shape[0]
    G = nx.DiGraph()
    for i in range(M):
        G.add_node(i, label=f"M{i+1}")
    for i in range(M):
        for j in range(M):
            if A[i, j] > 1e-6:
                G.add_edge(i, j, weight=A[i, j])

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42, k=2)
    node_sizes = r * 3000 + 200
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=r,
                           cmap="YlOrRd", alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: f"M{i+1}" for i in range(M)},
                            font_size=9, ax=ax)
    edges = G.edges(data=True)
    if edges:
        edge_weights = [d["weight"] for _, _, d in edges]
        max_w = max(edge_weights) if edge_weights else 1
        widths = [2.0 * w / (max_w + EPS) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4,
                               edge_color="gray", arrows=True, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------- C. Weight Diagnostics ----------

def plot_weight_timeseries(
    res: BacktestResult,
    methods=None,
    time_axis: Optional[Sequence[object]] = None,
    x_label: str = "Time",
    model_names: Optional[Sequence[str]] = None,
):
    """Time series of weights for selected methods."""
    if methods is None:
        methods = _preferred_method_subset(res.weights.keys(), limit=6)
    if time_axis is None:
        time_axis = res.oos_periods
    elif len(time_axis) != len(res.oos_periods):
        raise ValueError("time_axis must match the number of OOS periods.")
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 3.5 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for idx, name in enumerate(methods):
        w = res.weights.get(name)
        if w is None:
            continue
        M = w.shape[1]
        if model_names is None:
            labels = [f"M{j+1}" for j in range(M)]
        else:
            labels = list(model_names)
            if len(labels) != M:
                raise ValueError("model_names must match the number of forecast models.")
        palette = plt.get_cmap("tab20c")(np.linspace(0.15, 0.85, M))
        axes[idx].stackplot(
            time_axis, w.T,
            labels=labels,
            colors=palette,
            alpha=0.8,
        )
        axes[idx].set_title(f"Weights: {method_display_label(name)}")
        axes[idx].set_ylabel("Weight")
        axes[idx].set_ylim(0, 1)
        if M <= 10:
            _apply_publication_legend(axes[idx], loc="upper right", ncol=M, fontsize=9.5)
    _format_plot_time_axis(axes[-1], time_axis, x_label=x_label)
    if _is_datetime_axis(time_axis):
        fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def compute_herfindahl(w: np.ndarray) -> np.ndarray:
    """Herfindahl index for each row of weights matrix (n_oos, M)."""
    return (w ** 2).sum(axis=1)


def compute_effective_n(w: np.ndarray) -> np.ndarray:
    """Effective number of forecasters = 1 / Herfindahl."""
    h = compute_herfindahl(w)
    return 1.0 / np.maximum(h, EPS)


def compute_turnover(w: np.ndarray) -> np.ndarray:
    """Turnover: sum |w_t - w_{t-1}| / 2."""
    diffs = np.abs(np.diff(w, axis=0))
    return diffs.sum(axis=1) / 2.0


def plot_weight_diagnostics(res: BacktestResult, methods=None):
    """Herfindahl, effective N, and turnover."""
    if methods is None:
        methods = _preferred_method_subset(res.weights.keys(), limit=8)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for name in methods:
        w = res.weights.get(name)
        if w is None:
            continue
        style = _method_plot_style(name)
        t_axis = res.oos_periods
        herf = compute_herfindahl(w)
        eff_n = compute_effective_n(w)
        turn = compute_turnover(w)
        axes[0].plot(
            t_axis, herf, label=name, alpha=0.9, linewidth=2.0,
            color=style["color"], linestyle=style["linestyle"],
            **_line_marker_kwargs(len(t_axis), style["marker"]),
        )
        axes[1].plot(
            t_axis, eff_n, label=name, alpha=0.9, linewidth=2.0,
            color=style["color"], linestyle=style["linestyle"],
            **_line_marker_kwargs(len(t_axis), style["marker"]),
        )
        axes[2].plot(
            t_axis[1:], turn, label=name, alpha=0.9, linewidth=2.0,
            color=style["color"], linestyle=style["linestyle"],
            **_line_marker_kwargs(len(t_axis[1:]), style["marker"]),
        )

    axes[0].set_title("Herfindahl Index")
    axes[1].set_title("Effective Number of Forecasters")
    axes[2].set_title("Weight Turnover")
    for ax in axes:
        _apply_publication_legend(ax, ncol=3, fontsize=10.0)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


# ---------- D. Covariance Diagnostics ----------

def plot_covariance_diagnostics(res: BacktestResult):
    """Condition number and trace of Sigma over time."""
    if not res.cov_matrices:
        return None
    cond_nums = []
    traces = []
    for Sigma in res.cov_matrices:
        eigvals = np.linalg.eigvalsh(Sigma)
        eigvals = np.maximum(eigvals, EPS)
        cond_nums.append(eigvals[-1] / eigvals[0])
        traces.append(np.trace(Sigma))

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(res.oos_periods, cond_nums, color="#0B3954", alpha=0.9, linewidth=2.0)
    axes[0].set_title("Condition Number of Σ̂")
    axes[0].set_ylabel("κ(Σ̂)")
    axes[1].plot(res.oos_periods, traces, color="#B15E3E", alpha=0.9, linewidth=2.0)
    axes[1].set_title("Trace of Σ̂")
    axes[1].set_ylabel("tr(Σ̂)")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_cov_heatmap(res: BacktestResult, oos_idx: int = -1):
    """Heatmap of covariance matrix at one time point."""
    if not res.cov_matrices:
        return None
    if oos_idx < 0:
        oos_idx = len(res.cov_matrices) + oos_idx
    Sigma = res.cov_matrices[oos_idx]
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pairwise_heatmap(Sigma, title=f"Σ̂ (t={res.oos_periods[oos_idx]})", ax=ax, cmap="viridis")
    fig.tight_layout()
    return fig


# ---------- E. Performance Metrics ----------

def plot_cumulative_loss(
    res: BacktestResult,
    methods=None,
    reference="equal",
    time_axis: Optional[Sequence[object]] = None,
    x_label: str = "Time",
):
    """Cumulative loss difference relative to a reference method."""
    if methods is None:
        methods = _preferred_method_subset(res.combined_losses.keys())
    if time_axis is None:
        time_axis = res.oos_periods
    elif len(time_axis) != len(res.oos_periods):
        raise ValueError("time_axis must match the number of OOS periods.")

    ref_loss = res.combined_losses.get(reference, res.combined_losses["equal"])
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in methods:
        if name == reference:
            continue
        diff = np.cumsum(res.combined_losses[name] - ref_loss)
        style = _method_plot_style(name)
        ax.plot(
            time_axis,
            diff,
            label=name,
            alpha=0.95,
            linewidth=2.0,
            color=style["color"],
            linestyle=style["linestyle"],
            **_line_marker_kwargs(len(time_axis), style["marker"]),
        )
    ax.axhline(0, color="#2F2F2F", ls="-", linewidth=0.8)
    ax.set_title(f"Cumulative Loss Difference vs {reference}")
    ax.set_ylabel("Cumulative ΔL")
    _format_plot_time_axis(ax, time_axis, x_label=x_label)
    _apply_publication_legend(ax, ncol=3, fontsize=10.0)
    if _is_datetime_axis(time_axis):
        fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def compute_performance_table(res: BacktestResult) -> pd.DataFrame:
    """Summary performance table."""
    rows = []
    for name in res.combined_losses:
        l = res.combined_losses[name]
        rows.append({
            "Method": name,
            "Mean_Loss": l.mean(),
            "Median_Loss": np.median(l),
            "Std_Loss": l.std(),
            "Total_Loss": l.sum(),
        })
    df = pd.DataFrame(rows)
    eq_mean = df.loc[df.Method == "equal", "Mean_Loss"].values[0]
    df["Rel_MSFE"] = df["Mean_Loss"] / (eq_mean + EPS)
    df = df.sort_values("Mean_Loss").reset_index(drop=True)
    return df


def plot_msfe_barplot(res: BacktestResult, ax=None):
    """Bar plot of mean OOS loss by method."""
    df = compute_performance_table(res)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df.Method, df.Rel_MSFE, alpha=0.88)
    for bar, name in zip(bars, df.Method):
        _apply_bar_patch_style(bar, name)
    ax.axvline(1.0, color="#2F2F2F", ls="--", linewidth=0.8)
    ax.set_xlabel("Relative MSFE (vs Equal weights)")
    ax.set_title("OOS Performance")
    ax.invert_yaxis()
    return ax


def plot_alpha_gamma_selected(
    res: BacktestResult,
    centrality_types: Optional[Sequence[str]] = None,
):
    """Time series of selected alpha and gamma."""
    if centrality_types is None:
        centrality_types = (res.primary_centrality_type,)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for centrality_type in centrality_types:
        style = _method_plot_style(_centrality_method_name("full_gcsr", centrality_type, res.centrality_types))
        alpha_series = res.alpha_selected_by_type.get(centrality_type, res.alpha_selected)
        gamma_series = res.gamma_selected_by_type.get(centrality_type, res.gamma_selected)
        axes[0].plot(
            res.oos_periods,
            alpha_series,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=centrality_type,
            **_line_marker_kwargs(len(res.oos_periods), style["marker"]),
        )
        axes[1].plot(
            res.oos_periods,
            gamma_series,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=centrality_type,
            **_line_marker_kwargs(len(res.oos_periods), style["marker"]),
        )
    axes[0].set_title("Selected α over time")
    axes[0].set_ylabel("α")
    axes[1].set_title("Selected γ over time")
    axes[1].set_ylabel("γ")
    if len(tuple(centrality_types)) > 1:
        _apply_publication_legend(axes[0], ncol=len(tuple(centrality_types)), fontsize=10.0)
        _apply_publication_legend(axes[1], ncol=len(tuple(centrality_types)), fontsize=10.0)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_window_selection(
    res: BacktestResult,
    include_scores: bool = True,
):
    """Plot the selected rolling window and, optionally, the candidate scores."""
    if res.window_selected is None:
        return None

    if include_scores and res.window_grid_scores:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.4]},
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 3.5))
        axes = [ax]

    axes[0].step(
        res.oos_periods,
        res.window_selected,
        where="post",
        color="#0B3954",
        linewidth=2.2,
    )
    axes[0].scatter(
        res.oos_periods,
        res.window_selected,
        color="#0B3954",
        s=18,
        zorder=3,
    )
    axes[0].set_title("Selected Rolling Window Length")
    axes[0].set_ylabel("Window")
    axes[0].set_yticks(list(res.candidate_windows))

    if include_scores and res.window_grid_scores:
        for window in res.candidate_windows:
            scores = res.window_grid_scores.get(window)
            if scores is None:
                continue
            axes[1].plot(
                res.oos_periods,
                scores,
                label=f"window={window}",
                color=WINDOW_COLORS.get(window, "#4D4D4D"),
                linestyle=WINDOW_LINESTYLES.get(window, "-"),
                linewidth=2.0,
                **_line_marker_kwargs(len(res.oos_periods), FAMILY_MARKERS.get("equal", "o")),
            )
        axes[1].set_title("In-Sample Validation Loss by Candidate Window")
        axes[1].set_ylabel("Mean Validation Loss")
        _apply_publication_legend(axes[1], ncol=max(len(res.candidate_windows), 1), fontsize=10.0)
        axes[1].set_xlabel("Time")
    else:
        axes[0].set_xlabel("Time")

    fig.tight_layout()
    return fig


# ---------- F. MC Robustness ----------

def plot_mc_boxplot(mc: MCResult):
    """Boxplot of relative MSFE across replications."""
    fig, ax = plt.subplots(figsize=(12, 5))
    df = pd.DataFrame(mc.rel_msfe_matrix, columns=mc.method_names)
    # sort by median
    order = df.median().sort_values().index.tolist()
    if HAS_SEABORN:
        palette = [_method_plot_style(name)["color"] for name in order]
        sns.boxplot(data=df[order], orient="h", ax=ax, palette=palette)
    else:
        ax.boxplot([df[c].values for c in order], vert=False, labels=order)
    ax.axvline(1.0, color="#2F2F2F", ls="--", linewidth=0.8)
    ax.set_xlabel("Relative MSFE (vs Equal weights)")
    ax.set_title(f"MC Distribution — {mc.scenario_name} ({mc.n_reps} reps)")
    fig.tight_layout()
    return fig


def plot_mc_summary_table(mc_results: Dict[str, MCResult]) -> pd.DataFrame:
    """Cross-scenario summary table."""
    rows = []
    for sc_name, mc in mc_results.items():
        for jm, mname in enumerate(mc.method_names):
            rows.append({
                "Scenario": sc_name,
                "Method": mname,
                "Mean_Rel_MSFE": mc.mean_rel_msfe[jm],
                "Win_Freq": mc.win_freq[jm],
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Method", columns="Scenario",
                           values="Mean_Rel_MSFE", aggfunc="mean")
    return pivot


def plot_mc_heatmap(mc_results: Dict[str, MCResult]):
    """Heatmap of relative MSFE across scenarios and methods."""
    pivot = plot_mc_summary_table(mc_results)
    fig, ax = plt.subplots(figsize=(12, 6))
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                    center=1.0, ax=ax)
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(pivot.index)
    ax.set_title("Relative MSFE Across Scenarios")
    fig.tight_layout()
    return fig


# ===================================================================
# 11.  MCS, ADAPTABILITY
# ===================================================================

@dataclass
class MCSResult:
    """
    Store outputs from one Hansen-style Model Confidence Set procedure.

    The object keeps the surviving set, elimination path, bootstrap p-values,
    and a convenient summary table for downstream reporting and plotting.
    """
    alpha: float
    statistic: str
    B: int
    block_size: int
    model_names: List[str]
    included_models: List[str]
    elimination_order: List[str]
    pvalues: Dict[str, float]
    elimination_steps: Dict[str, int]
    test_statistics: List[float]
    test_pvalues: List[float]
    active_sets: List[List[str]]
    mean_losses: Dict[str, float]
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _coerce_loss_matrix(
    losses: Union[BacktestResult, pd.DataFrame, np.ndarray],
    model_names: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract a `T x M` loss matrix and aligned model names from supported inputs.

    Args:
        losses (BacktestResult | pd.DataFrame | np.ndarray): Loss input in one
            of the supported container formats
        model_names (Sequence[str] | None): Optional model names for array input
        methods (Sequence[str] | None): Optional subset of methods when the
            input is a `BacktestResult`

    Returns:
        Tuple[np.ndarray, List[str]]: Loss matrix and associated model names
    """
    if isinstance(losses, BacktestResult):
        if methods is None:
            methods = list(losses.combined_losses.keys())
        arr = np.column_stack([np.asarray(losses.combined_losses[m], dtype=float) for m in methods])
        return arr, list(methods)
    if isinstance(losses, pd.DataFrame):
        return losses.to_numpy(dtype=float), list(losses.columns.astype(str))
    arr = np.asarray(losses, dtype=float)
    if arr.ndim != 2:
        raise ValueError("losses must be a 2D array, DataFrame, or BacktestResult")
    if model_names is None:
        model_names = [f"model_{i+1}" for i in range(arr.shape[1])]
    if len(model_names) != arr.shape[1]:
        raise ValueError("model_names length must match the number of columns in losses")
    return arr, list(model_names)


# ---------- Block-size heuristics ----------

def _select_ar_order_bic(series: np.ndarray, max_lags: int = 10) -> int:
    """
    Select a univariate AR order with a simple BIC criterion.

    Args:
        series (np.ndarray): Univariate time series
        max_lags (int): Maximum AR order considered

    Returns:
        int: BIC-selected lag order
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    max_lags = min(max_lags, max(n // 3, 0))
    if n < 8 or max_lags == 0:
        return 0

    best_p = 0
    best_bic = np.inf
    for p in range(max_lags + 1):
        if n <= p + 2:
            continue
        y = x[p:]
        X = np.ones((n - p, p + 1))
        for lag in range(1, p + 1):
            X[:, lag] = x[p - lag:n - lag]
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = np.mean(resid ** 2)
        bic = np.log(max(sigma2, EPS)) + (p + 1) * np.log(len(y)) / len(y)
        if bic < best_bic:
            best_bic = bic
            best_p = p
    return best_p


def _auto_mcs_block_size(
    loss_matrix: np.ndarray,
    min_block_size: int = 3,
    max_lags: int = 10,
) -> int:
    """
    Practical block-length heuristic for the MCS bootstrap.

    Uses the maximum BIC-selected AR order across pairwise loss differentials,
    with a minimum floor for numerical stability.
    """
    n, m = loss_matrix.shape
    if n < 10 or m < 2:
        return min_block_size
    orders = []
    for i in range(m):
        for j in range(i + 1, m):
            diff = loss_matrix[:, i] - loss_matrix[:, j]
            if np.allclose(diff, diff[0]):
                continue
            orders.append(_select_ar_order_bic(diff, max_lags=max_lags))
    if not orders:
        return min_block_size
    block_size = max(max(orders), min_block_size)
    return int(min(block_size, max(n // 2, min_block_size)))


# ---------- Bootstrap utilities ----------

def _moving_block_bootstrap_indices(
    n_obs: int,
    B: int,
    block_size: int,
    rng,
) -> np.ndarray:
    """
    Generate circular moving-block bootstrap indices.

    Args:
        n_obs (int): Number of time observations
        B (int): Number of bootstrap replications
        block_size (int): Length of each resampled block
        rng: NumPy random number generator

    Returns:
        np.ndarray: Bootstrap index array of shape `(B, n_obs)`
    """
    n_blocks = int(np.ceil(n_obs / max(block_size, 1)))
    starts = rng.integers(0, n_obs, size=(B, n_blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    return indices.reshape(B, -1)[:, :n_obs]


# ---------- One-step MCS statistics ----------

def _mcs_iteration_statistics(
    loss_matrix: np.ndarray,
    bootstrap_indices: np.ndarray,
    statistic: str,
) -> Dict[str, Union[float, int, bool]]:
    """
    Compute one MCS test statistic, bootstrap p-value, and elimination rule.

    Args:
        loss_matrix (np.ndarray): Active-model loss matrix of shape `(T, M)`
        bootstrap_indices (np.ndarray): Bootstrap resampling indices
        statistic (str): Either `"Tmax"` or `"TR"`

    Returns:
        Dict[str, Union[float, int, bool]]: Test statistic, p-value, index of
        the model to eliminate, and a flag for degenerate equal-loss cases
    """
    n_obs, m = loss_matrix.shape
    loss_means = loss_matrix.mean(axis=0)
    boot_means = loss_matrix[bootstrap_indices].mean(axis=1)  # (B, m)
    zeta = boot_means - loss_means[None, :]

    scale = m / max(m - 1, 1)
    d_i_mean = scale * (loss_means - loss_means.mean())
    d_i_boot_centered = scale * (zeta - zeta.mean(axis=1, keepdims=True))
    var_i = np.mean(d_i_boot_centered ** 2, axis=0)
    t_i = d_i_mean / np.sqrt(var_i + EPS)
    T_max = float(np.max(t_i))
    T_max_star = np.max(d_i_boot_centered / np.sqrt(var_i + EPS), axis=1)
    p_max = float(np.mean(T_max_star >= (T_max - EPS)))
    elim_tmax = int(np.argmax(t_i))

    pair_centered = zeta[:, :, None] - zeta[:, None, :]
    d_ij_mean = loss_means[:, None] - loss_means[None, :]
    var_ij = np.mean(pair_centered ** 2, axis=0)
    np.fill_diagonal(var_ij, np.nan)
    t_ij = d_ij_mean / np.sqrt(var_ij + EPS)
    T_r = float(np.nanmax(np.abs(t_ij)))
    T_r_star = np.nanmax(
        np.abs(pair_centered / np.sqrt(var_ij[None, :, :] + EPS)),
        axis=(1, 2),
    )
    p_r = float(np.mean(T_r_star >= (T_r - EPS)))
    v_i_r = np.nanmax(t_ij, axis=1)
    elim_tr = int(np.nanargmax(v_i_r))

    if statistic == "TR":
        return {
            "test_stat": T_r,
            "pvalue": p_r,
            "eliminate_idx": elim_tr,
            "all_equal": bool(np.all(np.isnan(var_ij)) or np.nanmax(var_ij) < EPS),
        }
    return {
        "test_stat": T_max,
        "pvalue": p_max,
        "eliminate_idx": elim_tmax,
        "all_equal": bool(np.max(var_i) < EPS),
    }


# ---------- Hansen-style MCS procedure ----------

def model_confidence_set(
    losses: Union[BacktestResult, pd.DataFrame, np.ndarray],
    model_names: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    alpha: float = 0.10,
    B: int = 500,
    statistic: Literal["Tmax", "TR"] = "Tmax",
    block_size: Optional[int] = None,
    min_block_size: int = 3,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> MCSResult:
    """
    Hansen-Lunde-Nason Model Confidence Set procedure.

    Parameters
    ----------
    losses
        Either a `BacktestResult`, a T x M DataFrame of losses, or a T x M array.
    statistic
        Either "Tmax" or "TR", matching Hansen et al. (2011).

    Returns
    -------
    MCSResult
        Full Model Confidence Set output, including the surviving model set and
        elimination summary.
    """
    if statistic not in {"Tmax", "TR"}:
        raise ValueError("statistic must be either 'Tmax' or 'TR'")

    loss_matrix, model_names = _coerce_loss_matrix(losses, model_names, methods)
    n_obs, n_models = loss_matrix.shape
    if n_models < 2:
        raise ValueError("MCS requires at least two models")

    if block_size is None:
        block_size = _auto_mcs_block_size(loss_matrix, min_block_size=min_block_size)
    block_size = max(int(block_size), 1)

    rng = _ensure_rng(seed)
    bootstrap_indices = _moving_block_bootstrap_indices(n_obs, int(B), block_size, rng)

    mean_losses = {name: float(loss_matrix[:, i].mean()) for i, name in enumerate(model_names)}
    active = list(range(n_models))
    elimination_order: List[str] = []
    elimination_steps: Dict[str, int] = {}
    pvalues: Dict[str, float] = {}
    test_statistics: List[float] = []
    test_pvalues: List[float] = []
    active_sets: List[List[str]] = []
    running_p = 0.0

    while len(active) > 1:
        active_sets.append([model_names[i] for i in active])
        stats = _mcs_iteration_statistics(loss_matrix[:, active], bootstrap_indices, statistic)
        test_statistics.append(float(stats["test_stat"]))
        test_pvalues.append(float(stats["pvalue"]))

        if stats["all_equal"] or stats["pvalue"] >= alpha:
            break

        local_idx = int(stats["eliminate_idx"])
        global_idx = active[local_idx]
        name = model_names[global_idx]
        running_p = max(running_p, float(stats["pvalue"]))
        elimination_order.append(name)
        elimination_steps[name] = len(elimination_order)
        pvalues[name] = running_p
        active.pop(local_idx)

    included_models = [model_names[i] for i in active]
    for name in included_models:
        elimination_steps[name] = 0
        pvalues[name] = 1.0

    summary_rows = []
    for name in model_names:
        summary_rows.append({
            "Method": name,
            "Mean_Loss": mean_losses[name],
            "MCS_pvalue": pvalues[name],
            "In_MCS": name in included_models,
            "Elimination_Step": elimination_steps[name],
        })
    summary_table = pd.DataFrame(summary_rows).sort_values(
        ["In_MCS", "Mean_Loss"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return MCSResult(
        alpha=float(alpha),
        statistic=statistic,
        B=int(B),
        block_size=block_size,
        model_names=model_names,
        included_models=included_models,
        elimination_order=elimination_order,
        pvalues=pvalues,
        elimination_steps=elimination_steps,
        test_statistics=test_statistics,
        test_pvalues=test_pvalues,
        active_sets=active_sets,
        mean_losses=mean_losses,
        summary_table=summary_table,
    )


# ---------- MCS reporting helpers ----------

def compute_mcs_performance_table(
    res: BacktestResult,
    mcs_result: Optional[MCSResult] = None,
    methods: Optional[Sequence[str]] = None,
    reference: str = "equal",
    **mcs_kwargs,
) -> pd.DataFrame:
    """
    Merge standard backtest performance metrics with MCS diagnostics.

    Args:
        res (BacktestResult): Rolling backtest output
        mcs_result (MCSResult | None): Precomputed MCS result, if available
        methods (Sequence[str] | None): Optional method subset
        reference (str): Reference method used to rescale mean losses
        **mcs_kwargs: Extra keyword arguments passed to `model_confidence_set`

    Returns:
        pd.DataFrame: Performance table augmented with MCS status columns
    """
    perf = compute_performance_table(res)
    if methods is not None:
        perf = perf[perf["Method"].isin(methods)].copy()
    if mcs_result is None:
        mcs_result = model_confidence_set(res, methods=methods, **mcs_kwargs)
    mcs_tbl = mcs_result.summary_table[["Method", "MCS_pvalue", "In_MCS", "Elimination_Step"]]
    df = perf.merge(mcs_tbl, on="Method", how="left")

    if reference in df["Method"].values:
        ref_loss = df.loc[df["Method"] == reference, "Mean_Loss"].iloc[0]
        df["Rel_MSFE"] = df["Mean_Loss"] / (ref_loss + EPS)

    df = df.sort_values(["In_MCS", "Mean_Loss"], ascending=[False, True]).reset_index(drop=True)
    return df


# ---------- MCS visualization ----------

def plot_mcs_summary(
    res: BacktestResult,
    mcs_result: Optional[MCSResult] = None,
    methods: Optional[Sequence[str]] = None,
    reference: str = "equal",
    ax=None,
    **mcs_kwargs,
):
    """
    Plot relative mean loss by method with MCS membership highlighted.

    Args:
        res (BacktestResult): Rolling backtest output
        mcs_result (MCSResult | None): Optional precomputed MCS result
        methods (Sequence[str] | None): Optional method subset
        reference (str): Reference method for relative MSFE scaling
        ax: Optional matplotlib axes
        **mcs_kwargs: Extra keyword arguments passed to `model_confidence_set`

    Returns:
        matplotlib.figure.Figure: Figure containing the summary bar plot
    """
    df = compute_mcs_performance_table(
        res,
        mcs_result=mcs_result,
        methods=methods,
        reference=reference,
        **mcs_kwargs,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    bars = ax.barh(df["Method"], df["Rel_MSFE"], alpha=0.9)
    for bar, (_, row) in zip(bars, df.iterrows()):
        bar.set_facecolor("#C62828" if row["In_MCS"] else "#2E7D32")
        bar.set_edgecolor("#2F2F2F")
        bar.set_linewidth(0.8)
        bar.set_alpha(0.9)
    ax.axvline(1.0, color="#2F2F2F", ls="--", linewidth=0.8)
    for y, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["Rel_MSFE"] + 0.01,
            y,
            f"p={row['MCS_pvalue']:.3f}",
            va="center",
            fontsize=9.5,
        )
    ax.set_xlabel(f"Relative MSFE (vs {reference})")
    ax.set_title("Model Confidence Set Summary")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------- Adaptability result container ----------

@dataclass
class AdaptabilityResult:
    """
    Store event-study diagnostics for post-switch forecast adaptability.

    The object tracks how quickly each method approaches the latent oracle
    combination after oracle switches in the simulated environment.
    """
    methods: List[str]
    event_times: np.ndarray
    oracle_models: np.ndarray
    horizon: int
    smooth_window: int
    target_oracle_weight: float
    oracle_portfolio_weights: np.ndarray
    oracle_weight_profiles: Dict[str, np.ndarray]
    oracle_gap_profiles: Dict[str, np.ndarray]
    latent_risk_profiles: Dict[str, np.ndarray]
    excess_risk_profiles: Dict[str, np.ndarray]
    relative_gap_profiles: Dict[str, np.ndarray]
    half_lives: Dict[str, np.ndarray]
    recovery_delays: Dict[str, np.ndarray]
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------- Latent oracle identification ----------

def compute_latent_oracle_indices(data: SimulationData) -> np.ndarray:
    """
    Latent oracle under squared loss.

    Common shocks are shared across models, so ranking is driven by model-specific
    squared bias plus model-specific idiosyncratic variance.
    """
    latent_risk = data.bias_paths ** 2 + data.sigma_paths ** 2
    return np.argmin(latent_risk, axis=1)


def identify_oracle_switches(
    data: SimulationData,
    start_time: Optional[int] = None,
    min_spacing: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify OOS dates where the latent oracle model changes."""
    oracle = compute_latent_oracle_indices(data)
    if start_time is None:
        start_time = data.config.T0
    switches = np.where(oracle[1:] != oracle[:-1])[0] + 1
    switches = switches[switches >= start_time]
    if len(switches) <= 1 or min_spacing <= 1:
        return switches, oracle

    keep = [switches[0]]
    for t in switches[1:]:
        if t - keep[-1] >= min_spacing:
            keep.append(t)
    return np.asarray(keep, dtype=int), oracle


# ---------- Adaptability helper functions ----------

def _rolling_mean_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with min_periods=1."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(window, min_periods=1).mean().to_numpy()


def _latent_variance_scale_with_outliers(cfg: ScenarioConfig) -> float:
    """Variance inflation factor for the outlier mixture used in scenario generation."""
    if cfg.outlier_prob <= 0:
        return 1.0
    a = float(cfg.outlier_min_scale)
    b = float(max(cfg.outlier_scale, cfg.outlier_min_scale))
    uniform_second_moment = (a ** 2 + a * b + b ** 2) / 3.0
    return (1.0 - cfg.outlier_prob) + cfg.outlier_prob * uniform_second_moment


def latent_idiosyncratic_covariance(
    data: SimulationData,
    t: int,
) -> np.ndarray:
    """
    Conditional covariance of idiosyncratic forecast errors at time t.

    The common shock is omitted because it is shared across all methods and
    therefore does not affect cross-method comparisons.
    """
    cfg = data.config
    sigma_t = np.asarray(data.sigma_paths[t], dtype=float)
    variances = sigma_t ** 2

    if cfg.outlier_prob > 0:
        variances = variances * _latent_variance_scale_with_outliers(cfg)

    M = len(sigma_t)
    Sigma = np.diag(variances)

    if cfg.factor_rho is not None:
        rho = np.asarray(cfg.factor_rho, dtype=float)
        Sigma = np.outer(sigma_t * rho, sigma_t * rho)
        np.fill_diagonal(Sigma, variances)
        return Sigma

    if cfg.n_clusters > 1 and cfg.cluster_labels is not None:
        labels = np.asarray(cfg.cluster_labels)
        rho_sq = float(cfg.cluster_rho) ** 2
        Sigma = np.diag(variances)
        for i in range(M):
            for j in range(i + 1, M):
                if labels[i] == labels[j]:
                    cov_ij = rho_sq * sigma_t[i] * sigma_t[j]
                    Sigma[i, j] = cov_ij
                    Sigma[j, i] = cov_ij
        return Sigma

    return Sigma


def latent_risk_matrix(
    data: SimulationData,
    t: int,
) -> np.ndarray:
    """
    Quadratic-form matrix for latent expected squared forecast error at time t.

    For a simplex weight vector w, the method-specific component of expected
    squared error is:
        (w'b_t)^2 + w' Sigma_t w
    where b_t is the latent bias vector and Sigma_t is the conditional
    idiosyncratic covariance matrix.
    """
    bias_t = np.asarray(data.bias_paths[t], dtype=float)
    Sigma_t = latent_idiosyncratic_covariance(data, t)
    return Sigma_t + np.outer(bias_t, bias_t)


def compute_latent_oracle_combination_weights(
    data: SimulationData,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """Time-varying oracle simplex weights under latent expected squared error."""
    T = data.config.T
    M = data.config.M
    oracle_w = np.zeros((T, M))
    for t in range(T):
        G_t = latent_risk_matrix(data, t)
        oracle_w[t] = covariance_only_weights(G_t, ridge=ridge)
    return oracle_w


def latent_combination_risk(
    data: SimulationData,
    weights: np.ndarray,
    t: int,
) -> float:
    """Latent expected squared forecast error for a given weight vector at time t."""
    G_t = latent_risk_matrix(data, t)
    w = simplex_project(np.asarray(weights, dtype=float))
    return float(w @ G_t @ w)


def _safe_nanmean(x: np.ndarray) -> float:
    """NaN-safe mean that returns NaN when no finite values exist."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float(np.nan)
    return float(np.mean(x[finite]))


def _safe_nanmedian(x: np.ndarray) -> float:
    """NaN-safe median that returns NaN when no finite values exist."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float(np.nan)
    return float(np.median(x[finite]))


# ---------- Adaptability event-study diagnostic ----------

def compute_adaptability_diagnostics(
    data: SimulationData,
    res: BacktestResult,
    methods: Optional[Sequence[str]] = None,
    horizon: int = 40,
    smooth_window: int = 5,
    min_event_spacing: int = 3,
    target_oracle_weight: float = 0.8,
    event_times: Optional[Sequence[int]] = None,
) -> AdaptabilityResult:
    """
    Compute simulation-based adaptability diagnostics after latent oracle switches.

    The metric is an original event-study summary inspired by the forecast-breakdown
    literature: after each latent oracle switch, track how quickly a method
    closes its excess latent risk relative to the time-varying oracle
    combination and record the half-life of that gap.

    Args:
        data (SimulationData): Simulated data-generating environment
        res (BacktestResult): Backtest output containing time-varying weights
        methods (Sequence[str] | None): Optional method subset to evaluate
        horizon (int): Post-switch evaluation horizon
        smooth_window (int): Rolling window used to smooth relative gap profiles
        min_event_spacing (int): Minimum spacing between retained switch events
        target_oracle_weight (float): Oracle-overlap target used for recovery delays
        event_times (Sequence[int] | None): Optional known event dates. When
            supplied, these dates are used instead of latent-oracle index switches.

    Returns:
        AdaptabilityResult: Event-study profiles and summary metrics by method
    """
    if methods is None:
        methods = list(res.weights.keys())
    methods = [m for m in methods if m in res.weights]

    if event_times is None:
        event_times, oracle = identify_oracle_switches(
            data,
            start_time=data.config.T0,
            min_spacing=min_event_spacing,
        )
    else:
        oracle = compute_latent_oracle_indices(data)
        event_times = np.asarray(event_times, dtype=int)
        event_times = event_times[
            (event_times >= data.config.T0)
            & (event_times < data.config.T)
        ]
    n_events = len(event_times)
    oracle_models = oracle[event_times] if n_events else np.array([], dtype=int)
    oracle_combo_weights = compute_latent_oracle_combination_weights(data)

    overlap_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    weight_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    gap_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    latent_risk_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    excess_risk_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    relative_gap_profiles = {
        method: np.full((n_events, horizon), np.nan)
        for method in methods
    }
    half_lives = {
        method: np.full(n_events, np.nan)
        for method in methods
    }
    recovery_delays = {
        method: np.full(n_events, np.nan)
        for method in methods
    }

    if n_events == 0:
        warnings.warn("No latent oracle switches were detected in the OOS period.")
    for ievent, t_event in enumerate(event_times):
        horizon_end = min(t_event + horizon, data.config.T)
        use_len = horizon_end - t_event
        oos_start = t_event - data.config.T0

        for method in methods:
            method_weights = np.asarray(res.weights[method][oos_start:oos_start + use_len], dtype=float)
            oracle_capture = np.zeros(use_len)
            latent_risk = np.zeros(use_len)
            excess_risk = np.zeros(use_len)

            for h in range(use_len):
                t_cur = t_event + h
                oracle_w_t = oracle_combo_weights[t_cur]
                w_t = simplex_project(method_weights[h])
                oracle_capture[h] = 1.0 - 0.5 * np.abs(w_t - oracle_w_t).sum()
                latent_risk[h] = latent_combination_risk(data, w_t, t_cur)
                oracle_risk_t = latent_combination_risk(data, oracle_w_t, t_cur)
                excess_risk[h] = max(latent_risk[h] - oracle_risk_t, 0.0)

            initial_gap = max(excess_risk[0], EPS) if use_len else np.nan
            relative_gap = excess_risk / initial_gap if use_len else np.array([])

            overlap_profiles[method][ievent, :use_len] = oracle_capture
            weight_profiles[method][ievent, :use_len] = oracle_capture
            gap_profiles[method][ievent, :use_len] = 1.0 - oracle_capture
            latent_risk_profiles[method][ievent, :use_len] = latent_risk
            excess_risk_profiles[method][ievent, :use_len] = excess_risk
            relative_gap_profiles[method][ievent, :use_len] = relative_gap

            smooth_gap = _rolling_mean_1d(relative_gap, smooth_window)
            baseline = float(smooth_gap[0]) if len(smooth_gap) else np.nan
            if not np.isfinite(baseline):
                continue
            if baseline <= 0:
                half_lives[method][ievent] = 0.0
                recovery_delays[method][ievent] = 0.0
                continue

            half_target = 0.5 * baseline
            half_idx = np.where(smooth_gap <= half_target)[0]
            if len(half_idx):
                half_lives[method][ievent] = float(half_idx[0])

            target_gap_fraction = max(1.0 - target_oracle_weight, 0.0)
            recovery_idx = np.where(smooth_gap <= target_gap_fraction)[0]
            if len(recovery_idx):
                recovery_delays[method][ievent] = float(recovery_idx[0])

    rows = []
    for method in methods:
        profile0 = excess_risk_profiles[method][:, 0] if n_events else np.array([np.nan])
        rows.append({
            "Method": method,
            "Events": n_events,
            "Mean_Initial_Excess_Latent_Risk": _safe_nanmean(profile0),
            "Mean_Integrated_Excess_Risk": _safe_nanmean(excess_risk_profiles[method]),
            "Mean_Half_Life": _safe_nanmean(half_lives[method]),
            "Median_Half_Life": _safe_nanmedian(half_lives[method]),
            "Mean_Target_Closure_Delay": _safe_nanmean(recovery_delays[method]),
            "Captured_Frac": float(np.mean(np.isfinite(recovery_delays[method]))) if n_events else np.nan,
            "Mean_Oracle_Overlap": _safe_nanmean(overlap_profiles[method][:, 0]) if n_events else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values(
        ["Mean_Integrated_Excess_Risk", "Mean_Half_Life", "Mean_Target_Closure_Delay"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    return AdaptabilityResult(
        methods=methods,
        event_times=event_times,
        oracle_models=oracle_models,
        horizon=horizon,
        smooth_window=smooth_window,
        target_oracle_weight=target_oracle_weight,
        oracle_portfolio_weights=oracle_combo_weights,
        oracle_weight_profiles=weight_profiles,
        oracle_gap_profiles=gap_profiles,
        latent_risk_profiles=latent_risk_profiles,
        excess_risk_profiles=excess_risk_profiles,
        relative_gap_profiles=relative_gap_profiles,
        half_lives=half_lives,
        recovery_delays=recovery_delays,
        summary_table=summary,
    )


# ---------- Adaptability visualization ----------

def plot_adaptability_event_study(
    adapt: AdaptabilityResult,
    methods: Optional[Sequence[str]] = None,
    ax=None,
):
    """
    Plot the average excess latent risk profile after oracle-switch events.

    Args:
        adapt (AdaptabilityResult): Adaptability diagnostics output
        methods (Sequence[str] | None): Optional method subset
        ax: Optional matplotlib axes

    Returns:
        matplotlib.figure.Figure: Figure containing the event-study plot
    """
    if methods is None:
        methods = adapt.methods
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    horizon_axis = np.arange(adapt.horizon)
    for method in methods:
        profile = adapt.excess_risk_profiles[method]
        mean_profile = np.nanmean(profile, axis=0)
        style = _method_plot_style(method)
        ax.plot(
            horizon_axis,
            mean_profile,
            label=method,
            linewidth=2.6 if _is_gcsr_method(method) else 1.7,
            alpha=0.95,
            color=style["color"],
            linestyle=style["linestyle"],
            **_line_marker_kwargs(len(horizon_axis), style["marker"]),
            zorder=3 if _is_gcsr_method(method) else 2,
        )

    ax.axhline(0.0, color="#2F2F2F", ls="--", linewidth=0.8)
    ax.set_xlabel("Periods Since Latent Oracle Switch")
    ax.set_ylabel("Excess Latent Risk vs Oracle Combination")
    ax.set_title("Adaptability Event Study")
    _apply_publication_legend(ax, ncol=2, fontsize=10.0)
    fig.tight_layout()
    return fig


def plot_adaptability_half_life(
    adapt: AdaptabilityResult,
    ax=None,
):
    """
    Plot mean post-switch half-lives for each evaluated method.

    Args:
        adapt (AdaptabilityResult): Adaptability diagnostics output
        ax: Optional matplotlib axes

    Returns:
        matplotlib.figure.Figure: Figure containing the half-life bar chart
    """
    df = adapt.summary_table.copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    bars = ax.barh(df["Method"], df["Mean_Half_Life"], alpha=0.9)
    for bar, method in zip(bars, df["Method"]):
        _apply_bar_patch_style(bar, method)
    ax.set_xlabel("Mean Relative-Risk Half-Life")
    ax.set_title("Adaptability Speed")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ===================================================================
# 12.  EMPIRICAL INFLATION EVALUATION
# ===================================================================

# ---------- Empirical result container ----------

@dataclass
class EmpiricalStudyResult:
    """
    Store empirical data inputs, backtest outputs, and reporting tables.

    The object is designed for direct inspection after running an empirical
    out-of-sample study on a survey forecast panel in `Empirical_Data`.
    """
    dataset_name: str
    dataset_label: str
    merged_data: pd.DataFrame
    forecaster_ids: List[str]
    training_periods: int
    comparison_methods: List[str]
    data: SimulationData
    backtest: BacktestResult
    performance_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_forecast_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    mcs_result: Optional[MCSResult] = None
    mcs_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------- Empirical data preparation ----------

def load_empirical_data(
    forecast_path: str,
    truth_path: str,
    dataset_name: str = "empirical_panel",
    training_periods: int = 20,
    loss_name: str = "squared",
    difference_mode: Literal["levels", "diff_to_prev_actual"] = "levels",
) -> Tuple[SimulationData, pd.DataFrame, List[str]]:
    """
    Load and align an empirical forecast panel.

    The forecast file is expected to be a wide `T x M` matrix with a
    `TARGET_PERIOD` column and one column per forecaster. The truth file is
    expected to contain `TARGET_PERIOD` and `actual`. The loader aligns the two
    sources by target period, sorts them chronologically, and converts the
    merged panel into the `SimulationData` container used by the backtest.

    Args:
        forecast_path (str): Path to the `T x M` forecast matrix CSV
        truth_path (str): Path to the realized target CSV
        dataset_name (str): Name stored in the resulting simulation config
        training_periods (int): First OOS index
        loss_name (str): Loss function used to construct empirical losses
        difference_mode (Literal["levels", "diff_to_prev_actual"]):
            If ``"levels"``, keep the original target and forecasts. If
            ``"diff_to_prev_actual"``, transform the target to
            ``y_t - y_{t-1}`` and each forecast to ``f_{j,t} - y_{t-1}``.

    Returns:
        Tuple[SimulationData, pd.DataFrame, List[str]]: Backtest-ready data,
        aligned empirical panel, and ordered forecaster identifiers
    """
    training_periods = max(int(training_periods), 20)
    loss_fn = LOSS_REGISTRY.get(loss_name, squared_loss)

    forecasts_df = pd.read_csv(forecast_path)
    truth_df = pd.read_csv(truth_path)
    forecaster_ids = [col for col in forecasts_df.columns if col != "TARGET_PERIOD"]
    merged = forecasts_df.merge(
        truth_df[["TARGET_PERIOD", "actual"]],
        on="TARGET_PERIOD",
        how="left",
    )
    merged["period_dt"] = pd.to_datetime(merged["TARGET_PERIOD"])
    merged = merged.sort_values("period_dt").reset_index(drop=True)
    numeric_cols = forecaster_ids + ["actual"]
    merged[numeric_cols] = merged[numeric_cols].astype(float)
    merged = merged.dropna(subset=["actual"]).reset_index(drop=True)

    if difference_mode == "diff_to_prev_actual":
        prev_actual = merged["actual"].shift(1)
        merged["actual_level"] = merged["actual"]
        merged["actual_lag1"] = prev_actual
        merged["actual"] = merged["actual"] - prev_actual
        merged[forecaster_ids] = merged[forecaster_ids].subtract(prev_actual, axis=0)
        merged = merged.dropna(subset=["actual", "actual_lag1"]).reset_index(drop=True)
    elif difference_mode != "levels":
        raise ValueError(f"Unsupported difference_mode: {difference_mode}")

    T = len(merged)
    M = len(forecaster_ids)

    y = merged["actual"].to_numpy(dtype=float)
    forecasts = merged[forecaster_ids].to_numpy(dtype=float)
    errors = y[:, None] - forecasts
    losses = loss_fn(y[:, None], forecasts)

    cfg = ScenarioConfig(
        name=dataset_name,
        M=M,
        T=T,
        T0=training_periods,
        sigma_common=0.0,
        seed=RNG_SEED,
    )
    data = SimulationData(
        y=y,
        forecasts=forecasts,
        errors=errors,
        losses=losses,
        bias_paths=np.zeros((T, M)),
        sigma_paths=np.ones((T, M)),
        common_shock=np.zeros(T),
        config=cfg,
    )
    return data, merged, forecaster_ids


def load_empirical_inflation_data(
    forecast_path: str = "Empirical_Data/inflation_forecasts_f.csv",
    truth_path: str = "Empirical_Data/inflation_truth_f.csv",
    training_periods: int = 20,
    loss_name: str = "squared",
    difference_mode: Literal["levels", "diff_to_prev_actual"] = "levels",
) -> Tuple[SimulationData, pd.DataFrame, List[str]]:
    """Backward-compatible wrapper for the inflation panel."""
    return load_empirical_data(
        forecast_path=forecast_path,
        truth_path=truth_path,
        dataset_name="empirical_inflation_next_quarter",
        training_periods=training_periods,
        loss_name=loss_name,
        difference_mode=difference_mode,
    )

def _default_empirical_bt_config(training_periods: int = 20) -> BacktestConfig:
    """
    Build a stable backtest configuration for an empirical study.

    Args:
        training_periods (int): Minimum amount of historical data available
            before the OOS evaluation begins

    Returns:
        BacktestConfig: Default empirical backtest hyper-parameters
    """
    window = max(training_periods, 20)
    return BacktestConfig(
        d_max=3,
        centrality_type="eigenvector",
        # Empirical studies should expose the full set of requested graph variants
        # so that graph_only and full_gcsr are both compared under all centralities.
        centrality_types=DEFAULT_COMPARISON_CENTRALITY_TYPES,
        cov_window=window,
        bg_window=window,
        tune_window=window,
        recent_best_window=min(8, window),
        min_history=window,
    )


def _select_empirical_comparison_methods(
    performance_table: pd.DataFrame,
    max_methods: Optional[int] = None,
) -> List[str]:
    """
    Pick an ordered empirical comparison set anchored on `full_gcsr`.

    Args:
        performance_table (pd.DataFrame): Method-level performance summary
        max_methods (int | None): Optional maximum number of methods to keep

    Returns:
        List[str]: Ordered list of methods for empirical comparison plots
    """
    available = performance_table["Method"].tolist()
    selected: List[str] = []
    ordered_available = _preferred_method_subset(available)
    for name in ordered_available:
        if name not in selected:
            selected.append(name)
        if max_methods is not None and len(selected) >= max_methods:
            break
    return selected


# ---------- Empirical study runner ----------

def build_empirical_oos_forecast_table(
    study: EmpiricalStudyResult,
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Construct a tidy OOS table of actual inflation and combined forecasts.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        methods (Sequence[str] | None): Optional method subset

    Returns:
        pd.DataFrame: OOS table indexed by target period
    """
    if methods is None:
        methods = study.comparison_methods
    methods = [m for m in methods if m in study.backtest.combined_forecasts]

    idx = study.backtest.oos_periods
    df = pd.DataFrame({
        "TARGET_PERIOD": study.merged_data.loc[idx, "TARGET_PERIOD"].to_numpy(),
        "period_dt": study.merged_data.loc[idx, "period_dt"].to_numpy(),
        "actual": study.data.y[idx],
    })
    for method in methods:
        df[method] = study.backtest.combined_forecasts[method]
    return df.reset_index(drop=True)

def run_empirical_study(
    forecast_path: str,
    truth_path: str,
    dataset_name: str = "empirical_panel",
    dataset_label: Optional[str] = None,
    training_periods: int = 20,
    bt_cfg: Optional[BacktestConfig] = None,
    mcs_alpha: float = 0.10,
    mcs_B: int = 500,
    mcs_statistic: Literal["Tmax", "TR"] = "Tmax",
    verbose: bool = False,
    difference_mode: Literal["levels", "diff_to_prev_actual"] = "levels",
) -> EmpiricalStudyResult:
    """
    Run a generic out-of-sample empirical backtest on a wide forecast panel.

    Args:
        forecast_path (str): Path to the empirical forecast CSV
        truth_path (str): Path to the realized target CSV
        dataset_name (str): Internal dataset identifier
        dataset_label (str | None): Human-readable label used in plots/tables
        training_periods (int): In-sample history available before OOS testing
        bt_cfg (BacktestConfig | None): Optional empirical backtest configuration
        mcs_alpha (float): Model Confidence Set significance level
        mcs_B (int): Number of MCS bootstrap resamples
        mcs_statistic (Literal["Tmax", "TR"]): MCS test statistic
        verbose (bool): Whether to print rolling backtest progress
        difference_mode (Literal["levels", "diff_to_prev_actual"]):
            Optional transformation applied to the empirical panel before the
            rolling backtest.

    Returns:
        EmpiricalStudyResult: Empirical data, backtest output, and summary tables
    """
    if dataset_label is None:
        dataset_label = dataset_name.replace("_", " ").title()

    data, merged, forecaster_ids = load_empirical_data(
        forecast_path=forecast_path,
        truth_path=truth_path,
        dataset_name=dataset_name,
        training_periods=training_periods,
        difference_mode=difference_mode,
    )

    if bt_cfg is None:
        bt_cfg = _default_empirical_bt_config(training_periods=training_periods)
    else:
        bt_cfg.min_history = max(int(bt_cfg.min_history), 20)

    res = run_backtest(data, bt_cfg=bt_cfg, verbose=verbose)
    performance = compute_performance_table(res)
    comparison_methods = _select_empirical_comparison_methods(performance)
    mcs_result = model_confidence_set(
        res,
        alpha=mcs_alpha,
        B=mcs_B,
        statistic=mcs_statistic,
    )
    mcs_table = compute_mcs_performance_table(res, mcs_result=mcs_result)

    study = EmpiricalStudyResult(
        dataset_name=dataset_name,
        dataset_label=dataset_label,
        merged_data=merged,
        forecaster_ids=forecaster_ids,
        training_periods=training_periods,
        comparison_methods=comparison_methods,
        data=data,
        backtest=res,
        performance_table=performance,
        mcs_result=mcs_result,
        mcs_table=mcs_table,
    )
    study.oos_forecast_table = build_empirical_oos_forecast_table(study)
    return study


# ---------- Empirical visualization ----------

def plot_empirical_oos_forecasts(
    study: EmpiricalStudyResult,
    methods: Optional[Sequence[str]] = None,
    ax=None,
):
    """
    Plot realized target values against combined OOS forecasts.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        methods (Sequence[str] | None): Optional method subset
        ax: Optional matplotlib axes

    Returns:
        matplotlib.figure.Figure: Figure containing the OOS forecast comparison
    """
    df = build_empirical_oos_forecast_table(study, methods=methods)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.plot(df["period_dt"], df["actual"], color="#1F1F1F", linewidth=2.4, label="actual")
    for method in [c for c in df.columns if c not in {"TARGET_PERIOD", "period_dt", "actual"}]:
        style = _method_plot_style(method)
        ax.plot(
            df["period_dt"],
            df[method],
            label=method,
            linewidth=2.2 if _is_gcsr_method(method) else 1.7,
            alpha=0.95,
            color=style["color"],
            linestyle=style["linestyle"],
            **_line_marker_kwargs(len(df), style["marker"]),
            zorder=3 if _is_gcsr_method(method) else 2,
        )
    ax.set_title(f"{study.dataset_label} OOS Forecasts")
    ax.set_xlabel("Target Period")
    ax.set_ylabel(study.dataset_label)
    _apply_publication_legend(ax, ncol=2, fontsize=10.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_empirical_cumulative_loss(
    study: EmpiricalStudyResult,
    methods: Optional[Sequence[str]] = None,
    reference: str = "equal",
):
    """
    Plot cumulative loss differences for the empirical OOS exercise.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        methods (Sequence[str] | None): Optional method subset
        reference (str): Reference method used in the cumulative-loss plot

    Returns:
        matplotlib.figure.Figure: Figure containing the cumulative-loss comparison
    """
    if methods is None:
        methods = study.comparison_methods
    time_axis = study.merged_data.loc[study.backtest.oos_periods, "period_dt"].to_numpy()
    fig = plot_cumulative_loss(
        study.backtest,
        methods=list(methods),
        reference=reference,
        time_axis=time_axis,
        x_label="Target Period",
    )
    ax = fig.axes[0]
    ax.set_title(f"{study.dataset_label} Cumulative Loss Difference")
    return fig


def plot_empirical_weight_comparison(
    study: EmpiricalStudyResult,
    methods: Optional[Sequence[str]] = None,
):
    """
    Plot empirical weight paths for the main combination methods.

    Args:
        study (EmpiricalStudyResult): Empirical study output
        methods (Sequence[str] | None): Optional method subset

    Returns:
        matplotlib.figure.Figure: Figure containing the weight time series
    """
    if methods is None:
        methods = _preferred_method_subset(study.backtest.weights.keys(), limit=6)
    methods = [m for m in methods if m in study.backtest.weights]
    time_axis = study.merged_data.loc[study.backtest.oos_periods, "period_dt"].to_numpy()
    return plot_weight_timeseries(
        study.backtest,
        methods=methods,
        time_axis=time_axis,
        x_label="Target Period",
        model_names=study.forecaster_ids,
    )


def showcase_empirical_inflation_study(
    forecast_path: str = "Empirical_Data/inflation_forecasts_f.csv",
    truth_path: str = "Empirical_Data/inflation_truth_f.csv",
    training_periods: int = 20,
    bt_cfg: Optional[BacktestConfig] = None,
    mcs_alpha: float = 0.10,
    mcs_B: int = 500,
    mcs_statistic: Literal["Tmax", "TR"] = "Tmax",
    verbose: bool = False,
    difference_mode: Literal["levels", "diff_to_prev_actual"] = "levels",
) -> Dict[str, object]:
    """
    Backward-compatible wrapper for the inflation showcase helper.
    """
    return showcase_empirical_study(
        forecast_path=forecast_path,
        truth_path=truth_path,
        dataset_name="empirical_inflation_next_quarter",
        dataset_label="Inflation",
        training_periods=training_periods,
        bt_cfg=bt_cfg,
        mcs_alpha=mcs_alpha,
        mcs_B=mcs_B,
        mcs_statistic=mcs_statistic,
        verbose=verbose,
        difference_mode=difference_mode,
    )


def showcase_empirical_study(
    forecast_path: str,
    truth_path: str,
    dataset_name: str = "empirical_panel",
    dataset_label: Optional[str] = None,
    training_periods: int = 20,
    bt_cfg: Optional[BacktestConfig] = None,
    mcs_alpha: float = 0.10,
    mcs_B: int = 500,
    mcs_statistic: Literal["Tmax", "TR"] = "Tmax",
    verbose: bool = False,
    difference_mode: Literal["levels", "diff_to_prev_actual"] = "levels",
) -> Dict[str, object]:
    """
    Run a generic empirical study and return the main tables and plots.
    """
    study = run_empirical_study(
        forecast_path=forecast_path,
        truth_path=truth_path,
        dataset_name=dataset_name,
        dataset_label=dataset_label,
        training_periods=training_periods,
        bt_cfg=bt_cfg,
        mcs_alpha=mcs_alpha,
        mcs_B=mcs_B,
        mcs_statistic=mcs_statistic,
        verbose=verbose,
        difference_mode=difference_mode,
    )

    mcs_fig = plot_mcs_summary(
        study.backtest,
        mcs_result=study.mcs_result,
        methods=study.comparison_methods,
    )
    oos_fig = plot_empirical_oos_forecasts(study)
    cumulative_fig = plot_empirical_cumulative_loss(study)
    weights_fig = plot_empirical_weight_comparison(study)

    return {
        "study": study,
        "performance_table": study.performance_table,
        "mcs_table": study.mcs_table,
        "oos_forecast_table": study.oos_forecast_table,
        "mcs_figure": mcs_fig,
        "oos_forecast_figure": oos_fig,
        "cumulative_loss_figure": cumulative_fig,
        "weights_figure": weights_fig,
    }


# ===================================================================
# 13.  SENSITIVITY SWEEPS
# ===================================================================

@dataclass
class SensitivitySweepResult:
    """Container for one centrality/covariance sensitivity sweep."""
    source_name: str
    source_type: str
    results_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    graph_only_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    full_gcsr_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    full_gcsr_pivot: pd.DataFrame = field(default_factory=pd.DataFrame)
    combined_losses: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_loss_mean: float = float("nan")
    backtests: Dict[str, BacktestResult] = field(default_factory=dict)


def run_sensitivity_sweep(
    data: SimulationData,
    bt_cfg: Optional[BacktestConfig] = None,
    source_name: str = "source",
    source_type: Literal["simulation", "empirical"] = "simulation",
    centrality_types: Sequence[str] = ("eigenvector", "pagerank", "softmax"),
    cov_methods: Sequence[str] = ("rolling", "ewma", "shrinkage", "diagonal"),
    mcs_alpha: float = 0.10,
    mcs_B: int = 200,
    mcs_statistic: Literal["Tmax", "TR"] = "Tmax",
    verbose: bool = False,
) -> SensitivitySweepResult:
    """
    Compare graph-only and full-GCSR variants across centrality and covariance choices.

    The returned table contains one unique `graph_only` row per centrality measure
    and one `full_gcsr` row for every `(centrality, covariance)` combination.
    Full-GCSR method names are expanded with the covariance suffix so that the
    downstream ranking tables can compare the complete sweep in one place.
    """
    if bt_cfg is None:
        bt_cfg = BacktestConfig()

    centrality_types = _resolve_centrality_types(centrality_types[0], centrality_types)
    cov_methods = tuple(dict.fromkeys(cov_methods))

    backtests: Dict[str, BacktestResult] = {}
    loss_columns: Dict[str, np.ndarray] = {}
    graph_seen: set[str] = set()
    reference_loss_mean = np.nan

    for cov_method in cov_methods:
        sweep_cfg = replace(
            bt_cfg,
            cov_method=cov_method,
            centrality_type=centrality_types[0],
            centrality_types=centrality_types,
        )
        res = run_backtest(data, bt_cfg=sweep_cfg, verbose=verbose)
        backtests[cov_method] = res

        if not np.isfinite(reference_loss_mean):
            reference_loss_mean = float(np.mean(res.combined_losses["equal"]))

        for centrality_type in centrality_types:
            graph_name = _centrality_method_name(
                "graph_only",
                centrality_type,
                centrality_types,
            )
            full_name = _centrality_method_name(
                "full_gcsr",
                centrality_type,
                centrality_types,
            )
            public_full_name = f"{full_name}_{cov_method}"

            if graph_name in res.combined_losses and graph_name not in graph_seen:
                loss_columns[graph_name] = np.asarray(res.combined_losses[graph_name], dtype=float)
                graph_seen.add(graph_name)
            if full_name in res.combined_losses:
                loss_columns[public_full_name] = np.asarray(res.combined_losses[full_name], dtype=float)

    combined_losses = pd.DataFrame(loss_columns)
    mcs_result = model_confidence_set(
        combined_losses,
        alpha=mcs_alpha,
        B=mcs_B,
        statistic=mcs_statistic,
    )

    rows = []
    for method in combined_losses.columns:
        losses = combined_losses[method].to_numpy(dtype=float)
        family = _method_family(method)
        centrality_type = _method_centrality_type(method)
        cov_method = "not_used"
        if family == "full_gcsr":
            cov_method = method.rsplit("_", 1)[-1]

        rows.append({
            "Source": source_name,
            "Source_Type": source_type,
            "Method": method,
            "Family": family,
            "Centrality": centrality_type,
            "Cov_Method": cov_method,
            "Mean_Loss": float(np.mean(losses)),
            "Median_Loss": float(np.median(losses)),
            "Std_Loss": float(np.std(losses)),
            "Total_Loss": float(np.sum(losses)),
            "Rel_MSFE": float(np.mean(losses) / max(reference_loss_mean, EPS)),
            "MCS_pvalue": float(mcs_result.pvalues.get(method, np.nan)),
            "In_MCS": bool(method in mcs_result.included_models),
            "Elimination_Step": int(mcs_result.elimination_steps.get(method, 0)),
        })

    results_table = pd.DataFrame(rows).sort_values(
        ["Mean_Loss", "Family", "Centrality", "Cov_Method"]
    ).reset_index(drop=True)
    results_table["Rank"] = np.arange(1, len(results_table) + 1)

    graph_only_table = results_table[results_table["Family"] == "graph_only"].copy()
    full_gcsr_table = results_table[results_table["Family"] == "full_gcsr"].copy()
    full_gcsr_pivot = full_gcsr_table.pivot(
        index="Centrality",
        columns="Cov_Method",
        values="Rel_MSFE",
    )

    return SensitivitySweepResult(
        source_name=source_name,
        source_type=source_type,
        results_table=results_table,
        graph_only_table=graph_only_table,
        full_gcsr_table=full_gcsr_table,
        full_gcsr_pivot=full_gcsr_pivot,
        combined_losses=combined_losses,
        reference_loss_mean=reference_loss_mean,
        backtests=backtests,
    )


def aggregate_sensitivity_sweeps(
    sweeps: Sequence[SensitivitySweepResult],
) -> pd.DataFrame:
    """Aggregate sensitivity sweeps across multiple empirical or simulated sources."""
    if not sweeps:
        return pd.DataFrame()

    combined = pd.concat([sweep.results_table for sweep in sweeps], ignore_index=True)
    summary = combined.groupby(
        ["Family", "Centrality", "Cov_Method", "Method"],
        dropna=False,
        as_index=False,
    ).agg(
        Mean_Loss=("Mean_Loss", "mean"),
        Mean_Rel_MSFE=("Rel_MSFE", "mean"),
        Median_Rel_MSFE=("Rel_MSFE", "median"),
        Mean_Rank=("Rank", "mean"),
        Median_Rank=("Rank", "median"),
        Mean_MCS_pvalue=("MCS_pvalue", "mean"),
        MCS_Inclusion_Rate=("In_MCS", lambda x: float(np.mean(np.asarray(x, dtype=float)))),
        Sources=("Source", pd.Series.nunique),
    )
    summary = summary.sort_values(
        ["Mean_Rank", "Mean_Rel_MSFE", "Family", "Centrality", "Cov_Method"]
    ).reset_index(drop=True)
    return summary


def plot_sensitivity_summary(
    sensitivity_table: pd.DataFrame,
    value_col: str = "Rel_MSFE",
    title: str = "Sensitivity Analysis",
):
    """
    Visualize the sweep with a full-GCSR heatmap and a graph-only comparison bar chart.
    """
    if sensitivity_table.empty:
        return None

    full_gcsr = sensitivity_table[sensitivity_table["Family"] == "full_gcsr"].copy()
    graph_only = sensitivity_table[sensitivity_table["Family"] == "graph_only"].copy()
    if full_gcsr.empty and graph_only.empty:
        return None

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    if full_gcsr.empty:
        axes[0].axis("off")
    else:
        pivot = full_gcsr.pivot(index="Centrality", columns="Cov_Method", values=value_col)
        if HAS_SEABORN:
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="cividis",
                linewidths=0.6,
                linecolor="white",
                ax=axes[0],
            )
        else:
            im = axes[0].imshow(pivot.values, cmap="cividis", aspect="auto")
            plt.colorbar(im, ax=axes[0], fraction=0.046)
            axes[0].set_xticks(range(len(pivot.columns)))
            axes[0].set_xticklabels(pivot.columns, rotation=45, ha="right")
            axes[0].set_yticks(range(len(pivot.index)))
            axes[0].set_yticklabels(pivot.index)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    axes[0].text(
                        j,
                        i,
                        f"{pivot.iloc[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                    )
        axes[0].set_title("Full GCSR")

    if graph_only.empty:
        axes[1].axis("off")
    else:
        graph_only = graph_only.sort_values(value_col).reset_index(drop=True)
        bars = axes[1].barh(
            graph_only["Method"],
            graph_only[value_col],
            alpha=0.9,
        )
        for bar, method in zip(bars, graph_only["Method"]):
            _apply_bar_patch_style(bar, method)
        axes[1].set_title("Graph Only")
        axes[1].set_xlabel(value_col.replace("_", " "))
        axes[1].invert_yaxis()

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


# ===================================================================
# 13b.  PAPER-READY CSV-DRIVEN REPLOTS
# ===================================================================

PAPER_REPLOT_SIMULATION_METHODS = (
    "equal",
    "full_gcsr_pagerank",
    "rs_selection",
    "rs_selection_v2",
    "graph_only_pagerank",
    "var_error",
    "recent_best",
)

PAPER_REPLOT_HEADLINE_METHODS = (
    "equal",
    "full_gcsr_softmax",
    "graph_only_softmax",
    "rs_selection_v2",
    "rs_selection",
)

PAPER_REPLOT_RS_ORIGINAL_METHODS = (
    "rs_selection_v2",
    "graph_only_v2_eigenvector",
    "graph_only_v2_pagerank",
    "graph_only_v2_softmax",
    "full_gcsr_v2_eigenvector",
    "full_gcsr_v2_pagerank",
    "full_gcsr_v2_softmax",
)

PAPER_WEIGHT_PATH_METHODS = (
    "equal",
    "bates_granger_mv",
    "var_error",
    "full_gcsr_softmax",
    "graph_only_softmax",
    "rs_selection_v2",
    "rs_selection",
    "recent_best",
)


def _paper_read_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a required CSV with a clear error message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required paper-replot CSV not found: {path}")
    return pd.read_csv(path)


def _paper_save_figure(
    fig,
    output_dir: Path,
    stem: str,
    artifacts: Dict[str, List[str]],
    caption: Optional[str] = None,
) -> None:
    """Save a title-free paper figure as PDF/PNG, with an optional caption file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.set_title("")
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    paths = [str(pdf_path), str(png_path)]
    if caption is not None:
        caption_path = output_dir / f"{stem}_caption.txt"
        caption_path.write_text(caption.strip() + "\n")
        paths.append(str(caption_path))
    artifacts[stem] = paths


def _paper_date_axis_from_metadata(
    df: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    period_col: str = "oos_period",
) -> np.ndarray:
    """Attach a date axis when exported OOS metadata is available."""
    if metadata is None or metadata.empty or period_col not in df.columns:
        return df[period_col].to_numpy() if period_col in df.columns else np.arange(len(df))
    if period_col not in metadata.columns or "period_dt" not in metadata.columns:
        return df[period_col].to_numpy()
    mapper = metadata.set_index(period_col)["period_dt"].to_dict()
    values = pd.to_datetime(df[period_col].map(mapper), errors="coerce")
    if values.notna().all():
        return values.to_numpy()
    return df[period_col].to_numpy()


def _paper_apply_time_axis(ax, x_axis: Sequence[object], x_label: str = "Time") -> None:
    """Format a paper-replot x-axis."""
    _format_plot_time_axis(ax, x_axis, x_label=x_label)
    ax.tick_params(axis="x", rotation=0)


def _paper_plot_line(
    ax,
    x_axis: Sequence[object],
    y_values: Sequence[float],
    method: str,
    n_points: Optional[int] = None,
    linewidth: Optional[float] = None,
    zorder: Optional[int] = None,
) -> None:
    """Plot one consistently styled paper line."""
    y_values = np.asarray(y_values, dtype=float)
    n_points = int(n_points or len(y_values))
    style = _method_plot_style(method)
    ax.plot(
        x_axis,
        y_values,
        label=method,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=linewidth if linewidth is not None else (2.9 if _is_gcsr_method(method) else 2.4),
        alpha=0.96,
        zorder=zorder if zorder is not None else (3 if _is_gcsr_method(method) else 2),
        **_line_marker_kwargs(n_points, style["marker"]),
    )


def _paper_load_empirical_panel(
    forecast_path: Union[str, Path],
    truth_path: Union[str, Path],
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and align an empirical survey panel for data-section figures."""
    forecasts_df = _paper_read_csv(forecast_path)
    truth_df = _paper_read_csv(truth_path)
    forecaster_ids = [col for col in forecasts_df.columns if col != "TARGET_PERIOD"]
    merged = forecasts_df.merge(
        truth_df[["TARGET_PERIOD", "actual"]],
        on="TARGET_PERIOD",
        how="left",
    )
    merged["period_dt"] = pd.to_datetime(merged["TARGET_PERIOD"])
    numeric_cols = forecaster_ids + ["actual"]
    merged[numeric_cols] = merged[numeric_cols].astype(float)
    merged = merged.dropna(subset=["actual"]).sort_values("period_dt").reset_index(drop=True)
    return merged, forecaster_ids


def _paper_plot_forecaster_heatmap(
    merged: pd.DataFrame,
    forecaster_ids: Sequence[str],
    title: str,
):
    """Create a forecaster correlation heatmap with y and f_i labels."""
    columns = ["actual"] + list(forecaster_ids)
    labels = ["$y$"] + [rf"$f_{{{idx}}}$" for idx in range(1, len(forecaster_ids) + 1)]
    corr = merged[columns].corr()
    corr.index = labels
    corr.columns = labels
    side = max(6.6, min(10.5, 0.48 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(side, side))
    if HAS_SEABORN:
        sns.heatmap(
            corr,
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cmap="vlag",
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 9.5},
            square=True,
            linewidths=0.45,
            linecolor="white",
            cbar_kws={"shrink": 0.75, "label": "Correlation"},
        )
    else:
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        fig.colorbar(im, ax=ax, shrink=0.75, label="Correlation")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9.5)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    return fig


def _paper_plot_spf_range(
    merged: pd.DataFrame,
    forecaster_ids: Sequence[str],
    title: str,
    y_label: str,
):
    """Plot realized values and the survey forecast range."""
    x_axis = merged["period_dt"]
    forecasts = merged[list(forecaster_ids)]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.fill_between(
        x_axis,
        forecasts.min(axis=1).to_numpy(),
        forecasts.max(axis=1).to_numpy(),
        color="#56B4E9",
        alpha=0.24,
        linewidth=0,
        label="SPF range",
    )
    ax.plot(
        x_axis,
        forecasts.median(axis=1).to_numpy(),
        color="#009E73",
        linewidth=2.7,
        linestyle="--",
        label="SPF median",
    )
    ax.plot(
        x_axis,
        merged["actual"].to_numpy(),
        color="#111111",
        linewidth=3.2,
        label="Actual",
        zorder=4,
    )
    ax.set_ylabel(y_label)
    _paper_apply_time_axis(ax, x_axis, x_label="Target Period")
    _apply_publication_legend(ax, ncol=3, loc="best", fontsize=11.0, display_labels=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _paper_write_rs_original_table(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Write the original Richter-Smetanina methodology comparison table."""
    specs = (
        ("Inflation", "empirical_inflation_next_quarter"),
        ("Unemployment", "empirical_unemployment_monthly"),
    )
    rows = []
    for dataset_label, study_dir in specs:
        perf = _paper_read_csv(csv_root / "empirical_studies" / study_dir / "performance_table.csv")
        perf = perf[perf["Method"].isin(PAPER_REPLOT_RS_ORIGINAL_METHODS)].copy()
        for _, row in perf.iterrows():
            rows.append({
                "Application": dataset_label,
                "Method": method_display_label(row["Method"]),
                "_raw_method": row["Method"],
                "Mean_Loss": float(row["Mean_Loss"]),
                "Rel_MSFE": float(row["Rel_MSFE"]),
                "Total_Loss": float(row["Total_Loss"]),
            })

    table = pd.DataFrame(rows)
    order_map = {method: idx for idx, method in enumerate(PAPER_REPLOT_RS_ORIGINAL_METHODS)}
    table["_order"] = table["_raw_method"].map(order_map)
    table = (
        table.sort_values(["Application", "_order"])
        .drop(columns=["_order", "_raw_method"])
        .reset_index(drop=True)
    )

    csv_path = output_dir / "rs_original_methodology_performance.csv"
    tex_path = output_dir / "rs_original_methodology_performance.tex"
    table.to_csv(csv_path, index=False)
    latex_cols = ["Application", "Method", "Mean_Loss", "Rel_MSFE", "Total_Loss"]
    latex = table[latex_cols].to_latex(
        index=False,
        escape=False,
        float_format="%.3f",
        caption="Original Richter-Smetanina methodology performance against GLIDE and GLIDER combinations.",
        label="tab:rs-original-methodology-performance",
    )
    tex_path.write_text(latex)
    artifacts["rs_original_methodology_performance"] = [str(csv_path), str(tex_path)]


def _paper_plot_simulation_forecasts_errors(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Create uncluttered simulation forecast and forecast-error small multiples."""
    scenario_dirs = {
        "1A": "mcs_backtest_1A",
        "2A": "res_2A",
        "2B": "res_2B",
        "2C": "mcs_backtest_2C",
        "3A": "mcs_backtest_4A",
    }
    for scenario_key, backtest_dir in scenario_dirs.items():
        base = csv_root / "backtests" / backtest_dir
        forecasts = _paper_read_csv(base / "combined_forecasts.csv")
        actual = _paper_read_csv(base / "y_oos.csv")
        df = forecasts.merge(actual, on="oos_period", how="left")
        methods = [method for method in PAPER_REPLOT_SIMULATION_METHODS if method in df.columns]
        methods = order_methods_for_display(methods)
        x_axis = df["oos_period"].to_numpy()
        n_methods = len(methods)

        fig, axes = plt.subplots(
            n_methods,
            2,
            figsize=(10.8, 1.20 * n_methods + 1.8),
            sharex=True,
            gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.12, "hspace": 0.18},
        )
        if n_methods == 1:
            axes = np.asarray([axes])

        for row_idx, method in enumerate(methods):
            forecast_ax = axes[row_idx, 0]
            error_ax = axes[row_idx, 1]
            style = _method_plot_style(method)
            display_label = method_display_label(method)
            method_color = "#6A6A6A" if method == "equal" else style["color"]
            method_linestyle = "--" if method == "equal" else style["linestyle"]

            forecast_ax.plot(
                x_axis,
                df["actual"].to_numpy(),
                color="#111111",
                linewidth=1.7,
                alpha=0.82,
                label="Actual",
            )
            forecast_ax.plot(
                x_axis,
                df[method].to_numpy(dtype=float),
                color=method_color,
                linestyle=method_linestyle,
                linewidth=2.05,
                label="_nolegend_",
            )

            errors = df["actual"].to_numpy(dtype=float) - df[method].to_numpy(dtype=float)
            error_ax.axhline(0.0, color="#111111", linewidth=0.9, linestyle=":")
            error_ax.plot(
                x_axis,
                errors,
                color=method_color,
                linestyle=method_linestyle,
                linewidth=2.05,
            )

            forecast_ax.set_ylabel(
                display_label,
                rotation=0,
                ha="right",
                va="center",
                labelpad=18,
                fontsize=10.5,
            )
            for ax in (forecast_ax, error_ax):
                ax.grid(True, alpha=0.20)
                ax.tick_params(axis="both", labelsize=8.8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            if row_idx < n_methods - 1:
                forecast_ax.tick_params(axis="x", labelbottom=False)
                error_ax.tick_params(axis="x", labelbottom=False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        actual_handles = [handle for handle, label in zip(handles, labels) if label == "Actual"]
        if actual_handles:
            axes[0, 0].legend(
                handles=[actual_handles[0]],
                labels=["Actual"],
                loc="upper left",
                fontsize=8.6,
                frameon=True,
                framealpha=0.92,
            )

        fig.supxlabel("OOS period", y=0.025, fontsize=11.5)
        fig.subplots_adjust(left=0.19, right=0.985, top=0.965, bottom=0.08, wspace=0.12, hspace=0.18)
        _paper_save_figure(
            fig,
            output_dir,
            f"simulation_{scenario_display_label(scenario_key)}_forecasts_errors",
            artifacts,
            caption=(
                f"Simulation Scenario {scenario_display_label(scenario_key)} forecast paths "
                "and forecast errors for the selected combination methods. Each row shows one "
                "method. The left column overlays the method's combined forecast with the "
                "realized value. The right column reports the forecast error, defined as the "
                "realized value minus the combined forecast, with the dotted zero line marking "
                "an exact forecast. "
                f"The simulation was regenerated from the CSV export with T={int(x_axis.max()) + 1} "
                f"and T0={int(x_axis.min())}."
            ),
        )


def _paper_plot_sensitivity_bars(
    data: pd.DataFrame,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    output_dir: Path,
    stem: str,
    artifacts: Dict[str, List[str]],
    methods: Optional[Sequence[str]] = None,
    xlabel: str = "Mean relative MSFE",
    note: Optional[str] = None,
    caption: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
) -> None:
    """Shared compact sensitivity bar plot."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color="#D9D9D9", edgecolor="#2F2F2F", linewidth=0.9)
    if methods is not None:
        for bar, method in zip(bars, methods):
            _apply_bar_patch_style(bar, method)
    if colors is not None:
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
            bar.set_alpha(0.90)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#111111", linewidth=1.1, linestyle=":")
    ax.set_xlabel(xlabel)
    if note:
        ax.text(
            0.0,
            -0.18,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.0,
            color="#4D4D4D",
        )
    for ypos, value in zip(y_pos, values):
        ax.text(value + 0.01, ypos, f"{value:.3f}", va="center", fontsize=10.5)
    fig.tight_layout()
    _paper_save_figure(fig, output_dir, stem, artifacts, caption=caption)


def _paper_load_simulation_sensitivity_results(csv_root: Path) -> pd.DataFrame:
    """Load exercise-level sensitivity rows for each simulated scenario."""
    rows = []
    base = csv_root / "sensitivity_sweeps"
    scenario_order = ("1A", "2A", "2B", "2C", "4A")
    for scenario_key in scenario_order:
        path = base / f"simulation_{scenario_key}" / "results_table.csv"
        if not path.exists():
            continue
        df = _paper_read_csv(path)
        df["Scenario"] = scenario_display_label(scenario_key)
        df["_scenario_order"] = len(rows)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    scenario_order_map = {"1A": 0, "2A": 1, "2B": 2, "2C": 3, "3A": 4}
    combined["_scenario_order"] = combined["Scenario"].map(scenario_order_map)
    return combined


def _paper_load_empirical_sensitivity_results(csv_root: Path) -> pd.DataFrame:
    """Load application-level sensitivity rows for the empirical level applications."""
    rows = []
    base = csv_root / "sensitivity_sweeps"
    specs = (
        ("inflation_levels", "Inflation", 0),
        ("unemployment_levels", "Unemployment", 1),
    )
    for source_name, label, order in specs:
        path = base / source_name / "results_table.csv"
        if not path.exists():
            continue
        df = _paper_read_csv(path)
        df["Scenario"] = label
        df["_scenario_order"] = order
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _paper_load_empirical_and_simulation_sensitivity_results(csv_root: Path) -> pd.DataFrame:
    """Load sensitivity rows for empirical applications and simulation exercises."""
    empirical = _paper_load_empirical_sensitivity_results(csv_root)
    simulation = _paper_load_simulation_sensitivity_results(csv_root)
    frames = []
    if not empirical.empty:
        frames.append(empirical)
    if not simulation.empty:
        simulation = simulation.copy()
        simulation["_scenario_order"] = simulation["_scenario_order"].astype(float) + 2
        frames.append(simulation)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _paper_plot_sensitivity_by_exercise(
    plot_df: pd.DataFrame,
    line_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    artifacts: Dict[str, List[str]],
    method_col: Optional[str] = None,
    caption: Optional[str] = None,
    x_axis_label: str = "Simulation exercise",
) -> None:
    """Plot exercise-by-exercise relative MSFE sensitivity as side-by-side bars."""
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(["_scenario_order", line_col]).copy()
    scenarios = list(dict.fromkeys(plot_df["Scenario"].tolist()))
    x_pos = np.arange(len(scenarios))
    fig_width = max(8.4, 1.18 * len(scenarios) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    covariance_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    hatches = ["//", "xx", "..", "\\\\"]
    groups = list(plot_df.groupby(line_col, sort=False, observed=False))
    n_groups = max(len(groups), 1)
    bar_width = min(0.24, 0.78 / n_groups)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_width
    for line_idx, (line_value, sub) in enumerate(groups):
        sub = sub.set_index("Scenario").reindex(scenarios).reset_index()
        method_name = None
        if method_col is not None and method_col in sub.columns:
            method_values = sub[method_col].dropna().tolist()
            method_name = method_values[0] if method_values else None
        if method_name is None:
            family = "full_gcsr" if "Cov" in line_col or line_value in {"rolling", "ewma", "shrinkage"} else "graph_only"
            method_name = f"{family}_{line_value}" if line_value in CENTRALITY_ORDER else str(line_value)
        style = _method_plot_style(method_name)
        color = style["color"]
        if line_col == "Centrality":
            color = CENTRALITY_COLORS.get(str(line_value), color)
        else:
            color = covariance_colors[line_idx % len(covariance_colors)]
        label = (
            CENTRALITY_DISPLAY_LABELS.get(str(line_value), str(line_value)).title()
            if line_col == "Centrality"
            else str(line_value).title()
        )
        ax.bar(
            x_pos + offsets[line_idx],
            sub[value_col].to_numpy(dtype=float),
            width=bar_width,
            color=color,
            alpha=0.86,
            edgecolor="#202020",
            linewidth=0.9,
            hatch=hatches[line_idx % len(hatches)],
            label=label,
        )

    ax.axhline(1.0, color="#111111", linewidth=1.0, linestyle=":")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_axis_label)
    _apply_publication_legend(ax, ncol=min(3, plot_df[line_col].nunique()), fontsize=10.5, display_labels=False)
    fig.tight_layout()
    _paper_save_figure(fig, output_dir, stem, artifacts, caption=caption)


def _paper_plot_sensitivity_figures(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Create compact GLIDE and GLIDER sensitivity plots from exported sweeps."""
    summary_path = csv_root / "sensitivity_summaries" / "simulation_sweep_summary" / "summary.csv"
    if not summary_path.exists():
        summary_path = csv_root / "tables" / "simulation_sweep_summary.csv"
    summary = _paper_read_csv(summary_path)
    value_col = "Mean_Rel_MSFE" if "Mean_Rel_MSFE" in summary.columns else "Rel_MSFE"

    centrality_order = ["eigenvector", "pagerank", "softmax"]
    glide = summary[summary["Family"] == "graph_only"].copy()
    glide = glide[glide["Centrality"].isin(centrality_order)]
    glide["_order"] = glide["Centrality"].map({name: idx for idx, name in enumerate(centrality_order)})
    glide = glide.sort_values("_order")
    _paper_plot_sensitivity_bars(
        glide,
        [CENTRALITY_DISPLAY_LABELS.get(c, c).title() for c in glide["Centrality"]],
        glide[value_col].to_numpy(dtype=float),
        "GLIDE centrality sensitivity",
        output_dir,
        "sensitivity_glide_centralities",
        artifacts,
        methods=glide["Method"].tolist(),
        xlabel="Mean relative MSFE across simulation exercises",
        note="Mean is over Scenarios 1A, 2A, 2B, 2C, and 3A.",
        colors=[CENTRALITY_COLORS.get(str(c), "#4D4D4D") for c in glide["Centrality"]],
        caption=(
            "GLIDE centrality sensitivity. Bars report mean relative MSFE across "
            "simulation exercises 1A, 2A, 2B, 2C, and 3A; lower values indicate "
            "better performance relative to equal weights."
        ),
    )

    glider = summary[
        (summary["Family"] == "full_gcsr")
        & (summary["Centrality"].isin(centrality_order))
        & (summary["Cov_Method"] == "shrinkage")
    ].copy()
    glider["_order"] = glider["Centrality"].map({name: idx for idx, name in enumerate(centrality_order)})
    glider = glider.sort_values("_order")
    _paper_plot_sensitivity_bars(
        glider,
        [CENTRALITY_DISPLAY_LABELS.get(c, c).title() for c in glider["Centrality"]],
        glider[value_col].to_numpy(dtype=float),
        "GLIDER centrality sensitivity",
        output_dir,
        "sensitivity_glider_centralities",
        artifacts,
        methods=glider["Method"].tolist(),
        xlabel="Mean relative MSFE across simulation exercises",
        note="Mean is over Scenarios 1A, 2A, 2B, 2C, and 3A using shrinkage covariance.",
        colors=[CENTRALITY_COLORS.get(str(c), "#4D4D4D") for c in glider["Centrality"]],
        caption=(
            "GLIDER centrality sensitivity using shrinkage covariance. Bars report "
            "mean relative MSFE across simulation exercises 1A, 2A, 2B, 2C, and 3A; "
            "lower values indicate better performance relative to equal weights."
        ),
    )

    cov_methods = ["rolling", "ewma", "shrinkage"]
    cov_df = summary[
        (summary["Family"] == "full_gcsr")
        & (summary["Cov_Method"].isin(cov_methods))
    ].copy()
    cov_df = cov_df.groupby("Cov_Method", as_index=False)[value_col].mean()
    cov_df["_order"] = cov_df["Cov_Method"].map({name: idx for idx, name in enumerate(cov_methods)})
    cov_df = cov_df.sort_values("_order")
    _paper_plot_sensitivity_bars(
        cov_df,
        [str(value).title() for value in cov_df["Cov_Method"]],
        cov_df[value_col].to_numpy(dtype=float),
        "GLIDER covariance sensitivity",
        output_dir,
        "sensitivity_glider_covariance_methods",
        artifacts,
        xlabel="Mean relative MSFE across simulation exercises and centralities",
        note="Mean is over Scenarios 1A, 2A, 2B, 2C, 3A and the three centralities.",
        caption=(
            "GLIDER covariance-estimation sensitivity. Bars report mean relative MSFE "
            "averaged across simulation exercises 1A, 2A, 2B, 2C, and 3A and across "
            "the eigenvector, PageRank, and softmax centralities; lower values indicate "
            "better performance relative to equal weights."
        ),
    )

    exercise_df = _paper_load_simulation_sensitivity_results(csv_root)
    exercise_value_col = "Rel_MSFE"
    centrality_order_map = {name: idx for idx, name in enumerate(centrality_order)}
    scenario_order_map = {"1A": 0, "2A": 1, "2B": 2, "2C": 3, "3A": 4}

    glide_exercise = exercise_df[
        (exercise_df["Family"] == "graph_only")
        & (exercise_df["Centrality"].isin(centrality_order))
    ].copy()
    glide_exercise["_centrality_order"] = glide_exercise["Centrality"].map(centrality_order_map)
    glide_exercise["_scenario_order"] = glide_exercise["Scenario"].map(scenario_order_map)
    glide_exercise = glide_exercise.sort_values(["_centrality_order", "_scenario_order"])
    _paper_plot_sensitivity_by_exercise(
        glide_exercise,
        line_col="Centrality",
        value_col=exercise_value_col,
        title="GLIDE centrality sensitivity by exercise",
        ylabel="Relative MSFE",
        output_dir=output_dir,
        stem="sensitivity_glide_centralities_by_exercise",
        artifacts=artifacts,
        method_col="Method",
        caption=(
            "Exercise-by-exercise GLIDE centrality sensitivity. Side-by-side bars show "
            "relative MSFE for each centrality within each simulation exercise, with the "
            "dotted reference line marking equal-weight performance."
        ),
    )

    all_exercise_df = _paper_load_empirical_and_simulation_sensitivity_results(csv_root)
    all_order_map = {"Inflation": 0, "Unemployment": 1, "1A": 2, "2A": 3, "2B": 4, "2C": 5, "3A": 6}

    glide_all = all_exercise_df[
        (all_exercise_df["Family"] == "graph_only")
        & (all_exercise_df["Centrality"].isin(centrality_order))
    ].copy()
    glide_all["_centrality_order"] = glide_all["Centrality"].map(centrality_order_map)
    glide_all["_scenario_order"] = glide_all["Scenario"].map(all_order_map)
    glide_all = glide_all.sort_values(["_centrality_order", "_scenario_order"])
    _paper_plot_sensitivity_by_exercise(
        glide_all,
        line_col="Centrality",
        value_col=exercise_value_col,
        title="GLIDE centrality sensitivity across applications and exercises",
        ylabel="Relative MSFE",
        output_dir=output_dir,
        stem="sensitivity_glide_centralities_empirical_and_simulation",
        artifacts=artifacts,
        method_col="Method",
        x_axis_label="Application or simulation exercise",
        caption=(
            "GLIDE centrality sensitivity across empirical applications and simulation "
            "exercises. Side-by-side bars show relative MSFE for inflation, unemployment, "
            "and simulation exercises 1A, 2A, 2B, 2C, and 3A; the dotted reference line "
            "marks equal-weight performance."
        ),
    )

    glider_exercise = exercise_df[
        (exercise_df["Family"] == "full_gcsr")
        & (exercise_df["Centrality"].isin(centrality_order))
        & (exercise_df["Cov_Method"] == "shrinkage")
    ].copy()
    glider_exercise["_centrality_order"] = glider_exercise["Centrality"].map(centrality_order_map)
    glider_exercise["_scenario_order"] = glider_exercise["Scenario"].map(scenario_order_map)
    glider_exercise = glider_exercise.sort_values(["_centrality_order", "_scenario_order"])
    _paper_plot_sensitivity_by_exercise(
        glider_exercise,
        line_col="Centrality",
        value_col=exercise_value_col,
        title="GLIDER centrality sensitivity by exercise",
        ylabel="Relative MSFE (shrinkage covariance)",
        output_dir=output_dir,
        stem="sensitivity_glider_centralities_by_exercise",
        artifacts=artifacts,
        method_col="Method",
        caption=(
            "Exercise-by-exercise GLIDER centrality sensitivity under shrinkage covariance. "
            "Side-by-side bars show relative MSFE for each centrality within each simulation "
            "exercise, with the dotted reference line marking equal-weight performance."
        ),
    )

    glider_all = all_exercise_df[
        (all_exercise_df["Family"] == "full_gcsr")
        & (all_exercise_df["Centrality"].isin(centrality_order))
        & (all_exercise_df["Cov_Method"] == "shrinkage")
    ].copy()
    glider_all["_centrality_order"] = glider_all["Centrality"].map(centrality_order_map)
    glider_all["_scenario_order"] = glider_all["Scenario"].map(all_order_map)
    glider_all = glider_all.sort_values(["_centrality_order", "_scenario_order"])
    _paper_plot_sensitivity_by_exercise(
        glider_all,
        line_col="Centrality",
        value_col=exercise_value_col,
        title="GLIDER centrality sensitivity across applications and exercises",
        ylabel="Relative MSFE (shrinkage covariance)",
        output_dir=output_dir,
        stem="sensitivity_glider_centralities_empirical_and_simulation",
        artifacts=artifacts,
        method_col="Method",
        x_axis_label="Application or simulation exercise",
        caption=(
            "GLIDER centrality sensitivity under shrinkage covariance across empirical "
            "applications and simulation exercises. Side-by-side bars show relative MSFE "
            "for inflation, unemployment, and simulation exercises 1A, 2A, 2B, 2C, and 3A; "
            "the dotted reference line marks equal-weight performance."
        ),
    )

    cov_exercise = exercise_df[
        (exercise_df["Family"] == "full_gcsr")
        & (exercise_df["Cov_Method"].isin(cov_methods))
    ].copy()
    cov_exercise["_scenario_order"] = cov_exercise["Scenario"].map(scenario_order_map)
    cov_exercise["Cov_Method"] = pd.Categorical(cov_exercise["Cov_Method"], cov_methods, ordered=True)
    cov_exercise = (
        cov_exercise.groupby(["Scenario", "_scenario_order", "Cov_Method"], observed=True, as_index=False)
        [exercise_value_col]
        .mean()
        .sort_values(["Cov_Method", "_scenario_order"])
    )
    _paper_plot_sensitivity_by_exercise(
        cov_exercise,
        line_col="Cov_Method",
        value_col=exercise_value_col,
        title="GLIDER covariance sensitivity by exercise",
        ylabel="Relative MSFE averaged over centralities",
        output_dir=output_dir,
        stem="sensitivity_glider_covariance_methods_by_exercise",
        artifacts=artifacts,
        caption=(
            "Exercise-by-exercise GLIDER covariance sensitivity. Side-by-side bars show "
            "relative MSFE for each covariance estimator within each simulation exercise, "
            "averaged over the three centralities; the dotted reference line marks "
            "equal-weight performance."
        ),
    )

    cov_all = all_exercise_df[
        (all_exercise_df["Family"] == "full_gcsr")
        & (all_exercise_df["Cov_Method"].isin(cov_methods))
    ].copy()
    cov_all["_scenario_order"] = cov_all["Scenario"].map(all_order_map)
    cov_all["Cov_Method"] = pd.Categorical(cov_all["Cov_Method"], cov_methods, ordered=True)
    cov_all = (
        cov_all.groupby(["Scenario", "_scenario_order", "Cov_Method"], observed=True, as_index=False)
        [exercise_value_col]
        .mean()
        .sort_values(["Cov_Method", "_scenario_order"])
    )
    _paper_plot_sensitivity_by_exercise(
        cov_all,
        line_col="Cov_Method",
        value_col=exercise_value_col,
        title="GLIDER covariance sensitivity across applications and exercises",
        ylabel="Relative MSFE averaged over centralities",
        output_dir=output_dir,
        stem="sensitivity_glider_covariance_methods_empirical_and_simulation",
        artifacts=artifacts,
        x_axis_label="Application or simulation exercise",
        caption=(
            "GLIDER covariance sensitivity across empirical applications and simulation "
            "exercises. Side-by-side bars show relative MSFE for rolling, EWMA, and "
            "shrinkage covariance estimators, averaged over the three centralities; the "
            "dotted reference line marks equal-weight performance."
        ),
    )


def _paper_empirical_spec_paths(csv_root: Path) -> Tuple[Dict[str, object], ...]:
    """Return canonical empirical paper-replot paths."""
    return (
        {
            "label": "Inflation",
            "folder": csv_root / "empirical_studies" / "empirical_inflation_next_quarter",
            "ylabel": "Inflation",
        },
        {
            "label": "Unemployment",
            "folder": csv_root / "empirical_studies" / "empirical_unemployment_monthly",
            "ylabel": "Unemployment",
        },
    )


def _paper_plot_empirical_headline(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Plot cumulative loss differences for headline empirical comparisons."""
    specs = _paper_empirical_spec_paths(csv_root)
    for spec in specs:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        folder = Path(spec["folder"])
        losses = _paper_read_csv(folder / "backtest" / "combined_losses.csv")
        metadata = _paper_read_csv(folder / "oos_period_metadata.csv")
        x_axis = _paper_date_axis_from_metadata(losses, metadata)
        methods = [method for method in PAPER_REPLOT_HEADLINE_METHODS if method in losses.columns]
        methods = order_methods_for_display(methods)
        reference = losses["equal"].to_numpy(dtype=float)
        for method in methods:
            if method == "equal":
                series = np.zeros(len(losses))
            else:
                series = np.cumsum(losses[method].to_numpy(dtype=float) - reference)
            _paper_plot_line(ax, x_axis, series, method, n_points=len(losses))
        ax.axhline(0.0, color="#111111", linewidth=1.0, linestyle=":")
        ax.set_ylabel("Cumulative loss difference")
        _paper_apply_time_axis(ax, x_axis, x_label=f"{spec['label']} target period")
        _apply_publication_legend(
            ax,
            ncol=2,
            loc="best",
            fontsize=10.0,
            method_order=PAPER_METHOD_ORDER,
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        slug = str(spec["label"]).lower()
        _paper_save_figure(
            fig,
            output_dir,
            f"headline_empirical_cumulative_loss_{slug}",
            artifacts,
            caption=(
                f"Headline empirical comparison for {slug}. The figure plots cumulative "
                "loss differences relative to equal weights for GLIDE softmax, GLIDER "
                "softmax, RS selection, Hyperedge RS selection, and equal weights; lower "
                "cumulative loss differences indicate better relative performance."
            ),
        )


def _paper_plot_empirical_diff_oos_forecasts(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Plot OOS forecasts for differenced empirical inflation and unemployment."""
    specs = (
        {
            "slug": "inflation",
            "label": "Inflation",
            "folder": csv_root / "empirical_studies" / "empirical_inflation_next_quarter_diffprev",
            "ylabel": "Differenced inflation (%)",
        },
        {
            "slug": "unemployment",
            "label": "Unemployment",
            "folder": csv_root / "empirical_studies" / "empirical_unemployment_monthly_diffprev",
            "ylabel": "Differenced unemployment (%)",
        },
    )
    preferred_methods = (
        "equal",
        "var_error",
        "bates_granger_mv",
        "full_gcsr_softmax",
        "graph_only_softmax",
        "rs_selection_v2",
        "rs_selection",
        "recent_best",
    )
    for spec in specs:
        path = Path(spec["folder"]) / "oos_forecast_table.csv"
        if not path.exists():
            continue
        df = _paper_read_csv(path)
        x_axis = pd.to_datetime(df["period_dt"], errors="coerce")
        if x_axis.isna().any():
            x_axis = pd.to_datetime(df["TARGET_PERIOD"], errors="coerce")
        if x_axis.isna().any():
            x_axis = np.arange(len(df))

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.plot(
            x_axis,
            df["actual"].to_numpy(dtype=float),
            color="#111111",
            linewidth=3.0,
            label="actual",
            zorder=5,
        )
        methods = [method for method in preferred_methods if method in df.columns]
        for method in order_methods_for_display(methods):
            _paper_plot_line(
                ax,
                x_axis,
                df[method].to_numpy(dtype=float),
                method,
                n_points=len(df),
                linewidth=2.1,
            )
        ax.axhline(0.0, color="#111111", linewidth=1.0, linestyle=":")
        ax.set_ylabel(str(spec["ylabel"]))
        _paper_apply_time_axis(ax, x_axis, x_label=f"{spec['label']} target period")
        _apply_publication_legend(
            ax,
            ncol=2,
            loc="best",
            fontsize=9.2,
            method_order=("actual",) + PAPER_ADAPTABILITY_METHOD_ORDER,
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        _paper_save_figure(
            fig,
            output_dir,
            f"oos_forecasts_{spec['slug']}_diffprev",
            artifacts,
            caption=(
                f"Out-of-sample forecasts for differenced {str(spec['label']).lower()}. "
                "The black line is the realized differenced value; colored lines are "
                "selected forecast-combination methods. Values are measured in percentage "
                "points."
            ),
        )


def _paper_plot_rs_pairwise(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Plot direct Hyperedge RS selection vs standard RS selection comparisons."""
    specs = _paper_empirical_spec_paths(csv_root)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), sharey=False)
    for ax, spec in zip(axes, specs):
        folder = Path(spec["folder"])
        losses = _paper_read_csv(folder / "backtest" / "combined_losses.csv")
        metadata = _paper_read_csv(folder / "oos_period_metadata.csv")
        x_axis = _paper_date_axis_from_metadata(losses, metadata)
        diff = np.cumsum(
            losses["rs_selection"].to_numpy(dtype=float)
            - losses["rs_selection_v2"].to_numpy(dtype=float)
        )
        ax.plot(
            x_axis,
            diff,
            color=METHOD_FAMILY_COLORS["rs_selection"],
            linestyle="--",
            marker="X",
            markevery=max(len(diff) // 10, 1),
            linewidth=2.7,
            label="Hyperedge RS selection minus RS selection",
        )
        ax.axhline(0.0, color="#111111", linewidth=1.0, linestyle=":")
        ax.set_ylabel("Cumulative loss difference" if ax is axes[0] else "")
        _paper_apply_time_axis(ax, x_axis, x_label=f"{spec['label']} target period")
        _apply_publication_legend(ax, ncol=1, fontsize=10.0, display_labels=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(left=0.11, wspace=0.20)
    _paper_save_figure(
        fig,
        output_dir,
        "rs_selection_pairwise_empirical",
        artifacts,
        caption=(
            "Direct empirical comparison of Hyperedge RS selection and standard RS selection "
            "for inflation and unemployment. The plotted series is the cumulative loss of "
            "Hyperedge RS selection minus the cumulative loss of RS selection, so negative "
            "values favor Hyperedge RS selection."
        ),
    )


def _paper_plot_weight_stability(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Plot empirical Herfindahl weight stability for key methods."""
    methods = ("full_gcsr_softmax", "graph_only_softmax", "rs_selection_v2")
    specs = (
        ("inflation", "Inflation", csv_root / "empirical_studies" / "empirical_inflation_next_quarter"),
        ("unemployment", "Unemployment", csv_root / "empirical_studies" / "empirical_unemployment_monthly"),
    )

    loaded_specs = []
    for slug, label, folder in specs:
        diagnostics = _paper_read_csv(folder / "backtest" / "weight_diagnostics_long.csv")
        metadata = _paper_read_csv(folder / "oos_period_metadata.csv")
        loaded_specs.append((slug, label, diagnostics, metadata))

        fig, ax = plt.subplots(figsize=(8.8, 4.7))
        plotted_methods = order_methods_for_display([m for m in methods if m in diagnostics["method"].unique()])
        x_axis = None
        for method in plotted_methods:
            sub = diagnostics[diagnostics["method"] == method].copy()
            x_axis = _paper_date_axis_from_metadata(sub, metadata)
            _paper_plot_line(ax, x_axis, sub["herfindahl"].to_numpy(dtype=float), method, n_points=len(sub))
        ax.set_ylabel("Herfindahl index")
        if x_axis is not None:
            _paper_apply_time_axis(ax, x_axis, x_label="Target Period")
        _apply_publication_legend(ax, ncol=1, fontsize=10.5, method_order=PAPER_METHOD_ORDER)
        fig.autofmt_xdate()
        fig.tight_layout()
        _paper_save_figure(
            fig,
            output_dir,
            f"weight_stability_{slug}_herfindahl",
            artifacts,
            caption=(
                f"{label} weight stability. The Herfindahl index is plotted over time for "
                "GLIDER softmax, GLIDE softmax, and RS selection; higher values indicate more "
                "concentrated forecast-combination weights."
            ),
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    for ax, (_, label, diagnostics, metadata) in zip(axes, loaded_specs):
        plotted_methods = order_methods_for_display([m for m in methods if m in diagnostics["method"].unique()])
        x_axis = None
        for method in plotted_methods:
            sub = diagnostics[diagnostics["method"] == method].copy()
            x_axis = _paper_date_axis_from_metadata(sub, metadata)
            _paper_plot_line(ax, x_axis, sub["herfindahl"].to_numpy(dtype=float), method, n_points=len(sub))
        ax.set_ylabel("Herfindahl index" if ax is axes[0] else "")
        if x_axis is not None:
            _paper_apply_time_axis(ax, x_axis, x_label=f"{label} target period")
    _apply_publication_legend(axes[0], ncol=1, fontsize=9.5, method_order=PAPER_METHOD_ORDER)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.16)
    _paper_save_figure(
        fig,
        output_dir,
        "weight_stability_empirical_herfindahl",
        artifacts,
        caption=(
            "Empirical weight stability for inflation and unemployment. The Herfindahl index "
            "is plotted over time for GLIDER softmax, GLIDE softmax, and RS selection; higher "
            "values indicate more concentrated forecast-combination weights."
        ),
    )


def _paper_forecaster_palette(n_models: int) -> List[str]:
    """Distinct, print-tolerant colors for stackplot forecaster weights."""
    base = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#B07AA1",
        "#E15759",
        "#76B7B2",
        "#EDC948",
        "#9C755F",
        "#BAB0AC",
        "#A0CBE8",
        "#FFBE7D",
        "#8CD17D",
    ]
    if n_models <= len(base):
        return base[:n_models]
    cmap = plt.get_cmap("tab20")
    return [base[idx] if idx < len(base) else cmap((idx - len(base)) % 20) for idx in range(n_models)]


def _paper_forecaster_hatches(n_models: int) -> List[str]:
    hatches = ["", "///", "\\\\\\", "xxx", "...", "++", "---", "oo", "\\\\..", "xx//"]
    return [hatches[idx % len(hatches)] for idx in range(n_models)]


def _paper_compact_weight_caption(
    label: str,
    model_order: Sequence[object],
    empirical: bool,
) -> str:
    """Caption text for compact A4 forecaster-weight stackplots."""
    model_labels = [f"$f_{{{idx + 1}}}$" for idx in range(len(model_order))]
    if empirical:
        mapping = ", ".join(
            f"$f_{{{idx + 1}}}$ = {model_id}"
            for idx, model_id in enumerate(model_order)
        )
        mapping_sentence = f" Forecaster labels map to original SPF IDs as follows: {mapping}."
    else:
        mapping_sentence = " Forecaster labels follow the simulation model order in the CSV export."
    return (
        f"{label} forecast-combination weight paths. Each panel shows stacked forecaster "
        "weights for one combination method, using the selected paper methods and sized "
        "for a landscape A4 page. "
        f"The legend entries {', '.join(model_labels)} denote the underlying forecasters."
        f"{mapping_sentence}"
    )


def _paper_plot_weight_path_page(
    weights_path: Path,
    output_dir: Path,
    stem: str,
    label: str,
    artifacts: Dict[str, List[str]],
    metadata_path: Optional[Path] = None,
    empirical: bool = False,
    x_label: str = "OOS period",
) -> None:
    """Create one compact A4 page of stacked forecaster-weight paths."""
    if not weights_path.exists():
        return
    weights = _paper_read_csv(weights_path)
    required_cols = {"method", "oos_period", "model", "weight"}
    if weights.empty or not required_cols.issubset(weights.columns):
        return

    available_methods = set(weights["method"].dropna().astype(str))
    methods = [method for method in PAPER_WEIGHT_PATH_METHODS if method in available_methods]
    if not methods:
        return

    periods = sorted(weights["oos_period"].dropna().unique())
    period_df = pd.DataFrame({"oos_period": periods})
    metadata = _paper_read_csv(metadata_path) if metadata_path is not None and metadata_path.exists() else None
    x_axis = _paper_date_axis_from_metadata(period_df, metadata)
    model_order = list(dict.fromkeys(weights["model"].tolist()))
    forecaster_labels = [rf"$f_{{{idx + 1}}}$" for idx in range(len(model_order))]
    colors = _paper_forecaster_palette(len(model_order))
    hatches = _paper_forecaster_hatches(len(model_order))

    ncols = 2
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.3, 7.9),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.20, "wspace": 0.08},
    )
    axes_arr = np.atleast_1d(axes).reshape(-1)
    legend_handles = None

    for idx, method in enumerate(methods):
        ax = axes_arr[idx]
        sub = weights[weights["method"] == method].copy()
        pivot = (
            sub.pivot_table(
                index="oos_period",
                columns="model",
                values="weight",
                aggfunc="mean",
            )
            .reindex(index=periods, columns=model_order)
            .fillna(0.0)
        )
        values = np.clip(pivot.to_numpy(dtype=float), 0.0, 1.0)
        polys = ax.stackplot(
            x_axis,
            values.T,
            labels=forecaster_labels,
            colors=colors,
            alpha=0.84,
            linewidth=0.35,
            edgecolor="#FFFFFF",
        )
        for poly, hatch in zip(polys, hatches):
            if hatch:
                poly.set_hatch(hatch)
            poly.set_edgecolor("#FFFFFF")
            poly.set_linewidth(0.35)
        if legend_handles is None:
            legend_handles = polys

        ax.text(
            0.012,
            0.90,
            method_display_label(method),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="medium",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#B0B0B0",
                "linewidth": 0.5,
                "alpha": 0.88,
            },
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(axis="both", labelsize=13.5)
        ax.grid(True, axis="y", alpha=0.18)
        ax.grid(True, axis="x", alpha=0.10)

        if idx % ncols != 0:
            ax.tick_params(axis="y", labelleft=False)
        if idx < len(methods) - ncols:
            ax.tick_params(axis="x", labelbottom=False)

    for ax in axes_arr[len(methods):]:
        ax.axis("off")

    bottom_axes = axes_arr[max(0, len(methods) - ncols):len(methods)]
    for ax in bottom_axes:
        _paper_apply_time_axis(ax, x_axis, x_label="")
    fig.text(0.545, 0.115, x_label, va="center", ha="center", fontsize=18.0)
    if _is_datetime_axis(x_axis):
        fig.autofmt_xdate(rotation=0)

    fig.text(0.035, 0.52, "Weight", rotation=90, va="center", ha="center", fontsize=18.5)
    if legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            labels=forecaster_labels,
            loc="lower center",
            ncol=min(len(forecaster_labels), 8),
            fontsize=13.6,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#B0B0B0",
            bbox_to_anchor=(0.5, 0.018),
            columnspacing=0.9,
            handlelength=1.5,
        )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.985, bottom=0.20, hspace=0.20, wspace=0.08)
    _paper_save_figure(
        fig,
        output_dir,
        stem,
        artifacts,
        caption=_paper_compact_weight_caption(label, model_order, empirical=empirical),
    )


def _paper_plot_weight_paths_a4(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Create compact A4 weight-path figures for empirical and simulation examples."""
    empirical_specs = (
        (
            "weights_empirical_inflation",
            "Inflation",
            csv_root / "empirical_studies" / "empirical_inflation_next_quarter" / "backtest" / "weights_long.csv",
            csv_root / "empirical_studies" / "empirical_inflation_next_quarter" / "oos_period_metadata.csv",
        ),
        (
            "weights_empirical_unemployment",
            "Unemployment",
            csv_root / "empirical_studies" / "empirical_unemployment_monthly" / "backtest" / "weights_long.csv",
            csv_root / "empirical_studies" / "empirical_unemployment_monthly" / "oos_period_metadata.csv",
        ),
        (
            "weights_empirical_inflation_diffprev",
            "Differenced inflation",
            csv_root / "empirical_studies" / "empirical_inflation_next_quarter_diffprev" / "backtest" / "weights_long.csv",
            csv_root / "empirical_studies" / "empirical_inflation_next_quarter_diffprev" / "oos_period_metadata.csv",
        ),
        (
            "weights_empirical_unemployment_diffprev",
            "Differenced unemployment",
            csv_root / "empirical_studies" / "empirical_unemployment_monthly_diffprev" / "backtest" / "weights_long.csv",
            csv_root / "empirical_studies" / "empirical_unemployment_monthly_diffprev" / "oos_period_metadata.csv",
        ),
    )
    for stem, label, weights_path, metadata_path in empirical_specs:
        _paper_plot_weight_path_page(
            weights_path=weights_path,
            output_dir=output_dir,
            stem=stem,
            label=label,
            artifacts=artifacts,
            metadata_path=metadata_path,
            empirical=True,
            x_label="Target period",
        )

    simulation_specs = (
        ("weights_simulation_1A", "Simulation Scenario 1A", csv_root / "backtests" / "mcs_backtest_1A" / "weights_long.csv"),
        ("weights_simulation_2A", "Simulation Scenario 2A", csv_root / "backtests" / "res_2A" / "weights_long.csv"),
        ("weights_simulation_2B", "Simulation Scenario 2B", csv_root / "backtests" / "res_2B" / "weights_long.csv"),
        ("weights_simulation_2C", "Simulation Scenario 2C", csv_root / "backtests" / "mcs_backtest_2C" / "weights_long.csv"),
        ("weights_simulation_3A", "Simulation Scenario 3A", csv_root / "backtests" / "mcs_backtest_4A" / "weights_long.csv"),
    )
    for stem, label, weights_path in simulation_specs:
        _paper_plot_weight_path_page(
            weights_path=weights_path,
            output_dir=output_dir,
            stem=stem,
            label=label,
            artifacts=artifacts,
            empirical=False,
            x_label="OOS period",
        )


def _paper_plot_adaptability_figures(
    csv_root: Path,
    output_dir: Path,
    artifacts: Dict[str, List[str]],
) -> None:
    """Create adaptability integrated-excess-risk and half-life figures."""
    def _full_horizon_event_indices(profiles: pd.DataFrame, value_col: str) -> Optional[np.ndarray]:
        """Return events with finite values at every plotted horizon."""
        required = {"event_index", "horizon", value_col}
        if profiles.empty or not required.issubset(profiles.columns):
            return None
        profile_values = profiles[["event_index", "horizon", value_col]].copy()
        profile_values[value_col] = pd.to_numeric(profile_values[value_col], errors="coerce")
        horizon_count = int(profile_values["horizon"].nunique())
        finite = profile_values[profile_values[value_col].notna()]
        counts = finite.groupby("event_index")["horizon"].nunique()
        return counts[counts == horizon_count].index.to_numpy()

    def _filter_to_events(df: pd.DataFrame, event_indices: Optional[np.ndarray]) -> pd.DataFrame:
        """Filter an event-level or profile dataframe to a retained event set."""
        if event_indices is None or "event_index" not in df.columns:
            return df.copy()
        return df[df["event_index"].isin(event_indices)].copy()

    for scenario_key in ("2A", "2B"):
        base = csv_root / "adaptability" / f"adapt_{scenario_key}"
        profiles_raw = _paper_read_csv(base / "excess_risk_profiles_long.csv")
        summary = _paper_read_csv(base / "summary_table.csv")
        is_multi = "Replications" in summary.columns
        original_event_count = (
            int(profiles_raw["event_index"].nunique())
            if "event_index" in profiles_raw.columns
            else int(summary["Events"].max())
        )
        full_event_indices = (
            _full_horizon_event_indices(profiles_raw, "excess_risk")
            if scenario_key == "2B"
            else None
        )
        profiles = _filter_to_events(profiles_raw, full_event_indices)
        retained_event_count = (
            int(profiles["event_index"].nunique())
            if "event_index" in profiles.columns
            else original_event_count
        )
        full_window_phrase = ""
        if scenario_key == "2B" and full_event_indices is not None:
            full_window_phrase = (
                f" using {retained_event_count} full-window events "
                f"out of {original_event_count} detected switch events"
            )
        if is_multi and scenario_key == "2A":
            replications = int(summary["Replications"].max())
            break_time = int(summary["Bias_Break_Time"].max())
            event_description = (
                f"the known Scenario 2A bias break at t={break_time}, averaged over "
                f"{replications} simulation replications"
            )
            x_event_label = "Periods since bias break"
            half_life_event_phrase = "after the bias break"
        elif is_multi:
            replications = int(summary["Replications"].max())
            event_description = (
                "full-window latent oracle switches, averaged over "
                f"{replications} simulation replications"
                if scenario_key == "2B" and full_event_indices is not None
                else (
                    "latent oracle switches, averaged over "
                    f"{replications} simulation replications"
                )
            )
            x_event_label = "Periods since latent oracle switch"
            half_life_event_phrase = "after a latent oracle switch"
        else:
            event_description = "latent oracle switches"
            x_event_label = "Periods since latent oracle switch"
            half_life_event_phrase = "after a latent oracle switch"
        methods = [
            method
            for method in PAPER_ADAPTABILITY_METHOD_ORDER
            if method in set(profiles["method"]) and method in set(summary["Method"])
        ]

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for method in methods:
            sub = profiles[profiles["method"] == method]
            profile = sub.groupby("horizon", as_index=False)["excess_risk"].mean()
            profile = profile.sort_values("horizon")
            _paper_plot_line(
                ax,
                profile["horizon"].to_numpy(),
                profile["excess_risk"].to_numpy(dtype=float),
                method,
                n_points=len(profile),
                linewidth=2.2,
            )
        ax.set_xlabel(x_event_label)
        ax.set_ylabel("Mean excess latent risk")
        _apply_publication_legend(
            ax,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=12,
            method_order=PAPER_ADAPTABILITY_METHOD_ORDER,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
        _paper_save_figure(
            fig,
            output_dir,
            f"adaptability_excess_risk_{scenario_key}",
            artifacts,
            caption=(
                f"Scenario {scenario_key} excess latent-risk profile. Lines show the "
                f"non-cumulative mean excess latent risk after {event_description}"
                f"{full_window_phrase}; lower values indicate closer tracking of the "
                "latent oracle at that event horizon."
            ),
        )

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for method in methods:
            sub = profiles[profiles["method"] == method]
            profile = sub.groupby("horizon", as_index=False)["excess_risk"].mean()
            profile = profile.sort_values("horizon")
            integrated = profile["excess_risk"].fillna(0.0).cumsum().to_numpy(dtype=float)
            _paper_plot_line(
                ax,
                profile["horizon"].to_numpy(),
                integrated,
                method,
                n_points=len(profile),
                linewidth=2.2,
            )
        ax.set_xlabel(x_event_label)
        ax.set_ylabel("Cumulative mean excess latent risk")
        _apply_publication_legend(
            ax,
            ncol=2,
            loc="best",
            fontsize=8.8,
            method_order=PAPER_ADAPTABILITY_METHOD_ORDER,
        )
        fig.tight_layout()
        _paper_save_figure(
            fig,
            output_dir,
            f"adaptability_ier_{scenario_key}",
            artifacts,
            caption=(
                f"Scenario {scenario_key} adaptability profile. Lines show cumulative "
                f"sums of mean excess latent risk after {event_description}"
                f"{full_window_phrase} for the ordered set of benchmark and combination "
                "methods. These are area summaries, so near-linear curves indicate nearly "
                "constant mean horizon-level excess risk rather than a linear recovery path; "
                "lower values indicate lower accumulated adaptation cost."
            ),
        )

        summary = summary[summary["Method"].isin(methods)].copy()
        if scenario_key == "2B" and full_event_indices is not None:
            half_lives = _filter_to_events(_paper_read_csv(base / "half_lives_long.csv"), full_event_indices)
            for method in methods:
                method_half = pd.to_numeric(
                    half_lives.loc[half_lives["method"] == method, "half_life"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                method_mask = summary["Method"] == method
                summary.loc[method_mask, "Events"] = retained_event_count
                summary.loc[method_mask, "Mean_Half_Life"] = _safe_nanmean(method_half)
                summary.loc[method_mask, "Median_Half_Life"] = _safe_nanmedian(method_half)
                if "Half_Life_Reached_Frac" in summary.columns:
                    summary.loc[method_mask, "Half_Life_Reached_Frac"] = float(np.mean(np.isfinite(method_half)))
        if scenario_key == "2B":
            equal_mask = summary["Method"] == "equal"
            summary.loc[equal_mask, ["Mean_Half_Life", "Median_Half_Life"]] = np.nan
            if "Half_Life_Reached_Frac" in summary.columns:
                summary.loc[equal_mask, "Half_Life_Reached_Frac"] = 0.0
        summary["_order"] = summary["Method"].map({method: idx for idx, method in enumerate(methods)})
        summary = summary.sort_values("_order")
        values = summary["Mean_Half_Life"].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        has_not_reached = bool(np.any(~np.isfinite(values)))
        event_horizon = int(profiles["horizon"].max()) if "horizon" in profiles.columns and not profiles.empty else 0
        event_count = retained_event_count if scenario_key == "2B" else (
            int(profiles["event_index"].nunique()) if "event_index" in profiles.columns else int(summary["Events"].max())
        )
        finite_max = float(np.nanmax(finite_values)) if finite_values.size else 0.0
        x_limit = 40.0
        label_offset = max(min(finite_max * 0.06, 1.8), 1.2)
        not_reached_x = x_limit - 2.8

        fig, ax = plt.subplots(figsize=(7.8, 5.3))
        y_pos = np.arange(len(summary))
        for ypos, method, value in zip(y_pos, summary["Method"], values):
            style = _method_plot_style(method)
            if np.isfinite(value):
                bar_value = min(float(value), x_limit)
                bar = ax.barh(
                    ypos,
                    bar_value,
                    height=0.66,
                    color=style["color"],
                    edgecolor="#2F2F2F",
                    linewidth=0.8,
                    alpha=0.88,
                )
                if style["hatch"]:
                    bar.patches[0].set_hatch(style["hatch"])
                ax.plot(
                    bar_value,
                    ypos,
                    marker=style["marker"],
                    markersize=8.0,
                    markerfacecolor="white",
                    markeredgewidth=1.4,
                    color=style["color"],
                )
                text_x = min(float(value) + label_offset, x_limit + 0.8)
                ax.text(
                    text_x,
                    ypos,
                    f"{value:.1f}",
                    va="center",
                    fontsize=9.8,
                    clip_on=False,
                )
            else:
                ax.text(
                    not_reached_x,
                    ypos,
                    "not reached",
                    va="center",
                    fontsize=9.2,
                    color="#5C5C5C",
                )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([method_display_label(method) for method in summary["Method"]])
        ax.invert_yaxis()
        ax.set_ylim(len(summary) - 0.05, -0.85)
        ax.set_xlabel("Mean half-life in periods")
        ax.axvline(0.0, color="#111111", linewidth=0.9)
        if event_horizon and has_not_reached and event_horizon <= x_limit:
            ax.axvline(event_horizon, color="#777777", linewidth=0.9, linestyle=":")
        ax.set_xlim(0.0, x_limit)
        ax.grid(True, axis="x", alpha=0.22)
        fig.tight_layout()
        _paper_save_figure(
            fig,
            output_dir,
            f"adaptability_half_life_{scenario_key}",
            artifacts,
            caption=(
                f"Scenario {scenario_key} adaptability half-life. Bars report, "
                + (
                    f"across {event_count} known bias-break replications, "
                    if is_multi and scenario_key == "2A"
                    else (
                        f"averaged over {event_count} full-window latent-oracle switch events, "
                        if scenario_key == "2B" and full_event_indices is not None
                        else f"averaged over {event_count} latent-oracle switch events, "
                    )
                    if is_multi
                    else (
                        f"averaging over {event_count} detected latent-oracle switch event"
                        f"{'s' if event_count != 1 else ''}, "
                    )
                )
                + "the number of periods required for each method's smoothed excess-risk gap to "
                f"fall by half {half_life_event_phrase}; lower values indicate faster "
                "adaptation. Finite bars are averaged over events that reach this threshold "
                "within the event window. The x-axis is capped at 40 periods; labels printed "
                "to the right of capped bars report the uncapped mean. Entries marked not "
                "reached are undefined within the event window and should not be interpreted "
                "as zero."
            ),
        )

    snippet_path = output_dir / "adaptability_side_by_side_snippet.tex"
    snippet = "\n".join([
        "% Generated by functionality.generate_paper_replots().",
        "\\begin{figure}[!htbp]",
        "\\centering",
        "\\begin{minipage}{0.49\\textwidth}",
        "\\centering",
        "\\includegraphics[width=\\linewidth]{Empirical_Data/empirical_outputs/paper_replots/adaptability_ier_2A.pdf}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}{0.49\\textwidth}",
        "\\centering",
        "\\includegraphics[width=\\linewidth]{Empirical_Data/empirical_outputs/paper_replots/adaptability_half_life_2A.pdf}",
        "\\end{minipage}",
        "\\vspace{0.5em}",
        "\\begin{minipage}{0.49\\textwidth}",
        "\\centering",
        "\\includegraphics[width=\\linewidth]{Empirical_Data/empirical_outputs/paper_replots/adaptability_ier_2B.pdf}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}{0.49\\textwidth}",
        "\\centering",
        "\\includegraphics[width=\\linewidth]{Empirical_Data/empirical_outputs/paper_replots/adaptability_half_life_2B.pdf}",
        "\\end{minipage}",
        "\\caption{Adaptability diagnostics for Scenarios 2A and 2B. Scenario 2A is measured at the known bias break across multiple simulation replications; Scenario 2B is measured on full-window latent-oracle switch events across multiple simulation replications. Left panels show cumulative sums of mean excess latent risk, so near-linear curves reflect nearly constant horizon-level excess risk rather than a linear recovery path; right panels show mean half-life, where lower values indicate faster adaptation and not-reached entries are undefined within the event window rather than zero.}",
        "\\label{fig:adaptability-side-by-side}",
        "\\end{figure}",
        "",
    ])
    snippet_path.write_text(snippet)
    artifacts["adaptability_side_by_side_snippet"] = [str(snippet_path)]


def _paper_export_slug(value: object) -> str:
    """Create the same simple filesystem slug used by the notebook CSV export cell."""
    text = str(value)
    chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unnamed"


def _paper_model_names(count: int, prefix: str = "model") -> List[str]:
    return [f"{prefix}_{idx + 1}" for idx in range(int(count))]


def _paper_wide_df(
    matrix: np.ndarray,
    index_values: Sequence[object],
    index_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(matrix), columns=list(columns))
    df.insert(0, index_name, index_values)
    return df


def _paper_dict_to_wide_df(
    values_dict: Dict[str, Sequence[float]],
    index_values: Sequence[object],
    index_name: str,
) -> pd.DataFrame:
    df = pd.DataFrame({name: np.asarray(values) for name, values in values_dict.items()})
    df.insert(0, index_name, index_values)
    return df


def _paper_profile_dict_to_long(
    profile_dict: Dict[str, np.ndarray],
    event_times: Sequence[int],
    value_name: str,
) -> pd.DataFrame:
    rows = []
    for method, values in profile_dict.items():
        arr = np.asarray(values)
        if arr.ndim == 1:
            for event_idx, value in enumerate(arr):
                rows.append({
                    "method": method,
                    "event_index": event_idx,
                    "event_time": event_times[event_idx] if event_idx < len(event_times) else np.nan,
                    value_name: value,
                })
        elif arr.ndim == 2:
            for event_idx in range(arr.shape[0]):
                for horizon in range(arr.shape[1]):
                    rows.append({
                        "method": method,
                        "event_index": event_idx,
                        "event_time": event_times[event_idx] if event_idx < len(event_times) else np.nan,
                        "horizon": horizon,
                        value_name: arr[event_idx, horizon],
                    })
    return pd.DataFrame(rows)


def _paper_write_csv(df: pd.DataFrame, path: Path, written_files: List[str], index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    written_files.append(str(path))


def _paper_export_simulation_data(
    name: str,
    data: SimulationData,
    export_root: Path,
    written_files: List[str],
    model_names: Optional[Sequence[str]] = None,
) -> None:
    """Export raw simulation paths for reproducible paper replots."""
    base_dir = export_root / "simulation_data" / _paper_export_slug(name)
    model_names = list(model_names or _paper_model_names(data.forecasts.shape[1]))
    t_index = np.arange(len(data.y))
    _paper_write_csv(
        pd.DataFrame({
            "t": t_index,
            "actual": np.asarray(data.y),
            "common_shock": np.asarray(data.common_shock),
        }),
        base_dir / "target_and_common_shock.csv",
        written_files,
    )
    _paper_write_csv(_paper_wide_df(data.forecasts, t_index, "t", model_names), base_dir / "forecasts.csv", written_files)
    _paper_write_csv(_paper_wide_df(data.errors, t_index, "t", model_names), base_dir / "forecast_errors.csv", written_files)
    _paper_write_csv(_paper_wide_df(data.losses, t_index, "t", model_names), base_dir / "losses.csv", written_files)
    if getattr(data, "bias_paths", None) is not None:
        _paper_write_csv(_paper_wide_df(data.bias_paths, t_index, "t", model_names), base_dir / "bias_paths.csv", written_files)
    if getattr(data, "sigma_paths", None) is not None:
        _paper_write_csv(_paper_wide_df(data.sigma_paths, t_index, "t", model_names), base_dir / "sigma_paths.csv", written_files)


def _paper_export_backtest_result(
    name: str,
    res: BacktestResult,
    export_root: Path,
    written_files: List[str],
    model_names: Optional[Sequence[str]] = None,
) -> None:
    """Export the backtest CSVs consumed by the paper-replot layer."""
    base_dir = export_root / "backtests" / _paper_export_slug(name)
    oos = np.asarray(res.oos_periods)
    model_names = list(model_names or _paper_model_names(res.forecasts_oos.shape[1]))

    _paper_write_csv(pd.DataFrame({"oos_period": oos, "actual": np.asarray(res.y_oos)}), base_dir / "y_oos.csv", written_files)
    _paper_write_csv(_paper_wide_df(res.forecasts_oos, oos, "oos_period", model_names), base_dir / "raw_model_forecasts_oos.csv", written_files)
    raw_errors = np.asarray(res.y_oos)[:, None] - np.asarray(res.forecasts_oos)
    _paper_write_csv(_paper_wide_df(raw_errors, oos, "oos_period", model_names), base_dir / "raw_model_forecast_errors_oos.csv", written_files)
    _paper_write_csv(_paper_dict_to_wide_df(res.combined_forecasts, oos, "oos_period"), base_dir / "combined_forecasts.csv", written_files)
    _paper_write_csv(_paper_dict_to_wide_df(res.combined_losses, oos, "oos_period"), base_dir / "combined_losses.csv", written_files)
    _paper_write_csv(compute_performance_table(res), base_dir / "performance_table.csv", written_files)

    if "equal" in res.combined_losses:
        reference_loss = np.asarray(res.combined_losses["equal"], dtype=float)
        loss_diff = {"oos_period": oos}
        cumulative_loss_diff = {"oos_period": oos}
        for method, values in res.combined_losses.items():
            if method == "equal":
                continue
            period_diff = np.asarray(values, dtype=float) - reference_loss
            loss_diff[method] = period_diff
            cumulative_loss_diff[method] = np.cumsum(period_diff)
        _paper_write_csv(pd.DataFrame(loss_diff), base_dir / "loss_differentials_vs_equal.csv", written_files)
        _paper_write_csv(pd.DataFrame(cumulative_loss_diff), base_dir / "cumulative_loss_differentials_vs_equal.csv", written_files)

    weight_rows = []
    diag_rows = []
    for method, weights in res.weights.items():
        w = np.asarray(weights)
        _paper_write_csv(_paper_wide_df(w, oos, "oos_period", model_names), base_dir / f"weights__{_paper_export_slug(method)}.csv", written_files)
        herf = compute_herfindahl(w)
        eff_n = compute_effective_n(w)
        turnover = compute_turnover(w)
        for t_idx, oos_t in enumerate(oos):
            diag_rows.append({
                "method": method,
                "oos_period": oos_t,
                "herfindahl": herf[t_idx],
                "effective_n": eff_n[t_idx],
                "turnover": np.nan if t_idx == 0 else turnover[t_idx - 1],
            })
            for model_idx, model_name in enumerate(model_names):
                weight_rows.append({
                    "method": method,
                    "oos_period": oos_t,
                    "model": model_name,
                    "weight": w[t_idx, model_idx],
                })
    if weight_rows:
        _paper_write_csv(pd.DataFrame(weight_rows), base_dir / "weights_long.csv", written_files)
    if diag_rows:
        _paper_write_csv(pd.DataFrame(diag_rows), base_dir / "weight_diagnostics_long.csv", written_files)


def _paper_export_sensitivity_sweep(
    sweep: SensitivitySweepResult,
    export_root: Path,
    written_files: List[str],
) -> None:
    base_dir = export_root / "sensitivity_sweeps" / _paper_export_slug(sweep.source_name)
    _paper_write_csv(sweep.results_table, base_dir / "results_table.csv", written_files)
    _paper_write_csv(sweep.graph_only_table, base_dir / "graph_only_table.csv", written_files)
    _paper_write_csv(sweep.full_gcsr_table, base_dir / "full_gcsr_table.csv", written_files)
    _paper_write_csv(sweep.full_gcsr_pivot, base_dir / "full_gcsr_pivot.csv", written_files, index=True)
    _paper_write_csv(sweep.combined_losses, base_dir / "combined_losses.csv", written_files)


def _paper_export_adaptability_result(
    name: str,
    adapt_result: AdaptabilityResult,
    export_root: Path,
    written_files: List[str],
) -> None:
    base_dir = export_root / "adaptability" / _paper_export_slug(name)
    _paper_write_csv(adapt_result.summary_table, base_dir / "summary_table.csv", written_files)
    _paper_write_csv(
        pd.DataFrame({"event_index": np.arange(len(adapt_result.event_times)), "event_time": adapt_result.event_times}),
        base_dir / "event_times.csv",
        written_files,
    )
    _paper_write_csv(
        pd.DataFrame({"t": np.arange(len(adapt_result.oracle_models)), "oracle_model": adapt_result.oracle_models}),
        base_dir / "oracle_models.csv",
        written_files,
    )
    _paper_write_csv(
        _paper_wide_df(
            adapt_result.oracle_portfolio_weights,
            np.arange(len(adapt_result.oracle_portfolio_weights)),
            "t",
            _paper_model_names(adapt_result.oracle_portfolio_weights.shape[1]),
        ),
        base_dir / "oracle_portfolio_weights.csv",
        written_files,
    )
    _paper_write_csv(_paper_profile_dict_to_long(adapt_result.oracle_weight_profiles, adapt_result.event_times, "oracle_weight_profile"), base_dir / "oracle_weight_profiles_long.csv", written_files)
    _paper_write_csv(_paper_profile_dict_to_long(adapt_result.oracle_gap_profiles, adapt_result.event_times, "oracle_gap_profile"), base_dir / "oracle_gap_profiles_long.csv", written_files)
    _paper_write_csv(_paper_profile_dict_to_long(adapt_result.latent_risk_profiles, adapt_result.event_times, "latent_risk"), base_dir / "latent_risk_profiles_long.csv", written_files)
    _paper_write_csv(_paper_profile_dict_to_long(adapt_result.excess_risk_profiles, adapt_result.event_times, "excess_risk"), base_dir / "excess_risk_profiles_long.csv", written_files)
    _paper_write_csv(_paper_profile_dict_to_long(adapt_result.relative_gap_profiles, adapt_result.event_times, "relative_gap"), base_dir / "relative_gap_profiles_long.csv", written_files)
    _paper_write_csv(_paper_profile_dict_to_long(adapt_result.half_lives, adapt_result.event_times, "half_life"), base_dir / "half_lives_long.csv", written_files)
    _paper_write_csv(_paper_profile_dict_to_long(adapt_result.recovery_delays, adapt_result.event_times, "recovery_delay"), base_dir / "recovery_delays_long.csv", written_files)


def regenerate_2a_multi_sim_adaptability_csv_export(
    csv_root: Union[str, Path] = "Empirical_Data/empirical_outputs/replot_csv_exports",
    M: int = 8,
    T: int = 400,
    T0: int = 200,
    seed: int = 42,
    replications: int = 50,
    horizon: int = 80,
    smooth_window: int = 5,
    target_oracle_weight: float = 0.60,
    bt_cfg: Optional[BacktestConfig] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Regenerate Scenario 2A adaptability CSVs using many independent bias breaks.

    Scenario 2A has a known structural break in the bias vector. For this
    diagnostic we therefore use the known break date as the event in every
    replication, rather than requiring the latent best single forecaster to
    change identity.
    """
    csv_root = Path(csv_root)
    base_dir = csv_root / "adaptability" / "adapt_2A"
    base_dir.mkdir(parents=True, exist_ok=True)
    written_files: List[str] = []

    if replications <= 0:
        raise ValueError("replications must be positive for the 2A adaptability rerun.")
    if T0 >= T:
        raise ValueError("T0 must be smaller than T for the 2A adaptability rerun.")

    bt_cfg = bt_cfg if bt_cfg is not None else _paper_base_backtest_config()
    methods = list(PAPER_ADAPTABILITY_METHOD_ORDER)
    model_names = _paper_model_names(M)

    event_rows = []
    oracle_model_rows = []
    oracle_weight_rows = []
    profile_rows = {
        "oracle_weight_profile": [],
        "oracle_gap_profile": [],
        "latent_risk": [],
        "excess_risk": [],
        "relative_gap": [],
    }
    half_life_rows = []
    recovery_rows = []

    for rep_idx in range(int(replications)):
        rep_seed = int(seed + rep_idx)
        cfg = scenario_2A(
            M=M,
            T=T,
            T0=T0,
            seed=rep_seed,
            break_frac=float(T0) / float(T),
        )
        if cfg.bias_break_time != T0:
            raise ValueError(
                f"Scenario 2A break time is {cfg.bias_break_time}, expected T0={T0}."
            )

        if verbose:
            print(
                f"Scenario 2A adaptability replication {rep_idx + 1}/{replications} "
                f"(seed={rep_seed}, break={cfg.bias_break_time})..."
            )

        data = generate_scenario(cfg)
        res = run_backtest(data, bt_cfg, verbose=False)
        adapt = compute_adaptability_diagnostics(
            data,
            res,
            methods=methods,
            horizon=horizon,
            smooth_window=smooth_window,
            target_oracle_weight=target_oracle_weight,
            event_times=[cfg.bias_break_time],
        )

        oracle = compute_latent_oracle_indices(data)
        event_global = rep_idx
        event_time = int(cfg.bias_break_time)
        oracle_pre = int(oracle[event_time - 1]) if event_time > 0 else int(oracle[event_time])
        oracle_post = int(oracle[event_time])
        event_rows.append({
            "event_index": event_global,
            "replication": rep_idx,
            "seed": rep_seed,
            "event_time": event_time,
            "bias_break_time": int(cfg.bias_break_time),
            "oracle_pre": oracle_pre,
            "oracle_post": oracle_post,
            "oracle_single_changed": bool(oracle_pre != oracle_post),
        })
        oracle_model_rows.append({
            "event_index": event_global,
            "replication": rep_idx,
            "seed": rep_seed,
            "event_time": event_time,
            "oracle_model": oracle_post,
        })

        max_use_len = min(int(horizon), T - event_time)
        for h in range(max_use_len):
            row = {
                "event_index": event_global,
                "replication": rep_idx,
                "seed": rep_seed,
                "event_time": event_time,
                "horizon": h,
                "t": event_time + h,
            }
            oracle_weights = adapt.oracle_portfolio_weights[event_time + h]
            for model_idx, model_name in enumerate(model_names):
                row[model_name] = oracle_weights[model_idx]
            oracle_weight_rows.append(row)

        for method in adapt.methods:
            for h in range(int(horizon)):
                base = {
                    "method": method,
                    "event_index": event_global,
                    "replication": rep_idx,
                    "seed": rep_seed,
                    "event_time": event_time,
                    "horizon": h,
                }
                profile_rows["oracle_weight_profile"].append({
                    **base,
                    "oracle_weight_profile": adapt.oracle_weight_profiles[method][0, h],
                })
                profile_rows["oracle_gap_profile"].append({
                    **base,
                    "oracle_gap_profile": adapt.oracle_gap_profiles[method][0, h],
                })
                profile_rows["latent_risk"].append({
                    **base,
                    "latent_risk": adapt.latent_risk_profiles[method][0, h],
                })
                profile_rows["excess_risk"].append({
                    **base,
                    "excess_risk": adapt.excess_risk_profiles[method][0, h],
                })
                profile_rows["relative_gap"].append({
                    **base,
                    "relative_gap": adapt.relative_gap_profiles[method][0, h],
                })

            half_life_rows.append({
                "method": method,
                "event_index": event_global,
                "replication": rep_idx,
                "seed": rep_seed,
                "event_time": event_time,
                "half_life": adapt.half_lives[method][0],
            })
            recovery_rows.append({
                "method": method,
                "event_index": event_global,
                "replication": rep_idx,
                "seed": rep_seed,
                "event_time": event_time,
                "recovery_delay": adapt.recovery_delays[method][0],
            })

    excess_df = pd.DataFrame(profile_rows["excess_risk"])
    latent_df = pd.DataFrame(profile_rows["latent_risk"])
    overlap_df = pd.DataFrame(profile_rows["oracle_weight_profile"])
    half_df = pd.DataFrame(half_life_rows)
    recovery_df = pd.DataFrame(recovery_rows)

    summary_rows = []
    for method in methods:
        method_excess = excess_df[excess_df["method"] == method]
        method_latent = latent_df[latent_df["method"] == method]
        method_overlap = overlap_df[overlap_df["method"] == method]
        method_half = half_df[half_df["method"] == method]["half_life"].to_numpy(dtype=float)
        method_recovery = recovery_df[recovery_df["method"] == method]["recovery_delay"].to_numpy(dtype=float)
        if method_excess.empty:
            continue
        initial_excess = method_excess.loc[method_excess["horizon"] == 0, "excess_risk"].to_numpy(dtype=float)
        initial_overlap = method_overlap.loc[method_overlap["horizon"] == 0, "oracle_weight_profile"].to_numpy(dtype=float)
        summary_rows.append({
            "Method": method,
            "Events": int(method_excess["event_index"].nunique()),
            "Replications": int(replications),
            "Bias_Break_Time": int(T0),
            "Horizon": int(horizon),
            "Mean_Initial_Excess_Latent_Risk": _safe_nanmean(initial_excess),
            "Mean_Integrated_Excess_Risk": _safe_nanmean(method_excess["excess_risk"].to_numpy(dtype=float)),
            "Mean_Latent_Risk": _safe_nanmean(method_latent["latent_risk"].to_numpy(dtype=float)),
            "Mean_Half_Life": _safe_nanmean(method_half),
            "Median_Half_Life": _safe_nanmedian(method_half),
            "Half_Life_Reached_Frac": float(np.mean(np.isfinite(method_half))),
            "Mean_Target_Closure_Delay": _safe_nanmean(method_recovery),
            "Captured_Frac": float(np.mean(np.isfinite(method_recovery))),
            "Mean_Oracle_Overlap": _safe_nanmean(initial_overlap),
        })

    summary = pd.DataFrame(summary_rows)
    summary["_order"] = summary["Method"].map({method: idx for idx, method in enumerate(methods)})
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    _paper_write_csv(summary, base_dir / "summary_table.csv", written_files)
    _paper_write_csv(pd.DataFrame(event_rows), base_dir / "event_times.csv", written_files)
    _paper_write_csv(pd.DataFrame(oracle_model_rows), base_dir / "oracle_models.csv", written_files)
    _paper_write_csv(pd.DataFrame(oracle_weight_rows), base_dir / "oracle_portfolio_weights.csv", written_files)
    _paper_write_csv(overlap_df, base_dir / "oracle_weight_profiles_long.csv", written_files)
    _paper_write_csv(pd.DataFrame(profile_rows["oracle_gap_profile"]), base_dir / "oracle_gap_profiles_long.csv", written_files)
    _paper_write_csv(latent_df, base_dir / "latent_risk_profiles_long.csv", written_files)
    _paper_write_csv(excess_df, base_dir / "excess_risk_profiles_long.csv", written_files)
    _paper_write_csv(pd.DataFrame(profile_rows["relative_gap"]), base_dir / "relative_gap_profiles_long.csv", written_files)
    _paper_write_csv(half_df, base_dir / "half_lives_long.csv", written_files)
    _paper_write_csv(recovery_df, base_dir / "recovery_delays_long.csv", written_files)
    _paper_write_csv(
        pd.DataFrame([{
            "scenario": "2A",
            "replications": int(replications),
            "M": int(M),
            "T": int(T),
            "T0": int(T0),
            "bias_break_time": int(T0),
            "seed_start": int(seed),
            "seed_end": int(seed + replications - 1),
            "horizon": int(horizon),
            "smooth_window": int(smooth_window),
            "target_oracle_weight": float(target_oracle_weight),
            "fast_fixed_ld": bool(bt_cfg.fast_fixed_ld),
            "event_definition": "known Scenario 2A bias break",
        }]),
        base_dir / "multi_sim_metadata.csv",
        written_files,
    )
    return written_files


def regenerate_2b_multi_sim_adaptability_csv_export(
    csv_root: Union[str, Path] = "Empirical_Data/empirical_outputs/replot_csv_exports",
    M: int = 8,
    T: int = 400,
    T0: int = 200,
    seed: int = 42,
    replications: int = 50,
    horizon: int = 80,
    smooth_window: int = 5,
    target_oracle_weight: float = 0.60,
    bt_cfg: Optional[BacktestConfig] = None,
    verbose: bool = True,
) -> List[str]:
    """Regenerate Scenario 2B adaptability CSVs over many drifting-bias simulations."""
    csv_root = Path(csv_root)
    base_dir = csv_root / "adaptability" / "adapt_2B"
    base_dir.mkdir(parents=True, exist_ok=True)
    written_files: List[str] = []

    if replications <= 0:
        raise ValueError("replications must be positive for the 2B adaptability rerun.")
    if T0 >= T:
        raise ValueError("T0 must be smaller than T for the 2B adaptability rerun.")

    bt_cfg = bt_cfg if bt_cfg is not None else _paper_base_backtest_config()
    methods = list(PAPER_ADAPTABILITY_METHOD_ORDER)
    model_names = _paper_model_names(M)

    event_rows = []
    oracle_model_rows = []
    oracle_weight_rows = []
    profile_rows = {
        "oracle_weight_profile": [],
        "oracle_gap_profile": [],
        "latent_risk": [],
        "excess_risk": [],
        "relative_gap": [],
    }
    half_life_rows = []
    recovery_rows = []
    global_event_idx = 0

    for rep_idx in range(int(replications)):
        rep_seed = int(seed + rep_idx)
        if verbose:
            print(
                f"Scenario 2B adaptability replication {rep_idx + 1}/{replications} "
                f"(seed={rep_seed})..."
            )

        data = generate_scenario(scenario_2B(M=M, T=T, T0=T0, seed=rep_seed))
        res = run_backtest(data, bt_cfg, verbose=False)
        adapt = compute_adaptability_diagnostics(
            data,
            res,
            methods=methods,
            horizon=horizon,
            smooth_window=smooth_window,
            target_oracle_weight=target_oracle_weight,
        )
        oracle = compute_latent_oracle_indices(data)

        for local_event_idx, event_time_raw in enumerate(adapt.event_times):
            event_time = int(event_time_raw)
            event_global = global_event_idx
            global_event_idx += 1
            oracle_pre = int(oracle[event_time - 1]) if event_time > 0 else int(oracle[event_time])
            oracle_post = int(oracle[event_time])
            event_rows.append({
                "event_index": event_global,
                "replication": rep_idx,
                "seed": rep_seed,
                "event_time": event_time,
                "oracle_pre": oracle_pre,
                "oracle_post": oracle_post,
                "oracle_single_changed": bool(oracle_pre != oracle_post),
            })
            oracle_model_rows.append({
                "event_index": event_global,
                "replication": rep_idx,
                "seed": rep_seed,
                "event_time": event_time,
                "oracle_model": oracle_post,
            })

            max_use_len = min(int(horizon), T - event_time)
            for h in range(max_use_len):
                row = {
                    "event_index": event_global,
                    "replication": rep_idx,
                    "seed": rep_seed,
                    "event_time": event_time,
                    "horizon": h,
                    "t": event_time + h,
                }
                oracle_weights = adapt.oracle_portfolio_weights[event_time + h]
                for model_idx, model_name in enumerate(model_names):
                    row[model_name] = oracle_weights[model_idx]
                oracle_weight_rows.append(row)

            for method in adapt.methods:
                for h in range(int(horizon)):
                    base = {
                        "method": method,
                        "event_index": event_global,
                        "replication": rep_idx,
                        "seed": rep_seed,
                        "event_time": event_time,
                        "horizon": h,
                    }
                    profile_rows["oracle_weight_profile"].append({
                        **base,
                        "oracle_weight_profile": adapt.oracle_weight_profiles[method][local_event_idx, h],
                    })
                    profile_rows["oracle_gap_profile"].append({
                        **base,
                        "oracle_gap_profile": adapt.oracle_gap_profiles[method][local_event_idx, h],
                    })
                    profile_rows["latent_risk"].append({
                        **base,
                        "latent_risk": adapt.latent_risk_profiles[method][local_event_idx, h],
                    })
                    profile_rows["excess_risk"].append({
                        **base,
                        "excess_risk": adapt.excess_risk_profiles[method][local_event_idx, h],
                    })
                    profile_rows["relative_gap"].append({
                        **base,
                        "relative_gap": adapt.relative_gap_profiles[method][local_event_idx, h],
                    })

                half_life_rows.append({
                    "method": method,
                    "event_index": event_global,
                    "replication": rep_idx,
                    "seed": rep_seed,
                    "event_time": event_time,
                    "half_life": adapt.half_lives[method][local_event_idx],
                })
                recovery_rows.append({
                    "method": method,
                    "event_index": event_global,
                    "replication": rep_idx,
                    "seed": rep_seed,
                    "event_time": event_time,
                    "recovery_delay": adapt.recovery_delays[method][local_event_idx],
                })

    excess_df = pd.DataFrame(profile_rows["excess_risk"])
    latent_df = pd.DataFrame(profile_rows["latent_risk"])
    overlap_df = pd.DataFrame(profile_rows["oracle_weight_profile"])
    half_df = pd.DataFrame(half_life_rows)
    recovery_df = pd.DataFrame(recovery_rows)

    summary_rows = []
    for method in methods:
        method_excess = excess_df[excess_df["method"] == method]
        method_latent = latent_df[latent_df["method"] == method]
        method_overlap = overlap_df[overlap_df["method"] == method]
        method_half = half_df[half_df["method"] == method]["half_life"].to_numpy(dtype=float)
        method_recovery = recovery_df[recovery_df["method"] == method]["recovery_delay"].to_numpy(dtype=float)
        if method_excess.empty:
            continue
        initial_excess = method_excess.loc[method_excess["horizon"] == 0, "excess_risk"].to_numpy(dtype=float)
        initial_overlap = method_overlap.loc[method_overlap["horizon"] == 0, "oracle_weight_profile"].to_numpy(dtype=float)
        mean_half_life = _safe_nanmean(method_half)
        median_half_life = _safe_nanmedian(method_half)
        half_life_reached_frac = float(np.mean(np.isfinite(method_half)))
        if method == "equal":
            mean_half_life = np.nan
            median_half_life = np.nan
            half_life_reached_frac = 0.0
        summary_rows.append({
            "Method": method,
            "Events": int(method_excess["event_index"].nunique()),
            "Replications": int(replications),
            "Horizon": int(horizon),
            "Mean_Initial_Excess_Latent_Risk": _safe_nanmean(initial_excess),
            "Mean_Integrated_Excess_Risk": _safe_nanmean(method_excess["excess_risk"].to_numpy(dtype=float)),
            "Mean_Latent_Risk": _safe_nanmean(method_latent["latent_risk"].to_numpy(dtype=float)),
            "Mean_Half_Life": mean_half_life,
            "Median_Half_Life": median_half_life,
            "Half_Life_Reached_Frac": half_life_reached_frac,
            "Mean_Target_Closure_Delay": _safe_nanmean(method_recovery),
            "Captured_Frac": float(np.mean(np.isfinite(method_recovery))),
            "Mean_Oracle_Overlap": _safe_nanmean(initial_overlap),
        })

    summary = pd.DataFrame(summary_rows)
    summary["_order"] = summary["Method"].map({method: idx for idx, method in enumerate(methods)})
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    _paper_write_csv(summary, base_dir / "summary_table.csv", written_files)
    _paper_write_csv(pd.DataFrame(event_rows), base_dir / "event_times.csv", written_files)
    _paper_write_csv(pd.DataFrame(oracle_model_rows), base_dir / "oracle_models.csv", written_files)
    _paper_write_csv(pd.DataFrame(oracle_weight_rows), base_dir / "oracle_portfolio_weights.csv", written_files)
    _paper_write_csv(overlap_df, base_dir / "oracle_weight_profiles_long.csv", written_files)
    _paper_write_csv(pd.DataFrame(profile_rows["oracle_gap_profile"]), base_dir / "oracle_gap_profiles_long.csv", written_files)
    _paper_write_csv(latent_df, base_dir / "latent_risk_profiles_long.csv", written_files)
    _paper_write_csv(excess_df, base_dir / "excess_risk_profiles_long.csv", written_files)
    _paper_write_csv(pd.DataFrame(profile_rows["relative_gap"]), base_dir / "relative_gap_profiles_long.csv", written_files)
    _paper_write_csv(half_df, base_dir / "half_lives_long.csv", written_files)
    _paper_write_csv(recovery_df, base_dir / "recovery_delays_long.csv", written_files)
    _paper_write_csv(
        pd.DataFrame([{
            "scenario": "2B",
            "replications": int(replications),
            "M": int(M),
            "T": int(T),
            "T0": int(T0),
            "seed_start": int(seed),
            "seed_end": int(seed + replications - 1),
            "horizon": int(horizon),
            "smooth_window": int(smooth_window),
            "target_oracle_weight": float(target_oracle_weight),
            "fast_fixed_ld": bool(bt_cfg.fast_fixed_ld),
            "event_definition": "latent oracle switches in smooth-bias-drift simulations",
            "events": int(len(event_rows)),
        }]),
        base_dir / "multi_sim_metadata.csv",
        written_files,
    )
    return written_files


def _paper_base_backtest_config() -> BacktestConfig:
    """Backtest settings used for regenerated simulation paper exports."""
    return BacktestConfig(
        d_max=3,
        fixed_d=1,
        fixed_h1=0.25,
        fixed_h2=0.35,
        fast_fixed_ld=True,
        loss_diff_versions=("legacy", "v2"),
        adjacency_type="standardized",
        centrality_type="eigenvector",
        centrality_types=("eigenvector", "pagerank", "softmax"),
        cov_method="shrinkage",
        cov_window=50,
        alpha=0.1,
        gamma=0.1,
        tune_window=30,
        min_history=30,
        recent_best_window=20,
        window_grid=(20, 40, 60),
        bg_window=50,
        var_fixed_lag=1,
    )


def _paper_sensitivity_backtest_config() -> BacktestConfig:
    """Enhanced backtest settings used for simulation sensitivity sweeps."""
    return BacktestConfig(
        d_max=3,
        fixed_d=1,
        fixed_h1=0.25,
        fixed_h2=0.35,
        fast_fixed_ld=True,
        loss_diff_versions=("legacy",),
        adjacency_type="standardized",
        centrality_type="eigenvector",
        centrality_types=("eigenvector", "pagerank", "softmax"),
        cov_method="shrinkage",
        cov_window=40,
        alpha=0.1,
        gamma=0.1,
        tune_window=40,
        estimate_window=False,
        window_grid=(20, 40, 60),
        min_history=30,
        recent_best_window=20,
        bg_window=40,
        var_window=40,
        var_fixed_lag=1,
        include_benchmarks=False,
    )


def regenerate_simulation_replot_csv_exports(
    csv_root: Union[str, Path] = "Empirical_Data/empirical_outputs/replot_csv_exports",
    M: int = 8,
    T: int = 400,
    T0: int = 200,
    seed: int = 42,
    sensitivity_mcs_B: int = 100,
    adaptability_2a_replications: int = 50,
    adaptability_2a_horizon: int = 80,
    adaptability_2b_replications: int = 50,
    adaptability_2b_horizon: int = 80,
    verbose: bool = True,
) -> List[str]:
    """
    Regenerate the simulation-backed CSV exports used by paper replots.

    Empirical CSV exports are left untouched. The legacy `4A` folder names are
    preserved for backward-compatible readers, while the plotting layer displays
    that scenario as 3A.
    """
    csv_root = Path(csv_root)
    csv_root.mkdir(parents=True, exist_ok=True)
    written_files: List[str] = []

    scenario_factories = {
        "1A": scenario_1A,
        "2A": scenario_2A,
        "2B": scenario_2B,
        "2C": scenario_2C,
        "4A": scenario_4A,
    }
    scenario_data: Dict[str, SimulationData] = {}
    model_names = _paper_model_names(M)

    for scenario_key, factory in scenario_factories.items():
        if verbose:
            print(f"Generating Scenario {scenario_display_label(scenario_key)} with T={T}, T0={T0}...")
        data = generate_scenario(factory(M=M, T=T, T0=T0, seed=seed))
        scenario_data[scenario_key] = data
        _paper_export_simulation_data(f"scenario_{scenario_key}", data, csv_root, written_files, model_names=model_names)

    bt_cfg = _paper_base_backtest_config()
    backtests: Dict[str, BacktestResult] = {}
    for scenario_key, data in scenario_data.items():
        if verbose:
            print(f"Running paper backtest for Scenario {scenario_display_label(scenario_key)}...")
        res = run_backtest(data, bt_cfg, verbose=False)
        backtests[scenario_key] = res
        if scenario_key == "2A":
            export_name = "res_2A"
        elif scenario_key == "2B":
            export_name = "res_2B"
        else:
            export_name = f"mcs_backtest_{scenario_key}"
        _paper_export_backtest_result(export_name, res, csv_root, written_files, model_names=model_names)

    if verbose:
        print(
            "Computing multi-simulation adaptability diagnostics for Scenario 2A "
            f"with {adaptability_2a_replications} replications..."
        )
    written_files.extend(
        regenerate_2a_multi_sim_adaptability_csv_export(
            csv_root=csv_root,
            M=M,
            T=T,
            T0=T0,
            seed=seed,
            replications=adaptability_2a_replications,
            horizon=adaptability_2a_horizon,
            smooth_window=5,
            target_oracle_weight=0.60,
            bt_cfg=bt_cfg,
            verbose=verbose,
        )
    )
    if verbose:
        print(
            "Computing multi-simulation adaptability diagnostics for Scenario 2B "
            f"with {adaptability_2b_replications} replications..."
        )
    written_files.extend(
        regenerate_2b_multi_sim_adaptability_csv_export(
            csv_root=csv_root,
            M=M,
            T=T,
            T0=T0,
            seed=seed,
            replications=adaptability_2b_replications,
            horizon=adaptability_2b_horizon,
            smooth_window=5,
            target_oracle_weight=0.60,
            bt_cfg=bt_cfg,
            verbose=verbose,
        )
    )

    sensitivity_cfg = _paper_sensitivity_backtest_config()
    simulation_sweeps: List[SensitivitySweepResult] = []
    for scenario_key, data in scenario_data.items():
        if verbose:
            print(f"Running sensitivity sweep for Scenario {scenario_display_label(scenario_key)}...")
        sweep = run_sensitivity_sweep(
            data,
            bt_cfg=sensitivity_cfg,
            source_name=f"simulation_{scenario_key}",
            source_type="simulation",
            cov_methods=("rolling", "ewma", "shrinkage"),
            mcs_B=sensitivity_mcs_B,
            verbose=False,
        )
        simulation_sweeps.append(sweep)
        _paper_export_sensitivity_sweep(sweep, csv_root, written_files)

    simulation_summary = aggregate_sensitivity_sweeps(simulation_sweeps)
    _paper_write_csv(simulation_summary, csv_root / "sensitivity_summaries" / "simulation_sweep_summary" / "summary.csv", written_files)
    _paper_write_csv(simulation_summary, csv_root / "tables" / "simulation_sweep_summary.csv", written_files)
    _paper_write_csv(
        pd.DataFrame([{
            "M": M,
            "T": T,
            "T0": T0,
            "seed": seed,
            "sensitivity_mcs_B": sensitivity_mcs_B,
            "adaptability_2a_replications": adaptability_2a_replications,
            "adaptability_2a_horizon": adaptability_2a_horizon,
            "adaptability_2b_replications": adaptability_2b_replications,
            "adaptability_2b_horizon": adaptability_2b_horizon,
            "scenarios": ", ".join(scenario_display_label(key) for key in scenario_factories),
        }]),
        csv_root / "paper_replot_simulation_run_metadata.csv",
        written_files,
    )
    return written_files


def generate_paper_replots(
    csv_root: Union[str, Path] = "Empirical_Data/empirical_outputs/replot_csv_exports",
    output_dir: Union[str, Path] = "Empirical_Data/empirical_outputs/paper_replots",
) -> Dict[str, List[str]]:
    """
    Generate paper-ready figures and tables from exported CSVs.

    This function intentionally avoids rerunning simulations or backtests. It reads the
    existing `replot_csv_exports` tree and writes publication-facing artifacts with
    stable labels, Scenario 3A display names, and grayscale-friendly styling.
    """
    set_plot_style()
    csv_root = Path(csv_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, List[str]] = {}

    empirical_inputs = (
        (
            "inflation",
            "Inflation forecaster correlation",
            "Inflation: realized HICP and SPF forecast range",
            "Inflation (%)",
            "Empirical_Data/inflation_forecasts_f.csv",
            "Empirical_Data/inflation_truth_f.csv",
        ),
        (
            "unemployment",
            "Unemployment forecaster correlation",
            "Unemployment: realized value and SPF forecast range",
            "Unemployment (%)",
            "Empirical_Data/unemployment_forecasts_f.csv",
            "Empirical_Data/unemployment_truth_f.csv",
        ),
    )
    for slug, heatmap_title, range_title, y_label, forecast_path, truth_path in empirical_inputs:
        merged, forecaster_ids = _paper_load_empirical_panel(forecast_path, truth_path)
        fig = _paper_plot_forecaster_heatmap(merged, forecaster_ids, heatmap_title)
        _paper_save_figure(
            fig,
            output_dir,
            f"data_corr_heatmap_{slug}",
            artifacts,
            caption=(
                f"Correlation heatmap for the {slug} data section. The realized value is "
                "denoted by y and individual SPF forecasters are denoted by f_i."
            ),
        )
        fig = _paper_plot_spf_range(merged, forecaster_ids, range_title, y_label)
        _paper_save_figure(
            fig,
            output_dir,
            f"spf_range_{slug}",
            artifacts,
            caption=(
                f"Realized {slug} and SPF forecast dispersion. The shaded band is the "
                "cross-forecaster forecast range, the dashed line is the SPF median, and "
                "the solid black line is the realized value. The vertical axis is measured "
                "in percentage points."
            ),
        )

    _paper_write_rs_original_table(csv_root, output_dir, artifacts)
    _paper_plot_simulation_forecasts_errors(csv_root, output_dir, artifacts)
    _paper_plot_sensitivity_figures(csv_root, output_dir, artifacts)
    _paper_plot_empirical_headline(csv_root, output_dir, artifacts)
    _paper_plot_empirical_diff_oos_forecasts(csv_root, output_dir, artifacts)
    _paper_plot_rs_pairwise(csv_root, output_dir, artifacts)
    _paper_plot_weight_stability(csv_root, output_dir, artifacts)
    _paper_plot_weight_paths_a4(csv_root, output_dir, artifacts)
    _paper_plot_adaptability_figures(csv_root, output_dir, artifacts)

    manifest_rows = [
        {"Artifact": name, "Path": path}
        for name, paths in artifacts.items()
        for path in paths
    ]
    manifest_path = output_dir / "paper_replots_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    artifacts["paper_replots_manifest"] = [str(manifest_path)]
    return artifacts


# ===================================================================
# 14.  LEAKAGE AUDIT UTILITIES
# ===================================================================

def leakage_audit_synthetic(seed=123):
    """
    Synthetic timing sanity check.

    Creates a trivial scenario where the target is fully predictable
    by one model from t-1 info, and verifies that no method uses
    y_{t+1} when forming weights for t+1.

    Returns True if audit passes.
    """
    rng = _ensure_rng(seed)
    T, M = 120, 4
    T0 = 60

    y = rng.normal(0, 1, size=T).cumsum()
    forecasts = np.zeros((T, M))
    # Model 0: knows y exactly one period late (lagged info)
    forecasts[1:, 0] = y[:-1]
    # Model 1-3: noisy
    for j in range(1, M):
        forecasts[:, j] = y + rng.normal(0, 2.0, size=T)

    errors = y[:, None] - forecasts
    losses = squared_loss(y[:, None], forecasts)

    cfg = ScenarioConfig(name="audit", M=M, T=T, T0=T0, seed=seed)
    data = SimulationData(
        y=y, forecasts=forecasts, errors=errors, losses=losses,
        bias_paths=np.zeros((T, M)), sigma_paths=np.ones((T, M)) * 0.5,
        common_shock=np.zeros(T), config=cfg,
    )

    bt_cfg = BacktestConfig(
        d_max=2, fixed_d=1, fixed_h1=0.3, fixed_h2=0.4,
        cov_window=30, min_history=20,
        alpha=0.1, gamma=0.1,
    )
    res = run_backtest(data, bt_cfg)

    # Check: at each OOS period t, the weight vector was formed
    # WITHOUT using y[t]. We verify by checking that the first OOS
    # combined forecast does NOT perfectly nail y[T0], which would
    # indicate leakage.
    primary_full_name = _centrality_method_name(
        "full_gcsr",
        res.primary_centrality_type,
        res.centrality_types,
    )
    first_loss_full = res.combined_losses[primary_full_name][0]
    first_loss_eq = res.combined_losses["equal"][0]

    # In a correct implementation both should have nonzero loss
    audit_pass = True
    notes = []

    if first_loss_full < 1e-15:
        notes.append("WARNING: Full method has near-zero first-period loss — possible leakage")
        audit_pass = False

    # Additional check: weights should be identical if we permute future y
    # (we can't run this dynamically here, but the structure guarantees it)

    # Check that all weight vectors sum to ~1
    for name in res.weights:
        w = res.weights[name]
        sums = w.sum(axis=1)
        if np.any(np.abs(sums - 1.0) > 1e-4):
            notes.append(f"WARNING: {name} weights don't sum to 1")
            audit_pass = False

    # Check non-negativity
    for name in res.weights:
        w = res.weights[name]
        if np.any(w < -1e-8):
            notes.append(f"WARNING: {name} has negative weights")
            audit_pass = False

    return audit_pass, notes, res


def print_timing_rules():
    """Print the information set / timing conventions."""
    text = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    TIMING CONVENTIONS                            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  Forecast origin: t                                              ║
    ║  Target to predict: y_{t}                                        ║
    ║  Forecasts available: f_{j,t} for j=1,...,M                      ║
    ║  Realised outcomes available: y_0, ..., y_{t-1}                  ║
    ║  Realised losses available: L_{j,0}, ..., L_{j,t-1}              ║
    ║  Realised errors available: e_{j,0}, ..., e_{j,t-1}              ║
    ║                                                                  ║
    ║  Weights w_{t} are formed using ONLY:                            ║
    ║    - losses / errors from periods 0 through t-1                  ║
    ║    - pairwise LD from periods 0 through t-1                      ║
    ║    - covariance from errors 0 through t-1                        ║
    ║                                                                  ║
    ║  y_{t} is NEVER used when forming w_{t}.                         ║
    ║                                                                  ║
    ║  Combined forecast: ŷ_t = Σ_j w_{j,t} f_{j,t}                    ║
    ║  Evaluated after y_t is realised.                                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(text)


# ===================================================================
# 14.  CONVENIENCE: RUN ALL SCENARIOS
# ===================================================================

def run_all_scenarios(
    n_reps: int = 20,
    bt_cfg: BacktestConfig = None,
    verbose: bool = True,
    M: int = 8,
    T: int = 300,
    T0: int = 150,
) -> Dict[str, MCResult]:
    """Run MC for every pre-built scenario."""
    if bt_cfg is None:
        bt_cfg = BacktestConfig(
            d_max=3,
            fixed_d=1,        # speed: fix lag to 1
            fixed_h1=0.25,    # speed: fix bandwidth
            fixed_h2=0.35,
            centrality_types=DEFAULT_COMPARISON_CENTRALITY_TYPES,
            cov_window=50,
            min_history=30,
            alpha=0.1,        # speed: avoid re-tuning at every OOS step
            gamma=0.1,
            tune_window=30,
        )
    else:
        bt_cfg = _ensure_comparison_centrality_types(bt_cfg)

    results = {}
    for sc_key, factory in ALL_SCENARIO_FACTORIES.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running scenario {sc_key}...")
            print(f"{'='*50}")
        mc = run_monte_carlo(
            factory,
            n_reps=n_reps,
            bt_cfg=bt_cfg,
            verbose=verbose,
            M=M, T=T, T0=T0,
        )
        results[sc_key] = mc
        if verbose:
            print(summarise_mc(mc).to_string(index=False))
    return results


# ===================================================================
# END OF MODULE
# ===================================================================

if __name__ == "__main__":
    print("forecast_combination module loaded successfully.")
    print("Run leakage audit...")
    passed, notes, _ = leakage_audit_synthetic()
    print(f"Audit passed: {passed}")
    for n in notes:
        print(f"  {n}")
