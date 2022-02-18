import numpy as np


def vip_scores(W, T, Q):
    """
    Calculates variable importance in projection (VIP) scores
    from PLS X weights, X scores, and y loadings.

    Parameters
    ----------
    W : numpy.ndarray of shape (n_features, n_components)
        X weights

    T : numpy.ndarray of shape (n_samples, n_components)
        X scores

    Q : numpy.ndarray of shape (n_targes, n_components)
        y loadings

    Returns
    -------
    numpy.ndarray of shape (n_features,)
        VIP scores.

    References
    ----------
    Farrés, M., Platikanov, S., Tsakovski, S., and Tauler, R. (2015)
    Comparison of the variable importance in projection (VIP)
    and of the selectivity ratio (SR) methods for variable selection and interpretation.
    J. Chemometrics, 29: 528– 536. doi: 10.1002/cem.2736
    
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.2736
    """
    W0 = W / np.sqrt(np.sum(W**2, 0))
    p, _ = W.shape
    sum_sq = np.sum(T**2, 0) * np.sum(Q**2, 0)
    vips = np.sqrt(p * np.sum(sum_sq * (W0**2), 1) / np.sum(sum_sq))
    return vips


def selectivity_ratio(X, B):
    """
    Calculates the selectivity ratio (SR) from PLS decomposition.

    Parameters
    ----------
    X : numpy.ndarray of size (n_samples, n_features)
        Feature matrix.

    B : numpy.ndarray of size (n_features, n_targets)
        PLS coefficients.

    Returns
    -------
    numpy.ndarray of size (n_features,)
        Selectivity ratio.

    References
    ----------
    Farrés, M., Platikanov, S., Tsakovski, S., and Tauler, R. (2015)
    Comparison of the variable importance in projection (VIP)
    and of the selectivity ratio (SR) methods for variable selection and interpretation.
    J. Chemometrics, 29: 528– 536. doi: 10.1002/cem.2736
    
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.2736
    """
    t_tp = X @ B / np.linalg.norm(B)
    p_tp = X.T @ t_tp / (t_tp.T @ t_tp)

    exp_var = (t_tp @ p_tp.T)**2
    exp_res = (X - exp_var)**2

    ss_expl = np.sum(exp_var, axis=0)
    ss_res = np.sum(exp_res, axis=0)
    return ss_expl / ss_res
