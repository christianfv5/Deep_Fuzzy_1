"""
cmeans.py : Fuzzy C-means clustering algorithm.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import fractional_matrix_power,pinv
import tensorflow as tf

# def tf_pinv(A, b, reltol=1e-6):
#   # Compute the SVD of the input matrix A
#   s, u, v = tf.svd(A)
#
#   # Invert s, clear entries lower than reltol*s[0].
#   atol = tf.reduce_max(s) * reltol
#   s = tf.boolean_mask(s, s > atol)
#   s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))
#
#   # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
#   return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))


def _cmeans0(data, u_old, c, m,rho=1):
    """
    Single step in generic fuzzy c-means clustering algorithm.

    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.

    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T


    cntr = um.dot(data) / (np.ones((data.shape[1],
                                    1)).dot(np.atleast_2d(um.sum(axis=1))).T)

    # print(data.shape)
    # print(um.shape)
    # print(cntr.shape)

    sumf = np.atleast_2d(np.sum(um,axis=1))
    X1 = np.ones((data.shape[0], 1))
    d = np.zeros((c,data.shape[0]))
    # print(X1.shape)
    # print(np.atleast_2d(cntr[1,:]).shape)
    # print(sumf[0])
    # print(sumf.shape)
    for cluster_counter in range(c):
        xv = data - X1.dot(np.atleast_2d(cntr[cluster_counter,:]))
        # print("XV=",xv.shape)
        C = np.ones((data.shape[1], 1)).dot(np.atleast_2d(um[cluster_counter,:]))
        # print("C=", C.shape)
        B = np.multiply(C,xv.T).dot(xv)
        # print("B = ",B.shape)
        A = B / sumf[0,cluster_counter]
        M1 = 1
        M1 = np.power(np.linalg.det(A),1.0/int(data.shape[1]))
        # if(M1 == 0.0):
        #     print("oomadam")
        #     eig_values = list(np.linalg.eig(A))
        #     pseudo_determinent = -10000
        #     print("eig_values",eig_values)
        #     idx = (eig_values > 1)
        #     print(idx)
        #     pseudo_determinent = np.product(eig_values[eig_values > 1])
        #     print("pseudo_determinent",pseudo_determinent)
        #     M1 = np.power(pseudo_determinent, 1.0 / int(data.shape[1]))
        #     print("M1=",M1)

        # M1 = np.linalg.det(fractional_matrix_power(A,1.0/int(data.shape[1])))
        # print("int(data.shape[1]=",int(data.shape[1]))
        # print("M1=",M1)
        M2 =pinv(A)
        # M2 = tf.py_func(np.linalg.pinv, [A], tf.float32)
        M = rho * M1 * M2
        # print("M.shape = ",M.shape)
        # print("M = ", M.shape)
        # M = (1/det(pinv(A))/rho(j))^(1/n)*pinv(A);
        K = np.atleast_2d(np.sum(np.multiply(xv.dot(M), xv),axis=1))
        # print("K = ", K.shape)
        # print("d = ", d.shape)
        d[cluster_counter,:] =K
    #     for j = 1: c,
    #     xv = X - X1 * v(j,:);
    #     A = ones(n, 1) * fm(:, j)'.*xv' * xv / sumf(j);
    #     % M = (1 / det(pinv(A)) / rho(j)) ^ (1 / n) * pinv(A);
    #     M = rho(j) * det(A ^ (1 / n)) * pinv(A);
    #     d(:, j) = sum((xv * M. * xv), 2);
    # end
    #
    # distout = sqrt(d);
    # J(iter) = sum(sum(f0. * d));
    # d = (d + 1e-10). ^ (-1 / (m - 1));
    # f0 = (d. / (sum(d, 2) * ones(1, c)));

    # print("D1",d.shape)
    # d = _distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)
    # print("D2",d.shape)

    jm = (um * d ** 2).sum()

    u = d ** (- 1. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d


def _distance(data, centers):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.

    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers).T


def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def cmeans(data, c, m, error, maxiter,rho=1, init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].

    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns
    -------
    cntr : 2d array, size (S, c)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.


    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.

    Fuzzy C-Means has a known problem with high dimensionality datasets, where
    the majority of cluster centers are pulled into the overall center of
    gravity. If you are clustering data with very high dimensionality and
    encounter this issue, another clustering method may be required. For more
    information and the theory behind this, see Winkler et al. [2]_.

    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.

    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high
           dimensional spaces. 2012. Contemporary Theory and Pragmatic
           Approaches in Fuzzy Computing Utilization, 1.
    """
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        # print("iter = ",p)
        u2 = u.copy()
        [cntr, u, Jjm, d] = _cmeans0(data, u2, c, m,rho)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            print("Error Threshold",p)
            break
    if(p == maxiter -1):
        print("Iter Threshold")
    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)
    # print("iter = ",p)
    return cntr, u, u0, d, jm, p, fpc


def cmeans_predict(test_data, cntr_trained, m, error, maxiter, rho=1,init=None,
                   seed=None):
    """
    Prediction of new data in given a trained fuzzy c-means framework [1].

    Parameters
    ----------
    test_data : 2d array, size (S, N)
        New, independent data set to be predicted based on trained c-means
        from ``cmeans``. N is the number of data sets; S is the number of
        features within each sample vector.
    cntr_trained : 2d array, size (S, c)
        Location of trained centers from prior training c-means.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns
    -------
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.

    Notes
    -----
    Ross et al. [1]_ did not include a prediction algorithm to go along with
    fuzzy c-means. This prediction algorithm works by repeating the clustering
    with fixed centers, then efficiently finds the fuzzy membership at all
    points.

    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.
    """
    c = cntr_trained.shape[0]

    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = test_data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [u, Jjm, d] = _cmeans_predict0(test_data, cntr_trained, u2, c, m,rho)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return u, u0, d, jm, p, fpc


def _cmeans_predict0(test_data, cntr, u_old, c, m,rho=1):
    """
    Single step in fuzzy c-means prediction algorithm. Clustering algorithm
    modified from Ross, Fuzzy Logic w/Engineering Applications (2010)
    p.352-353, equations 10.28 - 10.35, but this method to generate fuzzy
    predictions was independently derived by Josh Warner.

    Parameters inherited from cmeans()

    Very similar to initial clustering, except `cntr` is not updated, thus
    the new test data are forced into known (trained) clusters.
    """
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    test_data = test_data.T

    # For prediction, we do not recalculate cluster centers. The test_data is
    # forced to conform to the prior clustering.

    sumf = np.atleast_2d(np.sum(um, axis=1))
    X1 = np.ones((test_data.shape[0], 1))
    d = np.zeros((c, test_data.shape[0]))
    for cluster_counter in range(c):
        xv = test_data - X1.dot(np.atleast_2d(cntr[cluster_counter, :]))
        C = np.ones((test_data.shape[1], 1)).dot(np.atleast_2d(um[cluster_counter, :]))
        B = np.multiply(C, xv.T).dot(xv)
        A = B / sumf[0, cluster_counter]
        M1 = np.power(np.linalg.det(A), 1.0 / int(test_data.shape[1]))
        M2 = pinv(A)
        M = rho * M1 * M2
        K = np.atleast_2d(np.sum(np.multiply(xv.dot(M), xv), axis=1))
        d[cluster_counter, :] = K
    # print("D1", d.shape)
    # d = _distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)
    # print("D2", d.shape)


    # d = _distance(test_data, cntr)
    # d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = d ** (- 1. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return u, jm, d
