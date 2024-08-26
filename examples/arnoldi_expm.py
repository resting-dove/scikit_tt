import scipy
import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.solvers import ode

from scipy.sparse.linalg import expm_multiply


def krylov_restarted2(operator: 'TT', initial_value: 'TT', dimension: int, step_size: float, restarts: int = 1,
                      threshold: float = 1e-12,
                      max_rank: int = 50, normalize: int = 0) -> 'TT':
    """
    Krylov method, see [1]_.

    Parameters
    ----------
    operator : TT
        TT operator

    initial_value : TT
        initial value for ODE

    dimension: int
        dimension of Krylov subspace, must be larger than 1

    step_size: float
        step size

    restarts: int
        number of restarts to perform at each step, default is 1, i.e. no restarts

    threshold : float, optional
        threshold for reduced SVD decompositions, default is 1e-12

    max_rank : int, optional
        maximum rank of the solution, default is 50

    normalize : {0, 1, 2}, optional
        no normalization if 0, otherwise the solution is normalized in terms of Manhattan or Euclidean norm in each step


    Returns
    -------
    TT
        approximated solution of the Schrödinger equation

    References
    ----------
    ..[1] S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck,
          C. Hubig, "Time-evolution methods for matrix-product states".
          Annals of Physics, 411, 167998, 2019
    """
    beta = initial_value.norm()
    # beta = initial_value.transpose(conjugate=True)@initial_value
    w = (1 / beta) * initial_value
    H_full = np.zeros([dimension * restarts + 2, dimension * restarts], dtype=complex)
    f = w.copy() * 0
    stopping_criterion = False
    breakdown = False
    m = dimension
    current_size = 0
    for k in range(restarts):
        if stopping_criterion:
            print(f"Stopping criteria reached in iteration {k}.")
            break
        # Arnoldi
        H_eff = np.zeros([dimension + 1, dimension], dtype=complex)
        krylov_tensors = [w]
        for i in range(dimension):
            w_tmp = operator @ krylov_tensors[i].copy()
            for j in range(i + 1):
                v_tmp = krylov_tensors[j].copy()
                ip = w_tmp.transpose(conjugate=True) @ v_tmp
                w_tmp = w_tmp - ip * v_tmp
                H_eff[j, i] += ip
            # eta = np.sqrt(w_tmp.transpose(conjugate=True) @ w_tmp)
            eta = w_tmp.norm()
            H_eff[i + 1, i] = eta
            if eta < 1e-10:
                breakdown = k
            w_tmp = (1 / eta) * w_tmp
            krylov_tensors.append(w_tmp)

        w = w_tmp.copy().ortho(threshold=threshold, max_rank=max_rank)
        if breakdown:
            stopping_criterion = True
            m = breakdown
        H_full[current_size: current_size + m + 1, current_size: current_size + m] = H_eff[:m + 1, :m]

        # compute time-evolved state

        e1 = np.zeros([current_size + m], dtype=complex)
        e1[0] = 1
        exp_col0 = expm_multiply(-1j * H_full[: current_size + m, : current_size + m] * step_size, e1)
        update = krylov_tensors[0] * exp_col0[current_size]
        for j in range(1, m):
            update = update + krylov_tensors[j] * exp_col0[current_size + j]
        update = update * beta
        if update.norm() < threshold:
            breakdown = True
        f = f + update
        current_size += m
    f = f.ortho(threshold=threshold, max_rank=max_rank)
    if normalize > 0:
        f = (1 / f.norm(p=normalize)) * f
    return f


def get_max_dimension(operator, initial_value, dimension):
    krylov_tensors = [(1 / initial_value.norm()) * initial_value]
    try:
        for i in range(1, dimension):
            w_tmp = operator @ krylov_tensors[-1].copy()
            v_tmp = w_tmp.copy()
            for j in range(i):
                v_tmp = v_tmp - (w_tmp.transpose(conjugate=True) @ krylov_tensors[j]) * krylov_tensors[j]
            krylov_tensors.append((1 / v_tmp.norm()) * v_tmp)
    except MemoryError as e:
        print(e)
    return i - 1


def get_exact_result(operator, v, t):
    def deriv(t, x):
        """
        x' = -iH @ x
        """
        return -1j * operator.matricize() @ x

    result = scipy.integrate.solve_ivp(deriv, [0, t], np.array(v.matricize(), dtype=complex))
    exact = result["y"][:, -1]
    return exact


if __name__ == '__main__':
    TT = tt.rand([2, 3, 4, 5, 4], [2, 3, 4, 5, 4], [1, 4, 3, 4, 4, 1])
    v = tt.rand([2, 3, 4, 5, 4], [1, 1, 1, 1, 1], TT.ranks)
    v = 1j * v
    v = 1j * v
    v = (1 / v.norm()) * v

    print(TT.full().nbytes, TT.matricize().shape, np.array([c.nbytes for c in TT.cores]).sum())

    # dimension = get_max_dimension(TT, v, dimension=8)
    dimension = 2
    print(f"dimension: {dimension}")

    exact = get_exact_result(TT, v, t=0.001)
    i_s = ode.implicit_euler(-1j * TT, v, v.ortho_right(), [0.001])
    ks = ode.krylov(TT, v, dimension, 0.001)
    exact_norm = np.linalg.norm(exact)
    print(f"Implicit Euler error: {np.linalg.norm(exact - i_s[0].matricize()) / exact_norm}")
    print(f"Krylov error: {np.linalg.norm(exact - ks.matricize()) / exact_norm}")

    for i in range(1, 5):
        ksr = krylov_restarted2(TT, v, dimension, 0.001, restarts=i)
        print(
            f"My Krylov restarted error with {i - 1} real restarts: {np.linalg.norm(exact - ksr.matricize()) / exact_norm}")
