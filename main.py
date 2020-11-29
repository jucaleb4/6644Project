import numpy as np
import numpy.linalg as la
#import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class ConvergenceTracker():
    """ Stores information such as iteration counts and solutions/residuals for
    analyzing convergence of iterative method."""
    def __init__(self, initial_res=None):
        self._iter_count = 0
        self._residuals = [initial_res]

    def __call__(self, residual=None):
        self._iter_count += 1
        self._residuals.append(residual)

    def niters(self):
        return self._iter_count


def nonlinear_PDE(u):
    """ Centered finite difference of nonlinear PDE
            -u_xx + 2b(e^u)_x + ce^u = R(x) for x \in [0,1]
        with homogenous Dirichlet boundary conditions.
    """
    n = len(u)
    h = 1 / (n + 1)
    # TODO: allow @b,@c to be parameters
    b = c = 1

    expu = np.exp(u)

    r1 = (1 / h**2) * (sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).dot(u))
    r2 = (1 /
          (2 * h)) * (sp.diags([-1, 0, 1], [-1, 0, 1], shape=(n, n)).dot(expu))
    r3 = expu

    r = -r1 + 2 * b * r2 + c * r3

    return r


def newtons(F, J, x0, tol=1e-6, solver="Direct"):
    """ Newton's Method with exact or FD Jacobian

    - F : Function eval. Returns 1d np.ndarray
    - J : Jacobian eval. Returns 2d np.ndarray
    - x0 : Initial guess. 1D np.ndarray
    - solver : Choose between solver types. Defaults to "Direct"

    Returns:
    - @x solution so that F(x)=0
    - @con_tracker ConvergencTracker object
    """
    x = x0
    con_tracker = ConvergenceTracker(la.norm(F(x)))

    if solver == 'GMRES':
        while (la.norm(F(x)) > tol):
            M = None
            s, _ = spla.gmres(J(F, x), -F(x), M=M, callback=con_tracker)
            x = x + s
    else:
        while (la.norm(F(x)) > tol):
            A = J(F, x)
            # use solver with/without preconditioner
            # TODO: When will this be singular?
            s = spla.spsolve(A, -F(x))
            x = x + s
            con_tracker(la.norm(F(x)))

    return x, con_tracker


def newtons_derfree(F, x0, tol=1e-6):
    """ Newton's Method with Directional Derivative

    - F : Function eval. Returns 1d np.ndarray
    - x0 : Initial guess. 1D np.ndarray

    Returns:
    - @x solution so that F(x)=0
    """
    x = x0
    n = len(x)
    slen = 1e-4
    omega = 1.
    gmres_counter = ConvergenceTracker(la.norm(F(x)))

    while (la.norm(F(x)) > tol):
        # use solver with/without preconditioner
        def mv(v):
            return (F(x + slen * v) - F(x)) / slen

        J = spla.LinearOperator((n, n), matvec=mv)
        # TODO: Use Krylov method here
        M = ssor_precon(F, omega, n, x)
        s, _ = spla.gmres(J, -F(x), M=M, callback=gmres_counter)
        x = x + s

    return x, gmres_counter


def fd_jacobian(F, x):
    """ Finite Difference Approximation to Jacobian.

    - F : Function eval. Returns 1d np.ndarray
    - x : Current solution

    Returns: @J csr_matrix
    """
    n = len(x)
    J = np.zeros((n, n))
    slen = 1e-4
    Fx = F(x)

    for i in range(n):
        e_i = np.append(np.zeros(i), np.append(1, np.zeros(n - i - 1)))
        # TODO: Drop terms with small values to induce sparsity?
        J[:, i] = (F(x + slen * e_i) - Fx) / slen

    return sp.csr_matrix(J)


def fd_direct_deriv(F, x, v):
    """Takes the directional derivative of F at x in direction of v"""

    h = 1e-6  #TODO: step size can be determined automagically

    return (F(x + h * v) - F(x)) / h


def exact_jacobian(x, u):
    """ Exact Jacobian of finite difference PDE from above """
    n = len(u)
    h = 1 / (n + 1)
    # TODO: allow @b,@c to be parameters
    b = c = 1
    expu = np.exp(u)

    subdiag = -(1 / h**2) - 2 * b / (2 * h) * expu[:n - 1]
    maindiag = 2 / (h**2) + c * expu
    supdiag = -(1 / h**2) + 2 * b / (2 * h) * expu[1:]

    J = sp.diags(subdiag, -1) + sp.diags(maindiag, 0) + sp.diags(supdiag, 1)

    return J


def ssor_precon(F, omega, n, x):
    """ Construct an ssor preconditioner for the cases when we have a jacobian
    matrix"""

    tol = 1e-2
    eps = 1e-4
    iter_limit = 1

    def Fw(w):
        return fd_direct_deriv(F, x, w)

    def mv(v):
        err = 1.
        n_iter = 0
        w_old = np.zeros(n)
        w = np.zeros(n)

        while err > tol and n_iter < iter_limit:
            w_old = np.copy(w)

            for i in range(n):
                inerr = 1.

                while inerr > tol:
                    pert = np.zeros(n)
                    pert[i] = eps
                    Fwi = Fw(w)[i] - v[i]
                    dfidwi = (Fw(w + pert)[i] - v[i] - Fwi) / eps
                    w[i] = w[i] - Fwi / stabilise(dfidwi)
                    inerr = abs(Fwi / stabilise(dfidwi))

                w[i] = (1 - omega) * w_old[i] + omega * w[i]
            err = la.norm(w - w_old) / stabilise(la.norm(w))
            n_iter += 1

        return w

    return spla.LinearOperator((n, n), matvec=mv)


def stabilise(a, small=1e-8):
    return a if abs(a) > small else small


def main():
    n = 10
    # utrue = np.sin(np.arange(n))
    utrue = np.ones(n)
    R = nonlinear_PDE(utrue)

    # Objective function
    def F(v):
        return nonlinear_PDE(v) - R

    # Starting guess
    u0 = np.random.normal(size=n)

    # Exact Jacobian direct solve
    u, con_tracker_ex = newtons(F, exact_jacobian, u0, tol=1e-6)
    print(">> Exact Jacobian: Direct solve")
    print("Converged in {} iterations".format(con_tracker_ex.niters()))
    print("Error residual={:.2e}".format(la.norm(u - utrue) / la.norm(utrue)))

    # Exact Jacobian GMRES
    u, con_tracker_ex = newtons(F,
                                exact_jacobian,
                                u0,
                                tol=1e-6,
                                solver='GMRES')
    print(">> Exact Jacobian: GMRES")
    print("Converged in {} iterations".format(con_tracker_ex.niters()))
    print("Error residual={:.2e}".format(la.norm(u - utrue) / la.norm(utrue)))

    # Finite difference Jacobian direct solve
    u, con_tracker_fd = newtons(F, fd_jacobian, u0, tol=1e-6)
    print("\n>> FD Jacobian: Direct solve")
    print("Converged in {} iterations".format(con_tracker_fd.niters()))
    print("Error residual={:.2e}".format(la.norm(u - utrue) / la.norm(utrue)))

    # Finite difference Jacobian GMRES
    u, con_tracker_fd = newtons(F, fd_jacobian, u0, tol=1e-6, solver='GMRES')
    print("\n>> FD Jacobian: GMRES")
    print("Converged in {} iterations".format(con_tracker_fd.niters()))
    print("Error residual={:.2e}".format(la.norm(u - utrue) / la.norm(utrue)))

    # Finite difference Jacobian derfree
    u, con_tracker_df = newtons_derfree(F, u0, tol=1e-6)
    print("\n>> Derfree Jacobian")
    print("Converged in {} iterations".format(con_tracker_df.niters()))
    print("Error residual={:.2e}".format(la.norm(u - utrue) / la.norm(utrue)))


if __name__ == '__main__':
    main()
