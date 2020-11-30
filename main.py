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


def newtons(F, J, x0, tol=1e-6, omega=1., solver="Direct"):
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
    A = J(F, x)

    if solver == 'GMRES':
        while (la.norm(F(x)) > tol):
            M = lin_ssor_precon(A, omega)
            # M = ssor_precon(F, omega, x)
            s, _ = spla.gmres(J(F, x), -F(x), M=M, callback=con_tracker)
            x = x + s
    else:
        while (la.norm(F(x)) > tol):
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
    omega = 1.0
    gmres_counter = ConvergenceTracker(la.norm(F(x)))

    while (la.norm(F(x)) > tol):
        # use solver with/without preconditioner
        def mv(v):
            return (F(x + slen * v) - F(x)) / slen

        J = spla.LinearOperator((n, n), matvec=mv)
        # TODO: Use Krylov method here
        M = ssor_precon(F, omega, x)
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


def ssor_precon(F, omega, x):
    """ Construct an ssor preconditioner for the cases when we don't have a jacobian
    matrix"""

    tol = 1e-10
    eps = 1e-4
    iter_out_limit = 1
    iter_in_limit = 1
    n = len(x)

    def Fw(w):
        return fd_direct_deriv(F, x, w)

    def mv(v):
        err = 1.
        n_out_iter = 0
        w_old = np.zeros(n)
        w = np.zeros(n)

        while (err > tol and n_out_iter < iter_out_limit):
            w_old = np.copy(w)

            for ii in range(2 * n):
                i = ii if ii < n else 2 * n - ii - 1

                w_i_old = w[i]
                inerr = 1.
                n_in_iter = 0

                Fwi = (F(x + eps * w)[i] - F(x)[i]) / eps - v[i]
                eps_e_i = np.append(np.zeros(i),
                                    np.append(eps, np.zeros(n - i - 1)))

                # Try Newton's method for root finding

                while (abs(Fwi) > tol and n_in_iter < iter_in_limit):
                    # Fwi = Fw(w)[i] - v[i]

                    # dfidwi = (Fw(w + pert)[i] - v[i] - Fwi) / eps
                    Fwi_fwd = (F(x + eps *
                                 (w + eps_e_i))[i] - F(x)[i]) / eps - v[i]
                    Fwi_bwd = (F(x + eps *
                                 (w - eps_e_i))[i] - F(x)[i]) / eps - v[i]
                    dFwi = (Fwi_fwd - Fwi_bwd) / (2 * eps)

                    # w[i] = w[i] - Fwi / stabilise(dfidwi)
                    w[i] = w[i] - Fwi / dFwi

                    # inerr = abs(Fwi / stabilise(dfidwi))
                    Fwi = (F(x + eps * w)[i] - F(x)[i]) / eps - v[i]
                    n_in_iter += 1

                # only update ith coordinate
                w[i] = (1 - omega) * w_i_old + omega * w[i]

            err = la.norm(w - w_old) / stabilise(la.norm(w))
            n_out_iter += 1

        return w

    return spla.LinearOperator((n, n), matvec=mv)


def lin_ssor_precon(A, omega):
    """ Construct an linear ssor preconditioner for the cases when we have a jacobian
    matrix"""

    #get the D, L and U parts of J
    n = A.shape[0]
    L = sp.tril(A, k=-1)
    D = sp.diags(A.diagonal())
    D_inv = sp.diags(1 / A.diagonal())
    U = sp.triu(A, k=1)

    M_1 = D_inv @ (D - omega * U)
    M_2 = omega * (2 - omega) * (D - omega * L)

    def mv(v):
        intermediate = spla.spsolve_triangular(M_2, v)
        w = spla.spsolve_triangular(M_1, intermediate, lower=False)
        # w = M_2 @ M_1 @ v

        return w

    return spla.LinearOperator((n, n), matvec=mv)


def stabilise(a, small=1e-8):
    return a if abs(a) > small else small


def main():
    n = 100
    # utrue = np.sin(np.arange(n))
    utrue = np.ones(n)
    R = nonlinear_PDE(utrue)

    # Objective function
    def F(v):
        return nonlinear_PDE(v) - R

    # Starting guess
    seed_num = np.random.randint(0, 1000)
    print("== Seed {} ==\n".format(seed_num))
    np.random.seed(seed_num)
    u0 = np.random.normal(size=n)

    # test clustering
    # omega = 1
    # J = exact_jacobian(None, u0)
    # L = sp.tril(J, k=-1)
    # D = sp.diags(J.diagonal())
    # D_inv = sp.diags(1/J.diagonal())
    # U = sp.triu(J, k=1)
    # M = omega * (2 - omega) * (D - omega * L) @ D_inv @ (D - omega * U)
    # M = M.todense()
    # M = la.inv(M)

    # print("Eigenvalues w/o preconditioner =", la.eig(J.toarray())[0])
    # print("Eigenvalues w/  preconditioner =", la.eig(M@J.toarray())[0])

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
                                omega=1,
                                solver='GMRES')
    print("\n>> Exact Jacobian: GMRES")
    print("Converged in {} iterations".format(con_tracker_ex.niters()))
    print("Error residual={:.2e}".format(la.norm(u - utrue) / la.norm(utrue)))

    # Finite difference Jacobian direct solve
    u, con_tracker_fd = newtons(F, fd_jacobian, u0, tol=1e-6)
    print("\n>> FD Jacobian: Direct solve")
    print("Converged in {} iterations".format(con_tracker_fd.niters()))
    print("Error residual={:.2e}".format(la.norm(u - utrue) / la.norm(utrue)))

    # Finite difference Jacobian GMRES
    u, con_tracker_fd = newtons(F,
                                fd_jacobian,
                                u0,
                                tol=1e-6,
                                omega=1,
                                solver='GMRES')
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
