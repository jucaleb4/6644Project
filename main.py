import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def nonlinear_PDE(u):
    """ Centered finite difference of nonlinear PDE
            -u_xx + 2b(e^u)_x + ce^u = R(x) for x \in [0,1]
        with homogenous Dirichlet boundary conditions.
    """
    n = len(u)
    h = 1/(n+1)
    # TODO: allow @b,@c to be parameters
    b = c = 1

    expu = np.exp(u)

    r1 = (1/h**2)  * ( sp.diags([1,-2,1],[-1,0,1], shape=(n,n)).dot(u) )
    r2 = (1/(2*h)) * ( sp.diags([-1,0,1],[-1,0,1], shape=(n,n)).dot(expu) )
    r3 = expu

    r = -r1 + 2*b*r2 + c*r3
    return r

def newtons(F,J,x0,tol=1e-6):
    """ Newton's Method with exact or FD Jacobian

    - F : Function eval. Returns 1d np.ndarray
    - J : Jacobian eval. Returns 2d np.ndarray
    - x0 : Initial guess. 1D np.ndarray

    Returns: 
    - @x solution so that F(x)=0
    - @numiters, int
    - @score_hist, history of "F(x)". 1d np.ndarray
    """
    x = x0
    niters = 0
    score_hist = np.array([la.norm(F(x))])
    while(la.norm(F(x)) > tol):
        A = J(F,x) 
        # use solver with/without preconditioner
        # TODO: When will this be singular?
        s = spla.spsolve( A,-F(x) )
        x = x+s
        niters += 1
        score_hist = np.append(score_hist, la.norm(F(x)))
    return x,niters,score_hist

def newtons_derfree(F,x0,tol=1e-6):
    """ Newton's Method with Directional Derivative

    - F : Function eval. Returns 1d np.ndarray
    - x0 : Initial guess. 1D np.ndarray

    Returns: 
    - @x solution so that F(x)=0
    """
    x = x0
    slen = 1e-4
    while(la.norm(F(x)) > tol):
        # use solver with/without preconditioner
        def mv(v):
            return (F(x+slen*v)-F(x))/slen
        J = LinearOperator((n,n), matvec=mv)
        # TODO: Use Krylov method here
        s = ...
        x = x+s
    return x

def fd_jacobian(F,x):
    """ Finite Difference Approximation to Jacobian. 

    - F : Function eval. Returns 1d np.ndarray
    - x : Current solution 

    Returns: @J csr_matrix
    """
    n = len(x) 
    J = np.zeros((n,n))
    slen = 1e-4
    Fx = F(x)

    for i in range(n):
        e_i = np.append(np.zeros(i), np.append(1, np.zeros(n-i-1)))
        # TODO: Drop terms with small values to induce sparsity?
        J[:,i] = ( F(x+slen*e_i) - Fx )/slen

    return sp.csr_matrix(J)

def exact_jacobian(x,u):
    """ Exact Jacobian of finite difference PDE from above """
    n = len(u)
    h = 1/(n+1)
    # TODO: allow @b,@c to be parameters
    b = c = 1
    expu = np.exp(u)

    subdiag  = -(1/h**2)-2*b/(2*h)*expu[:n-1]
    maindiag =  2/(h**2)+c*expu
    supdiag  = -(1/h**2)+2*b/(2*h)*expu[1:]

    J = sp.diags(subdiag,-1)+sp.diags(maindiag,0)+sp.diags(supdiag,1)
    return J

def main():
    n = 10
    # utrue = np.sin(np.arange(n))
    utrue = np.ones(n)
    R = nonlinear_PDE(utrue)

    # Objective function
    def F(v): 
        return nonlinear_PDE(v) - R

    # Starting guess
    u0 = 5*np.random.normal(size=n)

    # Exact Jacobian 
    u,niters,scores = newtons(F,exact_jacobian,u0,tol=1e-6)
    print(">> Exact Jacobian")
    print("Converged in {} iterations".format(niters))
    # print("Previous scores={}".format(scores))
    print("Error residual={:.2e}".format(la.norm(u-utrue)/la.norm(utrue)))

    # Finite difference Jacobian
    u,niters,scores = newtons(F,fd_jacobian,u0,tol=1e-6)
    print("\n>> FD Jacobian")
    print("Converged in {} iterations".format(niters))
    print("Error residual={:.2e}".format(la.norm(u-utrue)/la.norm(utrue)))

if __name__=='__main__':
    main()
