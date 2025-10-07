import numpy as np
from core import *
from jacobian import jacobian

def newton_multi(n, f, x0, h, max_iter, tol):
    x = [x0]
    n = len(x0)
    k = 0
    while True:
        xk = x[k]
        fk = f(xk)

        if max(fk) <= tol and max(-fk) <= tol:
            return x, ExitFlag.Converged
        
        # compute jacobian and check if invertible
        J = jacobian(f, xk, h, n)
        if np.linalg.det(J) == 0:
            return x, ExitFlag.DivideByZero
        
        # solve for the step and update x with new vector
        step = np.linalg.solve(J, -fk)
        x.append(x[k] + step)
        
        k+=1
        if k == max_iter:
            return x, ExitFlag.MaxIterations
