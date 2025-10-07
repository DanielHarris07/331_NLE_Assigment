from core import *
import numpy as np


def bisection(f, xl, xr, max_iter, tol):
    '''
    Nonlinear equation root finding by the bisection method.

    Inputs:
        f        : nonlinear function
        xl, xr   : initial root bracket
        max_iter : maximum number of iterations performed
        tol      : numerical tolerance used to check for root

    Outputs:
        x        : one-dimensional array containing estimates of root
        i        : number of iterations (number of bisections)
        e        : ExitFlag (enumeration)
    '''
    x=[xl]
    fxl=f(xl)
    
    # check the left side of interval for convergence
    if abs(fxl)<=tol:
        return x,0,ExitFlag.Converged

    x = [xl, xr]
    fxr = f(xr)
    
    # check the right side of interval for convergence / check that the l/r function values have different signs.
    if abs(fxr)<=tol:
        return x,0,ExitFlag.Converged
    elif fxl*fxr>0:
        return x,0,ExitFlag.NoRoot

    i=1
    while True:
        i+=1
        
        # compute new estimate appending to x, and evaluate the function value fx
        xm = (xl + xr) / 2
        fxm=f(xm)
        x.append(xm)
    
        if abs(fxm)<=tol:
            return x,i-1,ExitFlag.Converged
        elif i-1 == max_iter:
            return x,i-1,ExitFlag.MaxIterations

        # update interval
        if fxl*fxm > 0:
            xl = xm
            fxl = fxm
        else:
            xr = xm
            fxr = fxm

    
def secant(f, x0, x1, max_iter, tol):
    '''
    Nonlinear equation root finding by the secant method.

    Inputs:
        f        : nonlinear function
        x0, x1   : initial root bracket
        max_iter : maximum number of iterations performed
        tol      : numerical tolerance used to check for root

    Outputs:
        x        : one-dimensional array containing estimates of root
        i        : number of iterations (number of times a new point is attempted to be estimated)
        e        : ExitFlag (enumeration)
    '''
    x=[x0]
    fx0=f(x0)
    
    # check the left side of interval for convergence
    if abs(fx0)<=tol:
        return x,0,ExitFlag.Converged

    x = [x0,x1]
    fx1 = f(x1)
    
    # check the right side of interval for convergence / check that the l/r function values have different signs.
    if abs(fx1)<=tol:
        return x,0,ExitFlag.Converged
    elif fx0*fx1>0:
        return x,0,ExitFlag.NoRoot
    pass

    i = 0
    while True:
        i += 1
        # find gradient of line between bounds, and thus the intercept point
        grad = (fx1 - fx0) / (x1 - x0)
        if grad == 0:
            return x, i, ExitFlag.DivideByZero
        x_int = -(fx0 / grad) +x0
        x.append(x_int)

        # check for converged
        fx_int = f(x_int)
        if abs(fx_int)<=tol:
            return x,i,ExitFlag.Converged
        elif i == max_iter:
            return x,i,ExitFlag.MaxIterations
        x0 = x1
        fx0 = fx1
        x1 = x_int
        fx1 = fx_int
    

def regula_falsi(f, xl, xr, max_iter, tol):
    '''
    Nonlinear equation root finding by Regula Falsi method

    Inputs:
        f        : nonlinear function
        g        : nonlinear function derivative (gradient)
        x0       : initial root estimate
        max_iter : maximum number of iterations performed
        tol      : numerical tolerance used to check for root

    Outputs:
        x        : one-dimensional array containing estimates of root
        i        : number of iterations (number of times a new point is attempted to be estimated)
        e        : ExitFlag (enumeration)
    '''
    x=[xl]
    fxl=f(xl)
    
    # check the left side of interval for convergence
    if abs(fxl)<=tol:
        return x,0,ExitFlag.Converged

    x = [xl,xr]
    
    fxr = f(xr)
    
    # check the right side of interval for convergence / check that the l/r function values have different signs.
    if abs(fxr)<=tol:
        return x,0,ExitFlag.Converged
    elif fxl*fxr>0:
        return x,0,ExitFlag.NoRoot
    pass

    i = 0
    while True:
        i += 1
        # find gradient of line between bounds, and thus the intercept point
        grad = (fxr - fxl) / (xr - xl)
        if grad == 0:
            return x, i, ExitFlag.DivideByZero
        x_int = -(fxl / grad) +xl
        x.append(x_int)

        # check for converged
        fx_int = f(x_int)
        if abs(fx_int)<=tol:
            return x,i,ExitFlag.Converged
        elif i == max_iter:
            return x,i,ExitFlag.MaxIterations
        
        # update interval
        if fxr*fx_int > 0:
            xr = x_int
            fxr = fx_int
        else:
            xl = x_int
            fxl = fx_int


def newton(f, g, x0, max_iter, tol):
    '''
    Nonlinear equation root finding by Newton's method.

    Inputs:
        f        : nonlinear function
        g        : nonlinear function derivative (gradient)
        x0       : initial root estimate
        max_iter : maximum number of iterations performed
        tol      : numerical tolerance used to check for root

    Outputs:
        x        : one-dimensional array containing estimates of root
        i        : number of iterations (number of times a new point is attempted to be estimated)
        e        : ExitFlag (enumeration)
    '''
    x=[x0]
    fx0=f(x0)
    
    # check initial guess for convergence
    if abs(fx0)<=tol:
        return x,0,ExitFlag.Converged
    
    i = 0
    while True:
        i += 1
        
        # compute gradient, check gradient is not zero
        gx0 = g(x0)
        if gx0 == 0:
            return x, i, ExitFlag.DivideByZero
        
        # calculate new intercept point
        x0 = -(fx0 / gx0) + x0
        x.append(x0)
        fx0 = f(x0)

        # check for converged/max_iter
        if abs(fx0)<=tol:
            return x,i,ExitFlag.Converged
        elif i == max_iter:
            return x,i,ExitFlag.MaxIterations
    

def combined(f, g, xl, xr, max_iter, tol):
    '''
    Nonlinear equation root finding by the combined bisection/Newton's method.

    Inputs:
        f        : nonlinear function
        g        : nonlinear function derivative (gradient)
        xl, xr   : initial root bracket
        max_iter : maximum number of iterations performed
        tol      : numerical tolerance used to check for root

    Outputs:
        x        : one-dimensional array containing estimates of root
        i        : number of iterations (number of times a new point is attempted to be estimated)
        e        : ExitFlag (enumeration)
    '''
    x=[xl]
    fxl=f(xl)
    
    # check LHS for convergence
    if abs(fxl)<=tol:
        return x,0,ExitFlag.Converged
    
    x = [xl,xr]
    
    fxr = f(xr)
    
    # check the RHS of interval for convergence / check that the l/r function values have different signs.
    if abs(fxr)<=tol:
        return x,0,ExitFlag.Converged
    elif fxl*fxr>0:
        return x,0,ExitFlag.NoRoot
    
    i = 0
    while True:
        i += 1
        
        # compute gradient, check gradient is not zero
        gxl = g(xl)
        if gxl == 0:
            return x, i, ExitFlag.DivideByZero
        
        # calculate new intercept point and check if outside [xl, xr]
        x_int = -(fxl / gxl) + xl
        if (x_int < xl) or (x_int > xr):
            # use bisection instead this iteration
            x_int = (xl + xr) / 2
            fx_int=f(x_int)
            x.append(x_int)
        else:
            # otherwise stick with Netwon estimate
            x.append(x_int)
            fx_int = f(x_int)

        # check for converged/max_iter
        if abs(fx_int)<=tol:
            return x,i,ExitFlag.Converged
        elif i == max_iter:
            return x,i,ExitFlag.MaxIterations
        
        # update bounds
        if fxl*fx_int > 0:
            xl = x_int
            fxl = fx_int
        else:
            xr = x_int
            fxr = fx_int


def newton_damped(f, g, x0, max_iter, tol, beta):
    '''
    Nonlinear equation root finding by Newton's method with damping.

    Inputs:
        f        : nonlinear function
        g        : nonlinear function derivative (gradient)
        x0       : initial root estimate
        max_iter : maximum number of iterations performed
        tol      : numerical tolerance used to check for root
        beta     : damping factor

    Outputs:
        x        : one-dimensional array containing estimates of root
        i        : number of iterations (number of times a new point is attempted to be estimated)
        e        : ExitFlag (enumeration)
    '''
    x=[x0]
    fx0=f(x0)
    
    # check initial guess for convergence
    if abs(fx0)<=tol:
        return x,0,ExitFlag.Converged
    
    i = 0
    while True:
        i += 1
        
        # compute gradient, check gradient is not zero
        gx0 = g(x0)
        if gx0 == 0:
            return x, i, ExitFlag.DivideByZero
        
        # calculate new intercept point
        x1 = -(fx0 / gx0) + x0

        # calculate damping factor
        damping = 1 / (1 + beta*np.abs(x1 - x0))

        x0 = x0 + (x1-x0)*damping 
        x.append(x0)
        fx0 = f(x0)

        # check for converged/max_iter
        if abs(fx0)<=tol: 
            return x,i,ExitFlag.Converged
        elif i == max_iter:
            return x,i,ExitFlag.MaxIterations
