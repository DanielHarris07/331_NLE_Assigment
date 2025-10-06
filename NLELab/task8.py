import numpy as np
import matplotlib.pyplot as plt
from golden import golden
from brent import brent
from functions2 import f9, f10

# Task 8

Brent=True  # Set to False in order to not use Brent's method
Golden=True # Set to False in order to not use golden section search

f=[f9,f10] # Specify the functions to optimise and plot (must have length 2)
titles=['$f_{9}(x)=x^4-8x^3+24x^2-32x+17$','$f_{10}(x)=2-exp(-2x^2+8x-8)$'] # The expressions for the functions

# initialisation
tol = 10e-4
max_iter = 50
a = 0
b = 3

if Golden:
    fig, ax = plt.subplots(nrows=3,ncols=2)

    for i in range(2):
        x=np.linspace(a,b,100)
        ax[0,i].plot(x, [f[i](x1) for x1 in x], zorder=10)
        ax[0,i].set_title(titles[i])
        ax[0,i].set_xlabel('$x$')
        ax[0,i].set_ylabel('$f_{'+f'{i+9}'+'}(x)$')
        
        xmin, fmin, k, exit_flag = golden(f[i], a, b, max_iter, tol)
        print(f'Golden section applied to f{i+9} exited after {k} iterations, ')
        
        ax[0,i].scatter(xmin, fmin, zorder=11)
        ax[0,i].scatter(xmin[-1],fmin[-1], zorder=12)       
        ax[0,i].grid(True)

        ax[1,i].scatter(range(len(fmin)),fmin, zorder=10)
        ax[1,i].set_xlabel('$k$')
        ax[1,i].set_ylabel('$f_{'+f'{i+9}'+'}(x^{(k)})$')
        ax[1,i].grid(True)

        ax[2,i].scatter(range(len(fmin)),xmin, zorder=10)
        ax[2,i].set_xlabel('$k$')
        ax[2,i].set_ylabel('$x^{(k)}$')
        ax[2,i].grid(True)
        
        fig.tight_layout()
        fig.canvas.set_window_title('Golden search') 
    plt.show()

if Brent:
    fig, ax = plt.subplots(nrows=3,ncols=2)

    for i in range(2):
        x=np.linspace(a,b,100)
        ax[0,i].plot(x, [f[i](x1) for x1 in x], zorder=10)
        ax[0,i].set_title(titles[i])
        ax[0,i].set_xlabel('$x$')
        ax[0,i].set_ylabel('$f_{'+f'{i+9}'+'}(x)$')

        xmin, fmin, k, exit_flag = brent(f[i], a, b, max_iter, tol)
        print(f'Brent\'s method applied to f{i+9} exited after {k} iterations, '+str(exit_flag))
        
        ax[0,i].scatter(xmin, fmin, zorder=11)
        ax[0,i].scatter(xmin[-1],fmin[-1], zorder=12)
        ax[0,i].grid(True)

        ax[1,i].scatter(range(len(fmin)),fmin, zorder=10)
        ax[1,i].set_xlabel('$k$')
        ax[1,i].set_ylabel('$f_{'+f'{i+9}'+'}(x^{(k)})$')
        ax[1,i].grid(True)

        ax[2,i].scatter(range(len(fmin)),xmin, zorder=10)
        ax[2,i].set_xlabel('$k$')
        ax[2,i].set_ylabel('$x^{(k)}$')
        ax[2,i].grid(True)

        fig.tight_layout()
        fig.canvas.set_window_title('Brent\'s method') 
    plt.show()

