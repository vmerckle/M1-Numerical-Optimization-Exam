#!/usr/bin/env python
# coding: utf-8


import numpy as np
import timeit

def proximal_gradient_algorithm(F , f_grad , g_prox , x0 , step , PREC , ITE_MAX , PRINT ):
    x = np.copy(x0)
    x_tab = np.copy(x)
    if PRINT:
        print("------------------------------------\n Proximal gradient algorithm\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        g = f_grad(x)
        x = g_prox(x - step*g , step)  #######  ITERATION

        x_tab = np.vstack((x_tab,x))


    t_e =  timeit.default_timer()
    if PRINT:
        print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,F(x)))
    return x, x_tab

