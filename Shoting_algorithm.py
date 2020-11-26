'''
Zbigniew Pepliski
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')

'''
                              Relevant functions
'''                             

alpha = []
beta = []
H = []


def L(t, y, dy):
    '''
    Function that describes the companies profit, with repsect to a pennalty P
    over time t

    Parameters
    ----------

    t : vector
        Returning vector, Completes definition of ODE
    y : vector
        an aproximation of the profit curve used to find optimal solution
    dy : vector
        an aproximation of the forst derivative of the profit curve
         used to find optimal solution
    Returns
    -------

    P - y : vector
        Curve to be minimized to find maximalprofit
    '''
    assert((not np.any(np.isnan(y))) and np.all(np.isfinite(y)) and
    np.all(np.isreal(y))),\
        "y must be real, finite and not NaN"
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and
    np.all(np.isreal(t))),\
        "t must be real, finite and not NaN"
    assert((not np.any(np.isnan(dy))) and np.all(np.isfinite(dy)) and
    np.all(np.isreal(dy))),\
        "dy must be real, finite and not NaN"
    a = alpha[-1]
    b = beta[-1]
    P = a * (dy**2) + b * (t**2 - 1) * (dy**3)
    return P - y


def Euler_Lagrange(y, t):
    '''
    The Euler Lagrange equation, of a function y(t) for which if satisfied
     maximizes the companies profit

    Parameters
    ----------
    y : vector
        an aproximation of the profit curve used to find optimal solution
    t : vector
        Returning vector, Completes definition of ODE

    Returns
    -------

    coef : Vector
        A Vector that represents an approximation of the Euler Lagrnage
        equations, expressed in first order terms
    '''
    assert((not np.any(np.isnan(y))) and np.all(np.isfinite(y)) and
    np.all(np.isreal(y))),\
        "y must be real, finite and not NaN"
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and
    np.all(np.isreal(t))),\
        "t must be real, finite and not NaN"
    h = H[-1]
    y0 = y[0]
    dy = y[1]

    def dL_dy(t, y0, dy):
        return (L(t, y0 + h, dy) - L(t, y0 - h, dy))/(2*h)

    def dL_ddy(t, y0, dy):
        return (L(t, y0, dy + h) - L(t, y0, dy - h))/(2*h)

    d2L_dt_ddy = (dL_ddy(t + h, y0, dy) - dL_ddy(t - h, y0, dy))/(2*h)

    d2L_dy_ddy = (dL_ddy(t, y0 + h, dy) - dL_ddy(t, y0 - h, dy))/(2*h)

    d2L_2ddy = (L(t, y0, dy + h) + L(t, y0, dy - h) - 2*L(t, y0, dy))/(h**2)

    d2y = (dL_dy(t, y0, dy) - d2L_dt_ddy - (dy * d2L_dy_ddy))/d2L_2ddy

    coef = np.zeros(2)

    coef[0] = dy
    coef[1] = d2y

    return coef


def shooting(f, y_bc, t, guess_interval, a, b, h, dt):
    """
    Solve the BVP z' = f(x, z) on t in ivp_interval = [a, b]
    where z = [y, y'], subject to boundary conditions v_bc

     Parameters
    ----------
    f : function/vector
        Represent the function to be solved
    y_bc: vector
        two point vector representing both ends of the bracketing interval
        of the boundary conditions of the problem
    t : vector
        represent the time interval over which the function to be solved
        is analysed
    guess_interval : vector
        represents the initial guess, from which BVP can be converted into an
        IVP problem
    a, b : parameters
        Establish the coefficients of function L, which is to be solved to
        maximize profit funtion
    h : parameter
        detrmines the accuracy of the estimates of the derivatives of 'L'
    dt : parameter
        Determines the accuracy of the computed solution to the Euler Lagrange
        problem, by carying the number of time steps

    Returns
    -------

    [x, y] : Vector
        a list of numbers that represent the aproximated solution to the Euler
        lagrange problem, obtained from implementing the shooting algorithm
    """
    assert((not np.any(np.isnan(y_bc))) and np.all(np.isfinite(y_bc)) and
    np.all(np.isreal(y_bc))),\
        "y_bc must be real, finite and not NaN"
    assert(len(y_bc) == 2), "y_bc must have 2 components"
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and
    np.all(np.isreal(t))),\
        "t must be real, finite and not NaN"
    assert(len(guess_interval) == 2), "guess_interval must have 2 components"
    assert((not np.any(np.isnan(a))) and np.all(np.isfinite(a)) and
    np.all(np.isreal(a))),\
        "a must be real, finite and not NaN"
    assert((not np.any(np.isnan(b))) and np.all(np.isfinite(b)) and
    np.all(np.isreal(b))),\
        "b must be real, finite and not NaN"
    assert((not np.any(np.isnan(h))) and np.all(np.isfinite(h)) and
    np.all(np.isreal(h))),\
        "h must be real, finite and not NaN"
    assert(h > 0.0009), "h must be larger than 0.0009"
    assert((not np.any(np.isnan(dt))) and np.all(np.isfinite(dt)) and
    np.all(np.isreal(dt))),\
        "dt must be real, finite and not NaN"
    assert(dt > 0.0009), "dt must be larger than 0.0009"
    alpha.append(a)
    beta.append(b)
    H.append(h)

    def shooting_phi(guess):
        """
        Internal function for root-finding that defines the error in the
        boundary condition at b (at the second end of the bracketing interval),
        by solving the IVP at the first boundary condition and the guess
        Parameters
        """
        y0 = [y_bc[0], guess]
        y = integrate.odeint(f, y0, np.arange(dt, 1 + dt, dt))
        return y[-1, 0] - y_bc[1]

    guess = optimize.brentq(shooting_phi, guess_interval[0],
                            guess_interval[1])

    y0 = [y_bc[0], guess]
    x = np.arange(dt, 1 + dt, dt)
    y = integrate.odeint(f, y0, x)
    return [x, y]

'''
                             Running Algorithms
Solves the BVP of the Euler Lagrange equation using Alpha = 5 & Beta = 5,
given in the problem sheet. With relevant parameters, as specified in the
problem sheet
'''
guess_interval = np.array([-10, 10])
x, y_brentq = shooting(Euler_Lagrange, [1.0, 0.9],
                       [0.0, 1.0], guess_interval, 5, 5, 1e-2, 0.001)

'''
Solves the BVP of the Euler Lagrange equation using Alpha = 7/4 & Beta = 5,
given in the problem sheet. With relevant parameters, as specified in the
problem sheet
'''
x2, y2_brentq = shooting(Euler_Lagrange, [1.0, 0.9],
                         [0.0, 1.0], guess_interval, 7.0/4, 5, 1e-2, 0.001)


def plot_1and2():
    '''
    plots results from running the shooting algorithm over the Euler Lagrnage
    equation, that satisfies the maximization of the profit function, in terms
    of the price curve y(t)

    Figure 1: Plots the approximated solution of the price curve against time,
    with coefficient of the penalty function of: Alpha = 5 & Beta = 5

    Figure 2: Plots the approximated solution of the price curve against time,
    with coefficient of the penalty function of: Alpha = 7/4 & Beta = 5
    '''
    plt.figure(1, figsize=(12, 8))
    plt.plot(x, y_brentq[:, 0])
    plt.xlabel('$t$', size=16)
    plt.ylabel('$y$', size=16)
    plt.legend(('Shooting(brentq method)',))
    plt.title("Price Trend Over One Year That Maximize Profit,\
 Alpha = Beta = 5")

    plt.figure(2, figsize=(12, 8))
    plt.plot(x2, y2_brentq[:, 0])
    plt.xlabel('$t$', size=16)
    plt.ylabel('$y$', size=16)
    plt.legend(('Shooting(brentq method)',))
    plt.title("Price Trend Over One Year That Maximizes Profit,\
 Alpha = 7/4, Beta = 5")

plot_1and2()

'''
                                Convergence Check
for loops to prove convergence of the shooting algorithm for solving
the Euler lagrenage equation
'''

E = []
N = []
plt.figure(3, figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.xlabel('$t$', size=16)
plt.ylabel('$y$', size=16)
for i in np.arange(3, 9, 1):
    n = 2**i
    dt = 1/n
    N.append(np.log(n))
    guess_interval = np.array([10, -10])
    x, y_brentq = shooting(Euler_Lagrange, [1.0, 0.9],
                           [0.0, 1.0], guess_interval, 7.0/4, 5, 1e-2, dt)
    e = abs(y_brentq[-1][0] - 0.9)
    E.append(np.log(e))
    plt.plot(x, y_brentq[:, 0])
slope = np.polyfit(N, E, 1)[0]
Y_intercept = np.polyfit(N, E, 1)[1]

EY = []
for i in N:
    EY.append(Y_intercept+slope*i)

'''
yes, problem was posed correctly because the boundary conditions have been
applied correctly
'''


def plot_convergence():
    '''
    Figure 3:

    Subplot 1:
    Plots the profit curves for a range of 'dt's' that have been used to show
    convergence. Outlines the curve convergence across the entire range 't'

    Subplot 2:
    Plots the error of the algorith against the Number of timesteps,
    which outlines the convergance rate of the algorith which is directly
    corelated to the slope of the plot
    '''
    plt.subplot(2, 1, 2)
    plt.scatter(N, E)
    plt.plot(N, EY)
    plt.xlabel('$log(h)$', size=16)
    plt.ylabel('log(|Error|)', size=16)
    plt.legend(["Error Curve (slope = {})".format(round(slope, 2))],
               loc='upper left')
    plt.suptitle("Proof of Third order Convergence from the shooting method,\
 Alpha = 7/4, Beta = 5", size=16)
plot_convergence()
