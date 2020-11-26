# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')


def f(t, qn, R, e, w):
    """
    Function defines the modified Prothero-Robinson problem

    Parameters
    ----------

    t : function
        Returning vector, Completes definition of ODE
    qn : vector
        Containing the data for qn
    dt : vector
        giving the time step, which is the used for finding the solution
    r, e, w : parameters
        Describe the right hand side

    Returns
    -------

    f : Function
        modified Prothero-Robinson function at time 't'
    """
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and
    np.all(np.isreal(t))),\
        "t must be real, finite and not NaN"
    assert(len(qn) == 2), "qn must have length 2"
    assert((not np.any(np.isnan(R))) and np.all(np.isfinite(R)) and
    np.all(np.isreal(R))),\
        "r must be real, finite and not NaN"
    assert((not np.any(np.isnan(e))) and np.all(np.isfinite(e)) and
    np.all(np.isreal(e))),\
        "e must be real, finite and not NaN"
    assert((not np.any(np.isnan(w))) and np.all(np.isfinite(w)) and
    np.all(np.isreal(w))),\
        "w must be real, finite and not NaN"
    x = qn[0]
    y = qn[1]
    A = np.array([[R, e],
                 [e, -1]])
    B = np.array([(-1.0+x**2.0-np.cos(t))/(2.0*x),
                  ((-2+y**2.0-np.cos(w*t))/(2.0*y))])
    C = np.array([(np.sin(t)/(2*x)),
                  (w*np.sin(w*t)/(2*y))])
    return np.dot(A, B) - C


def MyRK3_step(f, t, qn, dt, R, e, w):
    """
    Function implements the Implicit third order Runge-Kutta algorithm on a
    ODE 'f'  from time step 't' to time 't+dt' with additional arguments

    Parameters
    ----------

    f : function
        Defines the ODE
    t : function
        Returning vector, Completes definition of ODE
    qn : vector
        Containing the data for qn
    dt : vector
        giving the time step, which is the used for finding the solution
    r, e, w : parameters
        Describe the right hand side

    Returns
    -------

    qnp1R : array of float
        Solution of the ODE at time 't+dt'
    """
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and
    np.all(np.isreal(t))),\
        "t must be real, finite and not NaN"
    assert((not np.any(np.isnan(dt))) and np.all(np.isfinite(dt)) and
    np.all(np.isreal(dt))),\
        "dt must be real, finite and not NaN"
    assert(len(qn) == 2), "qn must have length 2"
    assert(hasattr(f, '__call__')),\
        "f must be a callable function"
    assert((not np.any(np.isnan(R))) and np.all(np.isfinite(R)) and
    np.all(np.isreal(R))),\
        "r must be real, finite and not NaN"
    assert((not np.any(np.isnan(e))) and np.all(np.isfinite(e)) and
    np.all(np.isreal(e))),\
        "e must be real, finite and not NaN"
    assert((not np.any(np.isnan(w))) and np.all(np.isfinite(w)) and
    np.all(np.isreal(w))),\
        "w must be real, finite and not NaN"
    k1 = f(t, qn, R, e, w)
    k2 = f(t + dt/2.0, qn + dt*(k1/2.0), R, e, w)
    k3 = f(t + dt, qn + dt*(-k1 + 2*k2), R, e, w)
    qnp1R = qn + (dt*(k1+4*k2+k3))/6.0
    return qnp1R


def MyGRRK3_step(f, t, qn, dt, r, e, w):
    """
    Function implements the Explicit third order Runge-Kutta algorithm on a
    ODE 'f'  from time step 't' to time 't+dt' with additional arguments

    Parameters
    ----------

    f : function
        Defines the ODE
    t : function
        Returning vector, Completes definition of ODE
    qn : vector
        Containing the data for qn
    dt : vector
        giving the time step, which is the used for finding the solution
    r, e, w : parameters
        Describe the right hand side

    Returns
    -------

    qnpG1 : array of float
        Solution of the ODE at time 't+dt'
    """
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and
    np.all(np.isreal(t))), \
        "t must be real, finite and not NaN"
    assert((not np.any(np.isnan(dt))) and np.all(np.isfinite(dt)) and
    np.all(np.isreal(dt))), \
        "dt must be real, finite and not NaN"
    assert(len(qn) == 2), "qn must have length 2"
    assert(hasattr(f, '__call__')), \
        "f must be a callable function"
    assert((not np.any(np.isnan(r))) and np.all(np.isfinite(r)) and
    np.all(np.isreal(r))), \
        "r must be real, finite and not NaN"
    assert((not np.any(np.isnan(e))) and np.all(np.isfinite(e)) and
    np.all(np.isreal(e))), \
        "e must be real, finite and not NaN"
    assert((not np.any(np.isnan(w))) and np.all(np.isfinite(w)) and
    np.all(np.isreal(w))), \
        "w must be real, finite and not NaN"

    def F(k0):
        """
        Function defines the set of nonlinear equations describing k1 and k2
        of the third order explicit Runge-Kutta algorithm

        Parameters
        ----------

        k0 : vector
            intial guess for roots of the problem

        Returns
        -------

        f3 : vector
            set of nonlinear equations of k1 and k2
        """
        assert((not np.any(np.isnan(k0))) and np.all(np.isfinite(k0)) and
        np.all(np.isreal(k0))),\
            "k0 must be real, finite and not NaN"
        assert(len(k0) == 4), "K must have length 4"
        assert(hasattr(F, '__call__')), \
            "F must be a callable function"
        k1 = np.array([k0[0], k0[1]])
        k2 = np.array([k0[2], k0[3]])
        f1 = k1 - np.array([f(t + dt / 3,
                              qn + (dt / 12) * (5 * k1 - k2), r, e, w)])
        f2 = k2 - np.array([f(t + dt,
                              qn + (dt / 4) * (3 * k1 + k2), r, e, w)])
        f3 = np.reshape(np.array([f1, f2]), (4,))
        return f3

    k0 = np.reshape(np.array([f(t + dt / 3, qn, r, e, w),
                              f(t + dt, qn, r, e, w)]), (4,))
    k = fsolve(F, k0)
    k1 = np.array([k[0], k[1]])
    k2 = np.array([k[2], k[3]])
    qnpG1 = qn + (dt / 4) * (3 * k1 + k2)
    return qnpG1


def q(t, w):
    '''
    A function defininig the exact solution to the differential equation
    with respect to time 't' and w
    '''
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and
    np.all(np.isreal(t))),\
        "t must be real, finite and not NaN"
    assert((not np.any(np.isnan(w))) and np.all(np.isfinite(w)) and
    np.all(np.isreal(w))),\
        "w must be real, finite and not NaN"
    q1 = np.sqrt(1 + np.cos(t))
    q2 = np.sqrt(2 + np.cos(w*t))
    q = [q1, q2]
    return q


'''                              non-stiff scenarios                        '''
qnR = np.array([np.sqrt(2), np.sqrt(3)])
qnG = np.array([np.sqrt(2), np.sqrt(3)])

QR0 = []
QR1 = []
QG0 = []
QG1 = []
X0 = []
X1 = []
T = []

for i in np.arange(0, 1.05, 0.05):
    '''
    for loop to implement the implicit and explicit variations of the
    Runge-Kutta for the non-stiff scenario, from time t=0 to t=1
    '''
    dt = 0.05
    R = -2.0
    w = 5.0
    e = 0.05
    T.append(i)
    QR0.append(qnR[0])
    QR1.append(qnR[1])
    QG0.append(qnG[0])
    QG1.append(qnG[1])
    X0.append(q(i, w)[0])
    X1.append(q(i, w)[1])
    qnR = MyRK3_step(f, i, qnR, dt, R, e, w)
    qnG = MyGRRK3_step(f, i, qnG, dt, R, e, w)

'''plots the figures for the non-stif scenario for both x and y'''

eR = []
eG = []
edt = []

for j in np.arange(7, -1, -1):
    qneR = np.array([np.sqrt(2), np.sqrt(3)])
    qneG = np.array([np.sqrt(2), np.sqrt(3)])
    t = 0
    dt = 0.1/(2.0**j)
    edt.append(np.log(dt))
    R = -2.0
    w = 5.0
    e = 0.05
    eRj = []
    eGj = []
    for i in np.arange(0, 0.4, dt):
        yR = qneR[1]
        yG = qneG[1]
        qneR = MyRK3_step(f, i, qneR, dt, R, e, w)
        qneG = MyGRRK3_step(f, i, qneG, dt, R, e, w)
        yX = q(i, w)[1]
        eRj.append(abs(yR-yX))
        eGj.append(abs(yG-yX))
    eR.append(np.log(dt*sum(eRj)))
    eG.append(np.log(dt*sum(eGj)))
slopeR = np.polyfit(edt, eR, 1)[0]
slopeG = np.polyfit(edt, eG, 1)[0]
print(eR)
print(edt)
print(len(eR))
'''                              stiff scenarios                            '''

qnRS = np.array([np.sqrt(2), np.sqrt(3)])
QRS0 = []
QRS1 = []
XRS0 = []
XRS1 = []
TRS = []
for i in np.arange(0, 0.03, 0.001):
    '''
    for loop to implement the implicit formula of the
    Runge-Kutta for the stiff scenario, from time t=0 to t = 0.3
    '''
    dt = 0.001
    R = -2.0*(10**5)
    w = 20
    e = 0.5
    TRS.append(i)
    QRS0.append(qnRS[0])
    QRS1.append(qnRS[1])
    XRS0.append(q(i, w)[0])
    XRS1.append(q(i, w)[1])
    qnRS = MyRK3_step(f, i, qnRS, dt, R, e, w)

qnGS = np.array([np.sqrt(2), np.sqrt(3)])
QGS0 = []
QGS1 = []
XGS0 = []
XGS1 = []
TGS = []
for i in np.arange(0, 1, 0.005):
    '''
    for-loop to implement the explicit formula of the
    Runge-Kutta for the stiff scenario, from time t=0 to t=1
    '''
    dt = 0.005
    R = -2.0E5
    e = 0.5
    TGS.append(i)
    QGS0.append(qnGS[0])
    QGS1.append(qnGS[1])
    XGS0.append(q(i, w)[0])
    XGS1.append(q(i, w)[1])
    qnGS = MyGRRK3_step(f, i, qnGS, dt, R, e, w)

eSG = []
eSdt = []
for j in np.arange(7, -1, -1):
    '''
    for-loop to prove the convergence rate of the explicit Runge-Kutta
    algorith for the stiff scenario
    '''
    qneR = np.array([np.sqrt(2), np.sqrt(3)])
    qneG = np.array([np.sqrt(2), np.sqrt(3)])
    t = 0
    dt = 0.05/(2.0**j)
    eSdt.append(np.log(dt))
    R = -2.0E5
    w = 20.0
    e = 0.5
    eGj = []
    for i in np.arange(0, 0.4, dt):
        yG = qneG[1]
        qneG = MyGRRK3_step(f, i, qneG, dt, R, e, w)
        yX = q(i, w)[1]
        eGj.append(abs(yG-yX))
    eSG.append(np.log(dt*sum(eGj)))
slopeGS = np.polyfit(eSdt, eSG, 1)[0]


def plot1():
    '''
    plots results from running the implicit and explicit thrid order
    Runge-Kutta of the Modified Prothero-Robinson problem under non-stiff
    and stiff conditions

    Figure 1: plots the exact solution as well as two estimations using the
    implicit and explicit Runge-Kutta algorithms of the
    Modified Prothero-Robinson problem under non-stiff conditions

    Figure 2: Plots the norm 1 error of the implicit and explicit Runge-Kutta
    algorithms of the Modified Prothero-Robinson problem under non-stiff
    conditions

    Figure 3: plots the exact solution as well as the estimation found by using
    the implicit Runge-Kutta algorithm of the Modified Prothero-Robinson
    problem under stiff conditions

    Figure 4: plots the exact solution as well as the estimation found by using
    the explicit Runge-Kutta algorithm of the Modified Prothero-Robinson
    problem under stiff conditions

    Figure 5: Plots the norm 1 error of the explicit Runge-Kutta algorithm
    of the Modified Prothero-Robinson problem under stiff conditions
    '''
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(T, X0)
    plt.plot(T, QR0)
    plt.plot(T, QG0)
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    plt.legend(["Exact", "RK3", "GRRK3"])

    plt.subplot(1, 2, 2)
    plt.plot(T, X1)
    plt.plot(T, QR1)
    plt.plot(T, QG1)
    plt.xlabel('t[s]')
    plt.ylabel('y(t)')
    plt.suptitle("Modified Prothero-Robinson Solutions - non-stiff Case")
    plt.legend(["Exact", "RK3", "GRRK3"])
    plt.show()

    plt.figure(2)
    plt.plot(edt, eR)
    plt.plot(edt, eG)
    plt.plot(edt, eR, 'x')
    plt.plot(edt, eG, 'x')
    plt.xlabel('log($\Delta$t)')
    plt.ylabel('log(|Global Error|)')
    plt.legend(["RK3 (slope = {})".format(round(slopeR, 2)),
    "GRRK3 (slope = {})".format(round(slopeG, 2))], loc='upper left')
    plt.title("Proof of Third Order Convergence of the RK3 & GRRK3 Algorithms\
 for the non-Stiff Case")
    plt.show()

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.plot(TRS, XRS0)
    plt.plot(TRS, QRS0)
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    plt.legend(["Exact", "RK3"], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(TRS, XRS1)
    plt.plot(TRS, QRS1)
    plt.xlabel('t[s]')
    plt.ylabel('y(t)')
    plt.legend(["Exact", "RK3"], loc='lower left')
    plt.suptitle("Modified Prothero-Robinson Solutions - Stiff Case")
    plt.show()

    plt.figure(4)
    plt.subplot(1, 2, 1)
    plt.plot(TGS, XGS0)
    plt.plot(TGS, QGS0)
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    plt.legend(["Exact", "GRRK3"])
    plt.subplot(1, 2, 2)
    plt.plot(TGS, XGS1)
    plt.plot(TGS, QGS1)
    plt.xlabel('t[s]')
    plt.ylabel('y(t)')
    plt.legend(["Exact", "GRRK3"])
    plt.suptitle("Modified Prothero-Robinson Solutions - Stiff Case")
    plt.show()

    plt.figure(5)
    plt.plot(edt, eSG)
    plt.plot(edt, eSG, 'x')
    plt.xlabel('log($\Delta$t)')
    plt.ylabel('log(|Global Error|)')
    plt.legend(["GRRK3 (slope = {})".format(round(slopeGS, 2))],
    loc='upper left')
    plt.title("Proof of Third Order Convergence of the GRRK3 Algorithm for the\
 Stiff Case")
    plt.show()
plot1()
