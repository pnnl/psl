"""
Base Control Profiles for System excitation
"""

import numpy as np


def RandomWalk(nx=1, nsim=100, xmax=1, xmin=0, sigma=0.05):
    """

    :param nx:
    :param nsim:
    :param xmax:
    :param xmin:
    :return:
    """
    if type(xmax) is not np.ndarray:
        xmax = np.asarray(nx*[xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray(nx*[xmin]).ravel()

    Signals = []
    for k in range(nx):
        Signal = [0]
        for t in range(1, nsim):
            yt = Signal[t - 1] + np.random.normal(0, sigma)
            if (yt > 1):
                yt = Signal[t - 1] - abs(np.random.normal(0, sigma))
            elif (yt < 0):
                yt = Signal[t - 1] + abs(np.random.normal(0, sigma))
            Signal.append(yt)
        Signals.append(xmin[k] + (xmax[k] - xmin[k])*np.asarray(Signal))
    return np.asarray(Signals).T


def WhiteNoise(nx=1, nsim=100, xmax=1, xmin=0):
    """
    White Noise
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    """
    if type(xmax) is not np.ndarray:
        xmax = np.asarray([xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin]).ravel()
    Signal = []
    for k in range(nx):
        signal = xmin[k] + (xmax[k] - xmin[k])*np.random.rand(nsim)
        Signal.append(signal)
    return np.asarray(Signal).T


def Step(nx=1, nsim=100, tstep=50, xmax=1, xmin=0):
    """
    step change
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param tstep: (int) time of the step
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    """
    if type(xmax) is not np.ndarray:
        xmax = np.asarray([xmax]).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin]).ravel()
    Signal = []
    for k in range(nx):
        signal = np.ones(nsim)
        signal[0:tstep] = xmin[k]
        signal[tstep:] = xmax[k]
        Signal.append(signal)
    return np.asarray(Signal).T


def Ramp():
    """
    ramp change
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    """
    pass


def Periodic(nx=1, nsim=100, numPeriods=1, xmax=1, xmin=0, form='sin'):
    """
    periodic signals, sine, cosine
    :param nx: (int) Number signals
    :param nsim: (int) Number time steps
    :param numPeriods: (int) Number of periods
    :param xmax: (int/list/ndarray) signal maximum value
    :param xmin: (int/list/ndarray) signal minimum value
    :param form: (str) form of the periodic signal 'sin' or 'cos'
    """
    if type(xmax) is not np.ndarray:
        xmax = np.asarray([xmax]*nx).ravel()
    if type(xmin) is not np.ndarray:
        xmin = np.asarray([xmin]*nx).ravel()

    xmax = xmax.reshape(nx)
    xmin = xmin.reshape(nx)

    samples_period = nsim// numPeriods
    leftover = nsim % numPeriods
    Signal = []
    for k in range(nx):
        if form == 'sin':
            base = xmin[k] + (xmax[k] - xmin[k])*(0.5 + 0.5 * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / samples_period)))
        elif form == 'cos':
            base = xmin[k] + (xmax[k] - xmin[k])*(0.5 + 0.5 * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / samples_period)))
        signal = np.tile(base, numPeriods)
        signal = np.append(signal, base[0:leftover])
        Signal.append(signal)
    return np.asarray(Signal).T


def SignalComposite():
    """
    composite of signal excitations
    allows generating heterogenous signals
    """
    pass


def SignalSeries():
    """
    series of signal excitations
    allows combining sequence of different signals
    """
    pass

