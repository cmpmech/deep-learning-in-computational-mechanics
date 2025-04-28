import tensorflow as tf
import numpy as np


def evp1d_dlcm2(deps=1e-2):
    # Elastic
    E = 3.2                         # Elastic modulus
    Wtype = 2                       # 0: linear elastic, 1: Hencky - logarithmic, 2: Power law
    Wparam = np.array([0.7])        # Exponent for Wtype=2

    # Visco - elastic
    Ev = np.array([1.0, 0.4])       # Values of visco - elastic moduli
    tau = np.array([10.0, 40.0])    # Values of visco - elastic relaxation times
    mv = len(Ev)                    # Number of visco - elsatic branches
    WtypeV = 0                      # 0: linear elastic, 1: Hencky - logarithmic, 2: Power law
    WparamV = []                    # Exponent for WtypeV=2

    # Starting values & eps - increment: eps(t) = eps0 + t * deps
    eps0 = 0
    eps1 = 0.3                      # Strain
    # deps = 1e-2                   # Strain rate
    simcase = 0                     # 0: Loading + Unloading, 1: Loading + Relaxation

    # Time integration with backward Euler
    # Time steps
    n = np.ceil((eps1 - eps0) * 200)
    T = (eps1 - eps0) / deps
    dt = T / n
    if simcase == 0:
        n = int(2 * n)
    elif simcase == 1:
        n0 = n
        n = int(n + np.ceil(200 / dt))

    # Initalizations
    pt = np.zeros(shape=(n + 3, 1))
    pe = np.zeros(shape=(n + 3, 1))
    ps = np.zeros(shape=(n + 3, 1))
    pa = np.zeros(shape=(n + 3, mv))
    eps = 0
    alp = np.zeros(shape=(1, mv))

    # Loop
    for i in range(n+2):
        # Update total strain depending on load case
        if simcase == 0:
            if i <= n / 2:
                eps = eps + dt * deps
            else:
                eps = eps - dt * deps

        elif simcase == 1:
            if i <= n0:
                eps = eps + dt * deps

        # Backward Euler step on visco - equation
        alp = alp + dt / (dt + tau) * (eps - alp)
        dalp = (eps - alp) / tau

        # Visco - elastic stress
        sigV = 0
        for kv in range(mv):
            _, sigVm, _ = PsiE(eps - alp[0, kv], Ev[kv], WtypeV, WparamV)
            sigV = sigV + sigVm

        # Elastic stress
        W0, sigE, _ = PsiE(eps, E, Wtype, Wparam)

        # Total stress
        sig = sigE + sigV

        pt[i, 0] = i * dt
        pe[i+1] = eps
        ps[i+1] = sig
        pa[i+1, :] = alp

    pt[i+1, 0] = (i+1) * dt

    pt = tf.cast(tf.concat(pt, axis=0), dtype=tf.float32)
    pe = tf.cast(tf.concat(pe, axis=0), dtype=tf.float32)
    ps = tf.cast(tf.concat(ps, axis=0), dtype=tf.float32)

    return pt, pe, ps


def PsiE(eps, E, type, param):
    if type == 0:  # Linear / StVK
        W = 0.5 * E * eps ** 2

        sig = E * eps

        dsig = E

    elif type == 1:  # Hencky
        lne = np.log(1 + eps)
        W = 0.5 * E * lne ** 2

        sig = E * lne / (1 + eps)

        dsig = E / (1 + eps) ^ 2 * (1 - lne)

    elif type == 2:  # Exponential
        aa = param[0]
        W = E / (aa * (aa - 1)) * ((1 + eps) ** aa - aa * eps - 1)

        sig = E / (aa - 1) * ((1 + eps) ** (aa - 1) - 1)

        dsig = E * (1 + eps) ** (aa - 2)

    return W, sig, dsig
