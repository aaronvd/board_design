import scipy.constants
import numpy as np

C     = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0 #C^2/(N*m^2)
MU_0  = scipy.constants.mu_0    #m kg s^-2 A^-2
mm = 0.001
cm    = 0.01
GHz   = 1.0E9

def cutoff_frequency_10(eps_r, a):
    '''
    Calculates cutoff frequency for TE_10 mode.

    a: waveguide width
    eps_r: waveguide dielectric constant

    '''
    return 1/(2*a * np.sqrt(MU_0*EPS_0*eps_r))

def beta_g(eps_r, a, f):
    '''
    Calculates rectangular waveguide propagation constant.

    a: waveguide width
    eps_r: waveguide dielectric constant
    f: operating frequency

    '''
    fc = cutoff_frequency_10(eps_r, a)
    beta = 2*np.pi*f * np.sqrt(MU_0*EPS_0*eps_r)
    return beta * np.sqrt(1-(fc/f)**2)

def lam_g(eps_r, a, f):
    '''
    Calculates guided wavelength for rectangular waveguide

    a: waveguide width
    eps_r: waveguide dielectric constant
    f: operating frequency

    '''
    return 2*np.pi/beta_g(eps_r, a, f)

def siw_width(w_rw, d, s):
    '''
    Calculates SIW width corresponding to equivalent rectangular waveguide width w_rw
    '''
    a = 1
    b = -(1.08*d**2/s + w_rw)
    c = 0.1*d**2
    
    w_plus = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    w_minus = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    
    return w_plus, w_minus




