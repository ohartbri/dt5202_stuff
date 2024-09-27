#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter


import pandas as pd
import scipy.interpolate
import scipy.stats
import scipy.optimize
from timeit import default_timer as timer

import uproot

import cmasher as cmr

from ipywidgets import IntProgress
from IPython.display import display
from ipywidgets import interact



matplotlib.rcParams['figure.figsize'] = [12,8]
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.bf'] = 'Dejavu Sans:italic:bold'


M = {}
M['electron'] = 0.000511
M['pion'] =   0.13957
M['kaon'] =   0.49367
M['proton'] = 0.93827

particles = {11:'electron', -211:'pion', -321:'kaon', 2212:'proton'}


def pdg_to_mass(pdg):
    condlist = [pdg == 11, pdg == -211, pdg == -321, pdg == 2212]
    choicelist = [0.000511, 0.13957, 0.49367, 0.93827]
    return np.select(condlist, choicelist)

texname = {'electron':r'$e^\pm$', 'pion':r'$\pi^\pm$', 'kaon':r'$K^\pm$', 'proton':r'$p/\bar{p}$'  }

def tof(p, m, l):
    c = 299.792 #mm/ns
    p = p 
    m = m 
    betagamma = p/m
    #print(betagamma)
    gamma = np.sqrt(1+(betagamma)**2)
    #print(gamma)
    
    beta = betagamma/gamma
    
    #print(beta)
    
    return l/c/beta

def tof_msq(p, msq, l):
    c = 299.792 #mm/ns
    #beta = p / (np.sqrt(c**2 * msq + p**2))
    beta = p / (np.sqrt(msq + p**2))
    return l/c/beta

def mass(p, v):
    c = 299.792 #mm/ns
    beta = v/c
    gamma = 1/np.sqrt(1-(beta**2)) 
    return c*p/(gamma*v)

def mass_sq(p, v):
    c = 299.792 #mm/ns
    beta = (v/c)
#     gamma_sq = 1/(1-(beta**2))
#     return (c*p)**2/(gamma_sq*v**2)
    return p**2*(1/(beta**2) - 1) 

vec_tof = np.vectorize(tof)

def mass_sq2(p, beta):
    c = 299.792 #m/ns
    return p**2 * (1/beta**2 - 1)


# In[3]:


def radius(p, B, q = 1):
    '''return the radius of curvature of a charged particle in magnetic field B
    p: momentum in GeV/c
    B: magnetic field in T
    q: charge in units of the electron charge

    returns the radius in meters
    '''
    return 3.336 * (p / (q * B)) # 3.336 contains all the constants and units

def momentum_cutoff(r, B, q = 1):
    '''return the minimum momentum for a particle to have magnetic curl radius of at least r
    note that in a cylindrical (barrel detector) configuration, the minimum curling radius to reach the detector is half the detector barrel radius, since particles emanate from the IP at the center of the barrel.

    r: radius of the detector barrel in meters
    B: magnetic field in T
    q: charge in units of the electron charge

    returns the momentum in GeV/c
    '''
    return r /3.336 * q * B # 3.336 contains all the constants and units

def arc_length(r, c):
    '''return the arc length of a circle of radius r and chord length c
    (in the case of a charged particle track the chord length is the radial distance IP to detector, r is the radius of track curvature)
    r: radius of the circle
    c: chord length

    returns the arc length in meters
    '''

    theta = 2 * np.arcsin(c / (2 * r)) # angle subtended by the chord

    return r * theta 

def eta_to_theta(eta):
    '''return the polar angle theta from the pseudorapidity eta
    eta: pseudorapidity

    returns the polar angle in radians
    '''
    return 2 * np.arctan(np.exp(-eta))

def theta_to_eta(theta):
    '''return the pseudorapidity eta from the polar angle theta
    theta: polar angle in radians

    returns the pseudorapidity
    '''
    return -np.log(np.tan(theta / 2))

def split_momentum(p, theta):
    '''return the transverse and longitudinal momentum components from the total momentum p and polar angle theta
    p: total momentum in GeV/c
    theta: polar angle in radians
    
    return p_T, p_L'''

    return p * np.sin(theta), p * np.cos(theta)

def track_length_barrel(p, eta, B, r_detector, q = 1):
    '''return the track length in the barrel detector
    p: momentum in GeV/c
    eta: pseudorapidity
    B: magnetic field in T
    r_detector: radius of the barrel detector in meters
    q: charge in units of the electron charge

    returns the track length in meters
    '''
    theta = eta_to_theta(eta)
    p_T, p_L = split_momentum(p, theta)

    # transverse part:
    r_track = radius(p_T, B, q)
    l_T = arc_length(r_track, r_detector)

    # longitudinal part:
    l_L = r_detector * 1/np.tan(theta)

    return np.sqrt(l_T**2 + l_L**2)


def momentum_from_pt_eta(pt, eta):
    '''return the total momentum from the transverse momentum and pseudorapidity
    pt: transverse momentum in GeV/c
    eta: pseudorapidity

    returns the total momentum in GeV/c
    '''
    return pt / np.sin(eta_to_theta(eta))

def theta_from_xy(x, y):
    '''return the polar angle theta from the x and y coordinates
    x, y: coordinates

    returns the polar angle in radians
    '''
    return np.arctan2(y, x) #i do not know why arctan2() requires y first



# In[7]:


epic_B = 1.7 #epic B solenoid field in T

#pfRICH geometry:
rich_z = -1.725 #m

rich_radius_min = rich_z * np.tan(eta_to_theta(-3.5)) #was 0.095 #m
rich_radius_max = rich_z * np.tan(eta_to_theta(-1.5)) #was 0.659 #m

print('radii:', rich_radius_min, rich_radius_max)



rich_theta_min = theta_from_xy(rich_z, rich_radius_min)
rich_theta_max = theta_from_xy(rich_z, rich_radius_max)

rich_eta_min = theta_to_eta(rich_theta_min)
rich_eta_max = theta_to_eta(rich_theta_max)

rich_pt_min = momentum_cutoff(rich_radius_min/1, 1.7) #conservative estimate
rich_pt_min2 = momentum_cutoff(rich_radius_min/2, 1.7) #not so conservative estimate

print('theta:',rich_theta_min, rich_theta_max)
print('eta:',rich_eta_min, rich_eta_max)
print('pt min:', rich_pt_min)

#time resolutions
sigma_tof = 0.02 #ns as estimate by Alexander in email 09/24
sigma_t0 = 0.02 #ns as used in TOF estimates

sigma_total = np.sqrt(sigma_tof**2 + sigma_t0**2)


# In[9]:


# xx = np.linspace(rich_eta_min, rich_eta_max, 100)

# #lower momentum cutoff vs. eta
# plt.plot(xx, momentum_from_pt_eta(rich_pt_min, xx))
# plt.plot(xx, momentum_from_pt_eta(rich_pt_min2, xx))


# # In[10]:


# xx = np.linspace(rich_eta_min, rich_eta_max, 100)

# #direct path length vs. eta

# plt.plot(xx, 1/np.cos(eta_to_theta(xx))*rich_z)


# # In[11]:


# xx = np.linspace(0.5, 10, 1000)

# #plot some time of flight numbers at fixed eta = -2.5 as function of momentum

# eta = -2.5

# tof_electron = vec_tof(xx, pdg_to_mass(11), 1/np.cos(eta_to_theta(eta))*rich_z*1000)
# tof_pion = vec_tof(xx, pdg_to_mass(-211), 1/np.cos(eta_to_theta(eta))*rich_z*1000)
# #tof_pion_arc = vec_tof(xx, pdg_to_mass(-211), arc_length(radius(xx, epic_B), 1/np.cos(eta_to_theta(eta))*rich_z)*1000) #this is wrong. would need to be arc length applied to transverse part of track and then added in quadrature to longitudinal part.
# tof_kaon = vec_tof(xx, pdg_to_mass(-321), 1/np.cos(eta_to_theta(eta))*rich_z*1000)
# tof_proton = vec_tof(xx, pdg_to_mass(2212), 1/np.cos(eta_to_theta(eta))*rich_z*1000)

# plt.plot(xx, tof_electron)
# plt.plot(xx, tof_pion)
# #plt.plot(xx, tof_pion_arc)
# plt.plot(xx, tof_kaon)
# plt.plot(xx, tof_proton)


# # In[12]:


# #3 sigma separation points for same fixed eta = -2.5

# nsigma_pi_K = (tof_kaon - tof_pion)/sigma_total
# nsigma_K_proton = (tof_proton - tof_kaon)/sigma_total
# nsigma_e_pion = (tof_pion - tof_electron)/sigma_total

# plt.axhline(y = 3, color = 'black', linestyle = '--', label = '3 sigma', alpha = 0.3)

# plt.plot(xx, nsigma_pi_K, label = r'$\pi/K$ at $\eta$= -2.5')  
# plt.plot(xx, nsigma_K_proton, label = r'$K/p$ at $\eta$= -2.5')
# plt.plot(xx, nsigma_e_pion, label = r'$e/\pi$ at $\eta$= -2.5')

# plt.axvline(x = xx[np.argmax(nsigma_pi_K<3)], color = 'C0', linestyle = '--')
# plt.axvline(x = xx[np.argmax(nsigma_K_proton<3)], color = 'C1', linestyle = '--')
# plt.axvline(x = xx[np.argmax(nsigma_e_pion<3)], color = 'C2', linestyle = '--')



# plt.legend()

# plt.xlabel(r'p')
# plt.ylabel(r'PID separation in units of $\sigma$')

# plt.yscale('log')

# plt.xlim(0, 10)

# display(xx[np.argmax(nsigma_pi_K<3)], xx[np.argmax(nsigma_K_proton<3)], xx[np.argmax(nsigma_e_pion<3)])


# In[20]:


def pid_3sigma_cutoff(pdg_light, pdg_heavy, eta):
    pp = np.linspace(0.5, 10, 10000)
    
    tof_light = vec_tof(pp, pdg_to_mass(pdg_light), 1/np.cos(eta_to_theta(eta))*rich_z*1000)
    tof_heavy = vec_tof(pp, pdg_to_mass(pdg_heavy), 1/np.cos(eta_to_theta(eta))*rich_z*1000)
    
    nsigma = (tof_heavy - tof_light)/sigma_total
    #print(eta, pp[np.argmax(nsigma<3)]) 
    return pp[np.argmax(nsigma<3)]

xx = np.linspace(rich_eta_min, rich_eta_max, 10)

pid_3sigma_cutoff_pi_K = [pid_3sigma_cutoff(-211, -321, eta) for eta in xx]
pid_3sigma_cutoff_K_proton = [pid_3sigma_cutoff(-321, 2212, eta) for eta in xx]
pid_3sigma_cutoff_e_pi = [pid_3sigma_cutoff(11, -211, eta) for eta in xx]

pid_lower_cutoff = momentum_from_pt_eta(rich_pt_min, xx)
pid_lower_cutoff2 = momentum_from_pt_eta(rich_pt_min2, xx)

# plt.plot(xx, pid_3sigma_cutoff_pi_K)
# plt.plot(xx, pid_3sigma_cutoff_K_proton)
# plt.plot(xx, pid_3sigma_cutoff_e_pi)

plt.figure()

plt.fill_between(xx, pid_lower_cutoff, pid_3sigma_cutoff_pi_K, alpha = 0.3, label = r'$\pi/K$') 
#plt.fill_between(xx, pid_lower_cutoff, pid_3sigma_cutoff_K_proton, alpha = 0.3, label = r'$K/p$')
#plt.fill_between(xx, pid_lower_cutoff, pid_3sigma_cutoff_e_pi, alpha = 0.3, label = r'$e/\pi$')

plt.legend()

plt.xlabel(r'$\eta$')   
plt.ylabel(r'3sigma cutoff p [GeV/c]')


# In[15]:


plt.figure()
plt.fill_between(xx, pid_lower_cutoff, pid_3sigma_cutoff_pi_K, alpha = 0.3, label = r'$\pi/K$') 

plt.yscale('log')
plt.ylim(0.1,10)
plt.savefig('pfrich_contour_log_pi_K.svg')

plt.yscale('linear')
plt.ylim(0,5)
plt.savefig('pfrich_contour_lin_pi_K.svg')


# In[16]:

plt.figure()
plt.fill_between(xx, pid_lower_cutoff, pid_3sigma_cutoff_K_proton, alpha = 0.3, label = r'$\pi/K$') 

plt.yscale('log')
plt.ylim(0.1,10)
plt.savefig('pfrich_contour_log_K_proton.svg')

plt.yscale('linear')
plt.ylim(0,5)
plt.savefig('pfrich_contour_lin_K_proton.svg')


# In[17]:

plt.figure()
plt.fill_between(xx, pid_lower_cutoff, pid_3sigma_cutoff_e_pi, alpha = 0.3, label = r'$\pi/K$') 

plt.yscale('log')
plt.ylim(0.1,10)
plt.savefig('pfrich_contour_log_e_pi.svg')

plt.yscale('linear')
plt.ylim(0,5)
plt.savefig('pfrich_contour_lin_e_pi.svg')


# In[ ]:




