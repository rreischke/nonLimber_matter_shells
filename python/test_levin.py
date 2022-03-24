from cProfile import label
import levinpower

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import UnivariateSpline


if __name__ == "__main__":
    pk = np.load("./../data/pk.npz")
    kernels = np.load("./../data/kernels.npz")
    background = np.load("./../data/background.npz")


# Prepare example input, this can be any corresponding lists and arrays from your code.
    k_pk = pk["k"]
    z_pk = pk["z"]
    power_spectrum = pk["pk_nl"].flatten()
    backgound_z = background["z"]
    background_chi = background["chi"]
    chi_kernels = kernels["chi_cl"]

# Prepare redshift distribution input
    z_edges = np.loadtxt("./../data/zlim.txt")
    z_of_chi = UnivariateSpline(background_chi, backgound_z, s=0, ext=1)
    dz_dchi = z_of_chi.derivative()
    nbins = len(z_edges) - 1
    new_kernel = np.zeros((nbins, len(chi_kernels)))
    for i in range(nbins):
        for j in range(len(chi_kernels)):
            if(z_of_chi(chi_kernels[j]) > z_edges[i] and z_of_chi(chi_kernels[j]) < z_edges[i+1]):
                new_kernel[i, j] = dz_dchi(chi_kernels[j])

    number_count = 1#new_kernel.shape[0]
    kernels = new_kernel.T


# Setup the class with precomputed bessel functions (take a few moments)
    lp = levinpower.LevinPower(False, number_count,
                               backgound_z, background_chi,
                               chi_kernels, kernels,
                               k_pk, z_pk, power_spectrum, True)

    lp.set_parameters(1000,100,20,10,30000,20,100,300)
    
    ell = np.arange(2, 3000, 1)
    t0 = time.time()
    # actually calculate the Cls, returns a list for galaxy clustering, ggl and cosmic shear
    # Each one is again a list of n_tomo_tracer_A (n_tomo_tracer_B+1)/2 entries with the length
    # len(ell). The (0,0) bin corresponds to 0 and the (0,2) bin to 2 etc.
    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)
    t1 = time.time()
    total = t1-t0
    print(total)
    lp.set_parameters(1000,100,20,10,30000,30,100,300)
    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)
    

    plt.plot(ell, Cl_gg[0])
    '''plt.plot(ell, Cl_gg[nbins])
    plt.plot(ell, Cl_gg[nbins + nbins - 1])
    plt.plot(ell, Cl_gg[nbins + nbins + nbins - 2 - 1])
    plt.plot(ell, Cl_gg[4*nbins - 3 - 2 - 1])
    plt.plot(ell, Cl_gg[5*nbins -4  - 3 - 2 - 1])

    '''
    # updating the kernls, spectrum, background (is the same here, but could change)
    # lp.init_splines(backgound_z, background_chi,
    #                chi_kernels, kernels, k_pk, z_pk, power_spectrum)
    #Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)
    #plt.plot(ell, Cl_gg[0], "--")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

