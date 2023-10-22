"""
Aim: Call the .backward() function on plasma parameters I want to optimize

Procedure:
1) Create a sample vdf (start with Maxwellian)
2) Attach requires_grad=True to plasma parameters
3) Call the forward model (fast_spectral_density_arbdist) on vdf
4) Create a spectrum to which I will do fitting (Refer to Bryan's code?)
5) Perform SGD (Ask Mark how)

Flow: Tuple of parameters to optimize -> Generate vdf from them -> Pass through forward model
-> Generate S'(k,w) -> Loss MSE (S', S) -> Call optimizer
"""

# Import torch repo
!pip install git+https://github.com/v-kothale/ThomsonScattering.git
!pip install lmfit
!pip install numba
!pip install numba_scipy
!pip install corner
!pip install emcee==3.0.0
!pip install numdifftools

!git clone https://github.com/v-kothale/ThomsonScattering.git
%cd /content/ThomsonScattering/plasmapy
%ls

from plasmapy.diagnostics import thomson
from plasmapy.diagnostics import sgd_thomson_torch

%matplotlib inline

import torch
from torch import Tensor
import torch.nn.functional as F

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from tqdm import trange

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import warnings

from lmfit import Parameters

import corner
import emcee

from plasmapy.formulary import Maxwellian_1D
import matplotlib.pyplot as plt
import astropy.constants as const

from scipy.special import expit, gamma

import h5py

# Generate Maxwellian vdfs

def maxwellian_e(v: torch.Tensor, T: torch.Tensor, vd: torch.Tensor):
    T_K = T * 11605
    vth = torch.sqrt(2 * 1.5156e7 * T_K)
    exp_term = -1. * torch.square(v - vd) / torch.square(vth)
    return torch.exp(exp_term) / torch.sqrt(torch.pi * torch.square(vth))    # Normalizes distribution

def maxwellian_H(v: torch.Tensor, T: torch.Tensor, vd: torch.Tensor):
    T_K = T * 11605
    vth = torch.sqrt(2 * 8.2544e3 * T_K)
    exp_term = -1. * torch.square(v - vd) / torch.square(vth)
    return torch.exp(exp_term) / torch.sqrt(torch.pi * torch.square(vth))    # Normalizes distribution

probe_wavelength = 532 * u.nm
epw_wavelengths =np.linspace(probe_wavelength.value - 100, probe_wavelength.value + 100, num=200) * u.nm
iaw_wavelengths =np.linspace(probe_wavelength.value - 3, probe_wavelength.value + 3, num=200) * u.nm
ve = np.linspace(-2e7, 2e7, 500)
vH = np.linspace(-4e5, 6e5, 500)
n = 4e18

notch = np.array([520, 540])

# Test Case to plot vdfs (Velocity distribution functions)

fe_in = maxwellian_e(torch.from_numpy(ve), torch.tensor([200]), torch.tensor([1e6]))
fH_in = maxwellian_H(torch.from_numpy(vH), torch.tensor([50]), torch.tensor([1e5]))
plt.plot(vH, fH_in)

# Generate S(k,w) to fit to aka targets

alpha, epw_Pw_in = sgd_thomson_torch.spectral_density_maxwellian(
    wavelengths = epw_wavelengths,
    notches = [notch] * u.nm,
    probe_wavelength = probe_wavelength,
    Te = 200 * u.eV,
    Ti = 50 * u.eV,
    n = n * u.cm ** -3,
    electron_vel = [1e6 * np.array([-1, 1, 0]) / np.sqrt(2)] * u.m / u.s,   # What is e_vd?
    ion_vel = [1e5 * np.array([-1, 1, 0])/ np.sqrt(2)]* u.m / u.s,
    scattered_power = True,
)

alpha, iaw_Pw_in = sgd_thomson_torch.spectral_density_maxwellian(
    wavelengths = iaw_wavelengths,
    probe_wavelength = probe_wavelength,
    Te = 200 * u.eV,
    Ti = 50 * u.eV,
    n = n * u.cm ** -3,
    electron_vel = [1e6 * np.array([-1, 1, 0])/ np.sqrt(2)]* u.m / u.s,
    ion_vel = [1e5 * np.array([-1, 1, 0])/ np.sqrt(2)]* u.m / u.s,
    ion_species = ["H-1 1+"],
    scattered_power = True,
)

# Noise
epw_Pw_in += np.random.normal(loc = 0, scale = 0.1 * max(epw_Pw_in), size = len(epw_wavelengths))
iaw_Pw_in += np.random.normal(loc = 0, scale = 0.1 * max(iaw_Pw_in), size = len(iaw_wavelengths))

# Targets
epw_target = epw_Pw_in
iaw_target = iaw_Pw_in

# Convert 'em to tensors
epw_target = torch.from_numpy(epw_target)
iaw_target = torch.from_numpy(iaw_target)

# Fitting Routine with PyTorch Algorithm

probe_wavelength_TORCH = 532

# Hard convert wavelengths to nm to match Bryan's code
epw_wavelengths_TORCH = torch.linspace(probe_wavelength_TORCH - 100, probe_wavelength_TORCH + 100, steps=200, dtype=torch.float64) * 1e-9  # * u.nm
iaw_wavelengths_TORCH = torch.linspace(probe_wavelength_TORCH - 3, probe_wavelength_TORCH + 3, steps=200, dtype=torch.float64) * 1e-9 # * u.nm
ve_TORCH = torch.linspace(-2e7, 2e7, 500, dtype=torch.float64)
vH_TORCH = torch.linspace(-4e5, 6e5, 500, dtype=torch.float64)

probe_wavelength_TORCH *= 1e-9 # u.nm

notch_TORCH = torch.tensor([[520, 540]])

# Generate Super-Gaussian vdfs compatible with PyTorch

def supergaussian_e_TORCH(v: Tensor, T: Tensor, vd: Tensor, p: Tensor):
    T_K = T * 11605
    vth = torch.sqrt(2 * 8.2544e3 * T_K * 1836)  # 1836 = Ratio of electron to proton mass
    exp_term = torch.abs((v - vd) / (vth))
    vdf_unnormalized = torch.exp(-1. * torch.pow(exp_term, p))
    return vdf_unnormalized / torch.trapz(vdf_unnormalized, v)

def supergaussian_H_TORCH(v: Tensor, T: Tensor, vd: Tensor, p: Tensor):
    T_K = T * 11605
    vth = torch.sqrt(2 * 8.2544e3 * T_K)
    exp_term = torch.abs((v - vd) / (vth))
    vdf_unnormalized = torch.exp(-1. * torch.pow(exp_term, p))
    return vdf_unnormalized / torch.trapz(vdf_unnormalized, v)

# Alternate set of params ("Better" guesses)
e_vd = torch.tensor([100], dtype = torch.float64, requires_grad=True)
i_vd = torch.tensor([1e5], dtype = torch.float64, requires_grad=True)

ln_e_T = torch.log(torch.tensor([200], dtype = torch.float64))
ln_e_T.requires_grad=True   # Initialized as log

ln_i_T = torch.log(torch.tensor([100], dtype = torch.float64))
ln_i_T.requires_grad=True   # Initialized as log

ln_n = torch.log(torch.tensor([3.85e24], dtype = torch.float64))
ln_n.requires_grad=True   # Initialized as log

# Apply exp to pass these as "normal" values to vdf functions
e_T = torch.exp(ln_e_T) # * 100
i_T = torch.exp(ln_i_T) # * 100
n = torch.exp(ln_n) # * 1e24

e_p = torch.tensor([2.], dtype = torch.float64, requires_grad=True)
i_p = torch.tensor([2.], dtype = torch.float64, requires_grad=True)

# Call the forward model function

wavelengths_TORCH=epw_wavelengths_TORCH   # Switch off with IAW
probe_wavelength_TORCH = torch.tensor(probe_wavelength_TORCH, dtype=torch.float64)
e_velocity_axes_TORCH = torch.tensor(ve_TORCH, dtype=torch.float64)
i_velocity_axes_TORCH = torch.tensor(vH_TORCH, dtype=torch.float64)

efn_TORCH = maxwellian_e(e_velocity_axes_TORCH, torch.exp(ln_e_T), e_vd) # torch.tensor(fe_in_TORCH, dtype=torch.float64)
ifn_TORCH = maxwellian_H(i_velocity_axes_TORCH, torch.exp(ln_i_T), i_vd) # torch.tensor(fH_in_TORCH, dtype=torch.float64)

# print("efn_TORCH:", efn_TORCH)

n_TORCH = torch.exp(ln_n)
# n_TORCH *= 1e6

efract_TORCH = torch.tensor([1.0], dtype=torch.float64)
ifract_TORCH = torch.tensor([1.0], dtype=torch.float64)
ion_z_TORCH=torch.tensor([1], dtype=torch.float64)
ion_m_TORCH=torch.tensor([1], dtype=torch.float64)
notches_TORCH = notch_TORCH * 1e-9
probe_vec_TORCH = torch.tensor([1, 0, 0], dtype=torch.float64)
scatter_vec_TORCH = torch.tensor([0, 1, 0], dtype=torch.float64)
scattered_power_TORCH = True
# ion_species_TORCH = ['p']

"""
# Track gradients
efn_TORCH.requires_grad=True
ifn_TORCH.requires_grad=True
n_TORCH.requires_grad=True
"""

# print("ln_e_T:", ln_e_T)

thomson_alpha, Skw_arbdist = sgd_thomson_torch.fast_spectral_density_arbdist(
    wavelengths_TORCH,
    probe_wavelength_TORCH,
    e_velocity_axes_TORCH,
    i_velocity_axes_TORCH,
    efn_TORCH,
    ifn_TORCH,
    n_TORCH,
    notches_TORCH,
    efract_TORCH,
    ifract_TORCH,
    ion_z_TORCH,
    ion_m_TORCH,
    probe_vec_TORCH,
    scatter_vec_TORCH,
    scattered_power_TORCH,
    inner_range=0.1,
    inner_frac=0.8,
)

# Compute loss function
def dist_score(Skw_Prime, Skw):
    loss = F.mse_loss(Skw_Prime, Skw)
    loss.backward()
    return loss

# print(Skw_arbdist)

# Alternate set of params ("Better" guesses)
e_vd = torch.tensor([1e6], dtype = torch.float64, requires_grad=True)
i_vd = torch.tensor([1e5], dtype = torch.float64, requires_grad=True)

ln_e_T = torch.log(torch.tensor([100], dtype = torch.float64))
ln_e_T.requires_grad=True   # Initialized as log

ln_i_T = torch.log(torch.tensor([100], dtype = torch.float64))
ln_i_T.requires_grad=True   # Initialized as log

ln_n = torch.log(torch.tensor([3.85e24], dtype = torch.float64))
ln_n.requires_grad=True   # Initialized as log

"""
ln_i_T = torch.tensor([torch.log(torch.tensor([100]))], dtype = torch.float64, requires_grad=True)
ln_n = torch.tensor([torch.log(torch.tensor([3e24]))], dtype = torch.float64, requires_grad=True)
"""

# Apply exp to pass these as "normal" values to vdf functions
e_T = torch.exp(ln_e_T) # * 100
i_T = torch.exp(ln_i_T) # * 100
n = torch.exp(ln_n) # * 1e24

opt = torch.optim.NAdam((ln_e_T, e_vd, ln_i_T, i_vd, ln_n),lr=1e-3, betas=[0.9985, 0.999])

efn_TORCH = supergaussian_e_TORCH(e_velocity_axes_TORCH, torch.exp(ln_e_T), e_vd, e_p)
ifn_TORCH = supergaussian_H_TORCH(i_velocity_axes_TORCH, torch.exp(ln_i_T), i_vd, i_p)

# print("efn_TORCH:", efn_TORCH)

n_TORCH = torch.exp(ln_n)
# n_TORCH *= 1e6

efn_TORCH = maxwellian_e(e_velocity_axes_TORCH, torch.exp(ln_e_T), e_vd) # torch.tensor(fe_in_TORCH, dtype=torch.float64)
ifn_TORCH = maxwellian_H(i_velocity_axes_TORCH, torch.exp(ln_i_T), i_vd) # torch.tensor(fH_in_TORCH, dtype=torch.float64)

thomson_alpha_epw, Skw_arbdist_epw = sgd_thomson_torch.fast_spectral_density_arbdist(
    wavelengths_TORCH,
    probe_wavelength_TORCH,
    e_velocity_axes_TORCH,
    i_velocity_axes_TORCH,
    efn_TORCH,
    ifn_TORCH,
    n_TORCH,
    notches_TORCH,
    efract_TORCH,
    ifract_TORCH,
    ion_z_TORCH,
    ion_m_TORCH,
    probe_vec_TORCH,
    scatter_vec_TORCH,
    scattered_power_TORCH,
    inner_range=0.1,
    inner_frac=0.8,
)

dist_hist = [Skw_arbdist_epw.detach().clone()]

loss_hist = [float(dist_score(Skw_arbdist_epw,epw_target).detach())]
print(loss_hist)
for i in trange(4000):    # Can reduce this to 3500 for no noise fitting
    opt.zero_grad()

    # Reconstruct efn and ifn using updated Ts and vds
    efn_TORCH = maxwellian_e(e_velocity_axes_TORCH, torch.exp(ln_e_T), e_vd)
    ifn_TORCH = maxwellian_H(i_velocity_axes_TORCH, torch.exp(ln_i_T), i_vd)
    n_TORCH = torch.exp(ln_n)
    # print(n_TORCH)

    _, Skw_arbdist_epw = sgd_thomson_torch.fast_spectral_density_arbdist(
    wavelengths_TORCH,
    probe_wavelength_TORCH,
    e_velocity_axes_TORCH,
    i_velocity_axes_TORCH,
    efn_TORCH,
    ifn_TORCH,
    n_TORCH,
    notches_TORCH,
    efract_TORCH,
    ifract_TORCH,
    ion_z_TORCH,
    ion_m_TORCH,
    probe_vec_TORCH,
    scatter_vec_TORCH,
    scattered_power_TORCH,
    inner_range=0.1,
    inner_frac=0.8,
    )

    loss = dist_score(Skw_arbdist_epw,epw_target)
    opt.step()

    dist_hist.append(Skw_arbdist_epw.detach().clone())
    loss_hist.append(float(loss.detach()))

plt.plot(loss_hist)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("EPW Loss History")

plt.plot(epw_wavelengths_TORCH.numpy(), epw_target, label = "Theoretical Spectrum")
plt.plot(epw_wavelengths_TORCH.numpy(), Skw_arbdist_epw.detach().numpy(), label = "NAdam Fit")
plt.legend(loc = "upper right")
plt.xlabel("Wavelength (m)")
plt.ylabel("Spectral Density")
plt.title("EPW Spectrum")
plt.show()

efn_final = maxwellian_e(e_velocity_axes_TORCH, torch.exp(ln_e_T), e_vd)

plt.scatter(ve, fe_in, label = "Synthetic VDF", s=15)
plt.plot(ve, efn_final.detach().numpy(), label = "NAdam VDF", color="orange")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Probability Density")
plt.title("Electron VDF")
plt.legend(loc="upper right")
plt.show()

# Print optimized EPW parameters

e_vd_optim = e_vd
e_T_optim = torch.exp(ln_e_T)
n_optim = n_TORCH

print("e_vd:", e_vd_optim)
print("e_T:", e_T_optim)
print("i_vd:", i_vd)
print("i_T:", torch.exp(ln_i_T))
print("n:", n_optim)

# Redefining IAW params just in case...

# Detach optimized EPW params
e_vd_fit = e_vd_optim.detach()
e_T_fit = e_T_optim.detach()
n_fit = n_optim.detach()

opt_iaw = torch.optim.NAdam((ln_i_T, i_vd),lr=1e-3, betas=[0.9985, 0.999])

# Use optimized EPW params for temperatures
ln_e_T = torch.log(e_T_fit)

# Final changes for IAW
wavelengths_TORCH = iaw_wavelengths_TORCH
notches_TORCH = None

efn_TORCH_iaw = maxwellian_e(e_velocity_axes_TORCH, torch.exp(ln_e_T), e_vd_fit) # torch.tensor(fe_in_TORCH, dtype=torch.float64)
# efn_TORCH_iaw = maxwellian_e(e_velocity_axes_TORCH, torch.tensor([200.]), torch.tensor([1006716.]))
ifn_TORCH_iaw = maxwellian_H(i_velocity_axes_TORCH, torch.exp(ln_i_T), i_vd) # torch.tensor(fH_in_TORCH, dtype=torch.float64)
# ifn_TORCH_iaw = maxwellian_H(i_velocity_axes_TORCH, torch.tensor([50.]), torch.tensor([100429.]))

# print("i_velocity_axes_TORCH:",  i_velocity_axes_TORCH)

thomson_alpha_iaw, Skw_arbdist_iaw = sgd_thomson_torch.fast_spectral_density_arbdist(
    wavelengths_TORCH,
    probe_wavelength_TORCH,
    e_velocity_axes_TORCH,
    i_velocity_axes_TORCH,
    efn_TORCH_iaw,
    ifn_TORCH_iaw,
    n_fit,
    notches_TORCH,
    efract_TORCH,
    ifract_TORCH,
    ion_z_TORCH,
    ion_m_TORCH,
    probe_vec_TORCH,
    scatter_vec_TORCH,
    scattered_power_TORCH,
    inner_range=0.1,
    inner_frac=0.8,
)

plt.plot(iaw_wavelengths_TORCH, Skw_arbdist_iaw.detach().numpy())
plt.plot(iaw_wavelengths_TORCH, iaw_target, label = "Theoretical Spectrum")
plt.legend(loc = "upper right")
plt.show()

dist_hist_iaw = [Skw_arbdist_iaw.detach().clone()]

loss_hist_iaw = [float(dist_score(Skw_arbdist_iaw,iaw_target).detach())]
print(loss_hist_iaw)

for i in trange(4000):
    opt_iaw.zero_grad()

    # Reconstruct efn and ifn using updated Ts and vds
    efn_TORCH_iaw = maxwellian_e(e_velocity_axes_TORCH, torch.exp(ln_e_T), e_vd_fit)
    # efn_TORCH_iaw = maxwellian_e(e_velocity_axes_TORCH, torch.tensor([200.]), torch.tensor([1006716.]))
    ifn_TORCH_iaw = maxwellian_H(i_velocity_axes_TORCH, torch.exp(ln_i_T), i_vd)
    # ifn_TORCH_iaw = maxwellian_H(i_velocity_axes_TORCH, torch.tensor([50.]), torch.tensor([100429.]))
    # n_TORCH = torch.exp(ln_n)
    # print(n_TORCH)

    _, Skw_arbdist_iaw = sgd_thomson_torch.fast_spectral_density_arbdist(
    wavelengths_TORCH,
    probe_wavelength_TORCH,
    e_velocity_axes_TORCH,
    i_velocity_axes_TORCH,
    efn_TORCH_iaw,
    ifn_TORCH_iaw,
    n_fit,
    notches_TORCH,
    efract_TORCH,
    ifract_TORCH,
    ion_z_TORCH,
    ion_m_TORCH,
    probe_vec_TORCH,
    scatter_vec_TORCH,
    scattered_power_TORCH,
    inner_range=0.1,
    inner_frac=0.8,
    )

    loss_iaw = dist_score(Skw_arbdist_iaw,iaw_target)
    opt_iaw.step()

    dist_hist_iaw.append(Skw_arbdist_iaw.detach().clone())
    loss_hist_iaw.append(float(loss_iaw.detach()))

plt.plot(loss_hist_iaw) # [-100:])
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("IAW Loss History")

plt.plot(iaw_wavelengths_TORCH.numpy(), iaw_target, label = "Theoretical Spectrum")
plt.plot(iaw_wavelengths_TORCH.numpy(), Skw_arbdist_iaw.detach().numpy(), label = "NAdam Fit")
plt.legend(loc = "upper right")
plt.xlabel("Wavelength (m)")
plt.ylabel("Spectral Density")
plt.title("IAW Spectrum")
plt.show()

ifn_final = maxwellian_H(i_velocity_axes_TORCH, torch.exp(ln_i_T), i_vd)

plt.scatter(vH, fH_in, label="Synthetic VDF", s=15)
plt.plot(vH, ifn_final.detach().numpy(), label="NAdam VDF", color="orange")
plt.legend(loc="upper right")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Probability Density")
plt.title("Ion VDF")
plt.show()

# Print optimized IAW parameters

i_vd_fit = i_vd
i_T_fit = torch.exp(ln_i_T)

print("e_vd:", e_vd_fit)
print("e_T:", e_T_fit)
print("i_vd:", i_vd_fit)
print("i_T:", i_T_fit)
print("n:", n_fit)
