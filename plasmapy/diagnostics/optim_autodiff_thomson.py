"""
Defines the Thomson scattering analysis module as
part of the diagnostics package.
"""

__all__ = [
    "spectral_density_maxwellian",
    "spectral_density_arbdist",
    "scattered_power_model_maxwellian",
    "scattered_power_model_arbdist",
]

# Install torch dependencies
import torch
import torch.nn.functional as F

import astropy.constants as const
import astropy.units as u
import inspect
import numpy as np
import re
import warnings

from lmfit import Model
from typing import List, Tuple, Union, Optional    # Imported Optional

from plasmapy.formulary.dielectric import fast_permittivity_1D_Maxwellian
from plasmapy.formulary.parameters import fast_plasma_frequency, fast_thermal_speed
from plasmapy.particles import Particle, particle_mass
from plasmapy.utils.decorators import validate_quantities

# Make default torch tensor type
torch.set_default_dtype(torch.double)

_c = const.c.si.value  # Make sure C is in SI units
_e = const.e.si.value
_m_p = const.m_p.si.value
_m_e = const.m_e.si.value

@torch.jit.script
def derivative(f: torch.Tensor, x: torch.Tensor, derivative_matrices: List[torch.Tensor], order: int):
    dx = x[1]-x[0]

    order1_mat = derivative_matrices[0]
    order2_mat = derivative_matrices[1]

    if order == 1:
        f = (1./dx)*torch.matmul(order1_mat, f) # Use matrix for 1st order derivatives
        return f
    elif order == 2:
        f = (1./dx**2)*torch.matmul(order2_mat, f) # Use matrix for 1st order derivatives
        return f
    else:
        print("You can only choose an order of 1 or 2...")

@torch.jit.script
# Interpolation Function from Lars Du (end of thread): https://github.com/pytorch/pytorch/issues/1552
def torch_1d_interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: Optional[float] = None, #| None = None,
    right: Optional[float] = None #| None = None,
) -> torch.Tensor:

    """
    One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: 1d sequence of floats. x-coordinates. Must be increasing
        fp: 1d sequence of floats. y-coordinates. Must be same length as xp
        left: Value to return for x < xp[0], default is fp[0]
        right: Value to return for x > xp[-1], default is fp[-1]

    Returns:
        The interpolated values, same shape as x.
    """

    if left is None:
        left = fp[0]

    if right is None:
        right = fp[-1]

    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)

    answer = torch.where(
        x < xp[0],
        left,
        (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1]),
    )

    answer = torch.where(x > xp[-1], right, answer)
    return answer

def chi(
    f,
    derivative_matrices,
    u_axis,
    k,
    xi,
    v_th,
    n,
    particle_m,
    particle_q,
    phi=1e-5,
    nPoints=1e3,
    inner_range=0.3,
    inner_frac=0.8,
):
    """
    f: array, distribution function of velocities
    u_axis: normalized velocity axis
    k: wavenumber
    xi: normalized phase velocities
    v_th: thermal velocity of the distribution, used to normalize the velocity axis
    n: ion density
    m: particle mass in atomic mass units
    q: particle charge in fundamental charges
    phi: standoff variable used to avoid singularities
    nPoints: number of points used in integration
    deltauMax: maximum distance on the u axis to integrate to
    """

    # Take f' = df/du and f" = d^2f/d^2u
    fPrime = derivative(f=f, x=u_axis, derivative_matrices=derivative_matrices, order=1)
    fDoublePrime = derivative(f=f, x=u_axis, derivative_matrices=derivative_matrices, order=2)

    # Interpolate f' and f" onto xi
    g = torch_1d_interp(xi, u_axis, fPrime)
    gPrime = torch_1d_interp(xi, u_axis, fDoublePrime)

    # Set up integration ranges and spacing
    # We need fine divisions near the asymtorchote, but not at infinity

    """
    the fractional range of the inner fine divisions near the asymtorchote
    inner_range = 0.1
    the fraction of total divisions used in the inner range; should be > inner_range
    inner_frac = 0.8
    """

    outer_frac = torch.tensor([1.]) - inner_frac

    m_inner = torch.linspace(0, inner_range, int(torch.floor(torch.tensor([nPoints / 2 * inner_frac]))))
    p_inner = torch.linspace(0, inner_range, int(torch.ceil(torch.tensor([nPoints / 2 * inner_frac]))))
    m_outer = torch.linspace(inner_range, 1, int(torch.floor(torch.tensor([nPoints / 2 * outer_frac]))))
    p_outer = torch.linspace(inner_range, 1, int(torch.ceil(torch.tensor([nPoints / 2 * outer_frac]))))

    m = torch.concatenate((m_inner, m_outer))
    p = torch.concatenate((p_inner, p_outer))

    # Generate integration sample points that avoid the singularity
    # Create empty arrays of the correct size
    zm = torch.zeros((len(xi), len(m)))
    zp = torch.zeros((len(xi), len(p)))
    
    # Compute maximum width of integration range based on the size of the input array of normalized velocities
    deltauMax = max(u_axis) - min(u_axis)
    # print("deltauMax:", deltauMax)

    # Compute arrays of offsets to add to the central points in xi
    m_point_array = torch.tensor(phi + m * deltauMax)
    p_point_array = torch.tensor(phi + p * deltauMax)

    m_deltas = torch.concatenate((torch.tensor(torch.tensor(m_point_array[1:]) - torch.tensor(m_point_array[:-1])),
                               torch.tensor([0.])))

    p_deltas = torch.concatenate((torch.tensor(torch.tensor(p_point_array[1:]) - torch.tensor(p_point_array[:-1])),
                               torch.tensor([0.])))

    # The integration points on u
    for i in range(len(xi)):
        zm[i, :] = xi[i] + m_point_array
        zp[i, :] = xi[i] - p_point_array

    gm = torch_1d_interp(zm, u_axis, fPrime)
    gp = torch_1d_interp(zp, u_axis, fPrime)
    
    # Evaluate integral (df/du / (u - xi)) du
    M_array = m_deltas * gm / m_point_array
    P_array = p_deltas * gp / p_point_array

    integral = (
        torch.sum(M_array, axis=1)
        - torch.sum(P_array, axis=1)
        + 1j * torch.pi * g
        + 2 * phi * gPrime
    )

    # Convert mass and charge to SI units
    m_SI = torch.tensor([particle_m * 1.6605e-27])
    q_SI = torch.tensor([particle_q * 1.6022e-19])
    
    # Compute plasma frequency squared
    wpl2 = n * torch.square(q_SI) / (m_SI * 8.8541878e-12)

    # Coefficient
    v_th = torch.tensor([v_th])
    coefficient = -1. * wpl2 / k ** 2 / (torch.sqrt(torch.tensor([2])) * v_th)
    
    return coefficient * integral

def fast_spectral_density_arbdist(
    wavelengths,
    probe_wavelength,
    e_velocity_axes,
    i_velocity_axes,
    efn,
    ifn,
    derivative_matrices,
    n,
    notches = None,
    efract = torch.tensor([1.0], dtype=torch.float64),
    ifract = torch.tensor([1.0], dtype=torch.float64),
    ion_z=torch.tensor([1], dtype=torch.float64),
    ion_m=torch.tensor([1], dtype=torch.float64),
    probe_vec=torch.tensor([1, 0, 0]),
    scatter_vec=torch.tensor([0, 1, 0]),
    scattered_power=True,
    inner_range=0.1,
    inner_frac=0.8
):

    # Ensure unit vectors are normalized
    probe_vec = probe_vec / torch.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / torch.linalg.norm(scatter_vec)

    # Normal vector along k, assume all velocities lie in this direction
    k_vec = torch.tensor(scatter_vec - probe_vec)
    k_vec = k_vec / torch.linalg.norm(k_vec)  # normalization

    # Compute drift velocities and thermal speeds for all electrons and ion species
    electron_vel = torch.tensor([])  # drift velocities (vector)
    electron_vel_1d = torch.tensor([]) # 1D drift velocities (scalar)
    vTe = torch.tensor([])  # thermal speeds (scalar)

    # Note that we convert to SI, strip units, then reintroduce them outside the loop to get the correct objects
    for i, fn in enumerate(efn):
        v_axis = e_velocity_axes[i]
        moment1_integrand = torch.multiply(fn, v_axis)
        bulk_velocity = torch.trapz(moment1_integrand, v_axis)
        moment2_integrand = torch.multiply(fn, torch.square(v_axis - bulk_velocity))

        electron_vel = torch.concatenate((electron_vel, bulk_velocity * k_vec / torch.linalg.norm(k_vec)))
        electron_vel_1d = torch.concatenate((electron_vel_1d, torch.tensor([bulk_velocity])))
        vTe = torch.concatenate((vTe, torch.tensor([torch.sqrt(torch.trapz(moment2_integrand, v_axis))])))

    electron_vel = torch.reshape(electron_vel, (len(efn), 3))

    ion_vel = torch.tensor([])
    ion_vel_1d = torch.tensor([])
    vTi = torch.tensor([])

    for i, fn in enumerate(ifn):
        v_axis = i_velocity_axes[i]
        moment1_integrand = torch.multiply(fn, v_axis)
        bulk_velocity = torch.trapz(moment1_integrand, v_axis)
        moment2_integrand = torch.multiply(fn, torch.square(v_axis - bulk_velocity))

        ion_vel = torch.concatenate((ion_vel, bulk_velocity * k_vec / torch.linalg.norm(k_vec)))
        ion_vel_1d = torch.concatenate((ion_vel_1d, torch.tensor([bulk_velocity])))
        vTi = torch.concatenate((vTi, torch.tensor([torch.sqrt(torch.trapz(moment2_integrand, v_axis))])))

    ion_vel = torch.reshape(ion_vel, (len(ifn), 3))

    # Define some constants
    C = torch.tensor([299792458], dtype = torch.float64)  # speed of light

    # Calculate plasma parameters
    #zbar = torch.sum(ifract * ion_z)    # UNCOMMENT LINE 356
    zbar = torch.sum(torch.tensor(ifract) * ion_z)
    ne = efract * n
    ni = torch.tensor(ifract) * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    # wpe = plasma_frequency(n=n, particle="e-").to(u.rad / u.s).value
    n = torch.tensor(n * 3182.60735, dtype = torch.float64)
    wpe = torch.sqrt(n)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = torch.tensor(2 * torch.pi * C / wavelengths)
    wl = torch.tensor(2 * torch.pi * C / probe_wavelength)

    # Compute the frequency shift (required by energy conservation)
    w = torch.tensor(ws - wl)
    
    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = torch.sqrt((torch.square(ws) - torch.square(wpe))) / C
    kl = torch.sqrt((torch.square(wl) - torch.square(wpe))) / C

    # Compute the wavenumber shift (required by momentum conservation)
    scattering_angle = torch.arccos(torch.dot(probe_vec, scatter_vec))
    # Eq. 1.7.10 in Sheffield
    k = torch.sqrt((torch.square(ks) + torch.square(kl) - 2 * ks * kl * torch.cos(scattering_angle)))

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot product over all wavelengths
    # and ion components

    w_e = w - torch.matmul(electron_vel, torch.outer(k, k_vec).T)
    w_i = w - torch.matmul(ion_vel, torch.outer(k, k_vec).T)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = torch.sqrt(torch.tensor([2])) * wpe / torch.outer(k, vTe)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xie = (torch.outer(1 / vTe, 1 / k) * w_e) / torch.sqrt(torch.tensor([2]))
    xii = (torch.outer(1 / vTi, 1 / k) * w_i) / torch.sqrt(torch.tensor([2]))

    # Calculate the susceptibilities
    # Apply Sheffield (3.3.9) with the following substitutions
    # xi = w / (sqrt2 k v_th), u = v / (sqrt2 v_th)
    # Then chi = -w_pl ** 2 / (2 v_th ** 2 k ** 2) integral (df/du / (u - xi)) du

    # Electron susceptibilities
    chiE = torch.zeros((len(efract), len(w)), dtype=torch.complex128)
    for i in range(len(efract)):
        chiE[i, :] = chi(
            f=efn[i],
            derivative_matrices=derivative_matrices,
            u_axis=(
                e_velocity_axes[i] - electron_vel_1d[i]
            )
            / (torch.sqrt(torch.tensor(2)) * vTe[i]),
            k=k,
            xi=xie[i],
            v_th=vTe[i],
            n=ne[i],
            particle_m=5.4858e-4,
            particle_q=-1,
            inner_range = inner_range,
            inner_frac = inner_frac
        )

    # Ion susceptibilities
    chiI = torch.zeros((len(ifract), len(w)), dtype=torch.complex128)
    for i in range(len(ifract)):
        chiI[i, :] = chi(
            f=ifn[i],
            derivative_matrices=derivative_matrices,
            u_axis=(i_velocity_axes[i] - ion_vel_1d[i])
            / (torch.sqrt(torch.tensor([2])) * vTi[i]),
            k=k,
            xi=xii[i],
            v_th=vTi[i],
            n=ni[i],
            particle_m=ion_m[i],
            particle_q=ion_z[i],
            inner_range = inner_range,
            inner_frac = inner_frac
        )

    # Calculate the longitudinal dielectric function
    epsilon = 1 + torch.sum(chiE, axis=0) + torch.sum(chiI, axis=0)

    # Make a for loop to calculate and interplate necessary arguments ahead of time
    eInterp = torch.zeros((len(efract), len(w)), dtype=torch.complex128)
    for m in range(len(efract)):
        longArgE = (e_velocity_axes[m] - electron_vel_1d[m]) / (torch.sqrt(torch.tensor(2)) * vTe[m])
        eInterp[m] = torch_1d_interp(xie[m], longArgE, efn[m])

    # Electron component of Skw from Sheffield 5.1.2
    econtr = torch.zeros((len(efract), len(w)), dtype=torch.complex128)
    for m in range(len(efract)):
        econtr[m] = efract[m] * (
            2
            * torch.pi
            / k
            * torch.pow(torch.abs(1 - torch.sum(chiE, axis=0) / epsilon), 2)
            * eInterp[m]
        )

    iInterp = torch.zeros((len(ifract), len(w)), dtype=torch.complex128)
    for m in range(len(ifract)):
        longArgI = (i_velocity_axes[m] - ion_vel_1d[m]) / (torch.sqrt(torch.tensor(2)) * vTi[m])
        iInterp[m] = torch_1d_interp(xii[m], longArgI, ifn[m])

    # ion component
    icontr = torch.zeros((len(ifract), len(w)), dtype=torch.complex128)
    for m in range(len(ifract)):
        icontr[m] = ifract[m] * (
            2
            * torch.pi
            * ion_z[m]
            / k
            * torch.pow(torch.abs(torch.sum(chiE, axis=0) / epsilon), 2)
            * iInterp[m]
        )

    # Recast as real: imaginary part is already zero
    Skw = torch.real(torch.sum(econtr, axis=0) + torch.sum(icontr, axis=0))

    # Convert to power spectrum if otorchion is enabled
    if scattered_power:
        # Conversion factor
        Skw = Skw * (1 + 2 * w / wl) * 2 / (torch.square(wavelengths))
        #this is to convert from S(frequency) to S(wavelength), there is an
        #extra 2 * pi * c here but that should be removed by normalization

    # Work under assumption only EPW wavelengths require notch(es)
    if notches != None:
        # Account for notch(es) in differentiable manner
        bools = torch.ones(len(Skw), dtype = torch.bool)
        for i, j in enumerate(notches):
            if len(j) != 2:
                raise ValueError("Notches must be pairs of values")
            x0 = torch.argmin(torch.abs(wavelengths - j[0]))
            x1 = torch.argmin(torch.abs(wavelengths - j[-1]))
            bools[x0:x1] = False
        Skw = torch.mul(Skw, bools)

    # Normalize result to have integral 1
    Skw = Skw / torch.trapz(Skw, wavelengths)

    # print("S(k,w) after normalization:", Skw)  # UNCOMMENT TO GET SPECTRA AS TENSOR

    return torch.mean(alpha), Skw

def spectral_density_arbdist(
    wavelengths,
    probe_wavelength,
    e_velocity_axes,
    i_velocity_axes,
    efn,
    ifn,
    derivative_matrices,
    n,
    notches = None,
    efract = None,
    ifract = None,
    ion_species: Union[str, List[str], Particle, List[Particle]] = "p",
    probe_vec=torch.tensor([1, 0, 0]),
    scatter_vec=torch.tensor([0, 1, 0]),
    scattered_power=False,
    inner_range=0.1,
    inner_frac=0.8,
):
    
    if efract is None:
        efract = torch.ones(1)
    else:
        efract = torch.tensor(efract, dtype=torch.float64)

    if ifract is None:
        ifract = torch.ones(1)
        
    #Check for notches
    if notches is None:
        notches = torch.tensor([[520, 540]]) # * u.nm

    # Regarding conversion to SI, check with Mark and Derek about altering code to take np inputs
    # For now, assume all args passed as tensors
    
    # Condition ion_species
    if isinstance(ion_species, (str, Particle)):
        ion_species = [ion_species]
    if len(ion_species) == 0:
        raise ValueError("At least one ion species needs to be defined.")
    for ii, ion in enumerate(ion_species):
        if isinstance(ion, Particle):
            continue
        ion_species[ii] = Particle(ion)
    
    # Create arrays of ion Z and mass from particles given
    ion_z = torch.zeros(len(ion_species))
    ion_m = torch.zeros(len(ion_species))
    for i, particle in enumerate(ion_species):
        ion_z[i] = particle.charge_number
        ion_m[i] = ion_species[i].mass_number
        
    
    probe_vec = probe_vec / torch.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / torch.linalg.norm(scatter_vec)
    
    return fast_spectral_density_arbdist(
        wavelengths, 
        probe_wavelength, 
        e_velocity_axes, 
        i_velocity_axes, 
        efn, 
        ifn,
        derivative_matrices,
        n,
        notches,
        efract,
        ifract,
        ion_z,
        ion_m,
        probe_vec,
        scatter_vec,
        scattered_power,
        inner_range,
        inner_frac
        )

# ========== IGNORE EVERYTHING SOUTH OF HERE ==========

def fast_spectral_density_maxwellian(
    wavelengths,
    probe_wavelength,
    n,
    Te,
    Ti,
    notches: np.ndarray = None,
    efract: np.ndarray = np.array([1.0]),
    ifract: np.ndarray = np.array([1.0]),
    ion_z=np.array([1]),
    ion_m=np.array([1]),
    ion_vel=None,
    electron_vel=None,
    probe_vec=np.array([1, 0, 0]),
    scatter_vec=np.array([0, 1, 0]),
    inst_fcn_arr=None,
    scattered_power=False,
):

    """
    Te : np.ndarray
        Temperature in Kelvin
    """

    if electron_vel is None:
        electron_vel = np.zeros([efract.size, 3])

    if ion_vel is None:
        ion_vel = np.zeros([ifract.size, 3])
    
    if notches is None:
        notches = [(0, 0)]

    scattering_angle = np.arccos(np.dot(probe_vec, scatter_vec))

    # Calculate plasma parameters
    # Temperatures here in K!
    vTe = fast_thermal_speed(Te, _m_e)
    vTi = fast_thermal_speed(Ti, ion_m * _m_p)
    zbar = np.sum(ifract * ion_z)

    # Compute electron and ion densities
    ne = efract * n
    ni = ifract * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    wpe = fast_plasma_frequency(n, 1, _m_e)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = 2 * np.pi * _c / wavelengths
    wl = 2 * np.pi * _c / probe_wavelength

    # Compute the frequency shift (required by energy conservation)
    w = ws - wl

    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = np.sqrt(ws ** 2 - wpe ** 2) / _c
    kl = np.sqrt(wl ** 2 - wpe ** 2) / _c

    # Compute the wavenumber shift (required by momentum conservation)\
    # Eq. 1.7.10 in Sheffield
    k = np.sqrt(ks ** 2 + kl ** 2 - 2 * ks * kl * np.cos(scattering_angle))
    # Normal vector along k
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / np.linalg.norm(k_vec)

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot product over all wavelengths
    # and ion components
    w_e = w - np.matmul(electron_vel, np.outer(k, k_vec).T)
    w_i = w - np.matmul(ion_vel, np.outer(k, k_vec).T)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = np.sqrt(2) * wpe / np.outer(k, vTe)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xe = np.outer(1 / vTe, 1 / k) * w_e
    xi = np.outer(1 / vTi, 1 / k) * w_i

    # Calculate the susceptibilities
    chiE = np.zeros([efract.size, w.size], dtype=np.complex128)
    for i, fract in enumerate(efract):
        wpe = fast_plasma_frequency(ne[i], 1, _m_e)
        chiE[i, :] = fast_permittivity_1D_Maxwellian(w_e[i, :], k, vTe[i], wpe)

    # Treatment of multiple species is an extension of the discussion in
    # Sheffield Sec. 5.1
    chiI = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for i, fract in enumerate(ifract):
        wpi = fast_plasma_frequency(ni[i], ion_z[i], ion_m[i] * _m_p)
        chiI[i, :] = fast_permittivity_1D_Maxwellian(w_i[i, :], k, vTi[i], wpi)

    # Calculate the longitudinal dielectric function
    epsilon = 1 + np.sum(chiE, axis=0) + np.sum(chiI, axis=0)

    econtr = np.zeros([efract.size, w.size], dtype=np.complex128)
    for m in range(efract.size):
        econtr[m, :] = efract[m] * (
            2
            * np.sqrt(np.pi)
            / k
            / vTe[m]
            * np.power(np.abs(1 - np.sum(chiE, axis=0) / epsilon), 2)
            * np.exp(-xe[m, :] ** 2)
        )

    icontr = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for m in range(ifract.size):
        icontr[m, :] = ifract[m] * (
            2
            * np.sqrt(np.pi)
            * ion_z[m]
            / k
            / vTi[m]
            * np.power(np.abs(np.sum(chiE, axis=0) / epsilon), 2)
            * np.exp(-xi[m, :] ** 2)
        )

    # Recast as real: imaginary part is already zero
    Skw = np.real(np.sum(econtr, axis=0) + np.sum(icontr, axis=0))

    # Apply an insturment function if one is provided
    if inst_fcn_arr is not None:
        Skw = np.convolve(Skw, inst_fcn_arr, mode="same")

    if scattered_power:
        Skw = Skw * (1 + 2 * w / wl) * 2 / (wavelengths ** 2) 
    
    #Account for notch(es)
    for myNotch in notches:
        if len(myNotch) != 2:
            raise ValueError("Notches must be pairs of values")
            
        x0 = np.argmin(np.abs(wavelengths - myNotch[0]))
        x1 = np.argmin(np.abs(wavelengths - myNotch[1]))
        Skw[x0:x1] = 0
        
    Skw = Skw / np.trapz(Skw, wavelengths)

    return np.mean(alpha), Skw


@validate_quantities(
    wavelengths={"can_be_negative": False, "can_be_zero": False},
    probe_wavelength={"can_be_negative": False, "can_be_zero": False},
    n={"can_be_negative": False, "can_be_zero": False},
    Te={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    Ti={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)

def spectral_density_maxwellian(
    wavelengths: u.nm,
    probe_wavelength: u.nm,
    n: u.m ** -3,
    Te: u.K,
    Ti: u.K,
    notches: u.nm = None,
    efract: np.ndarray = None,
    ifract: np.ndarray = None,
    ion_species: Union[str, List[str], Particle, List[Particle]] = "p",
    electron_vel: u.m / u.s = None,
    ion_vel: u.m / u.s = None,
    probe_vec=np.array([1, 0, 0]),
    scatter_vec=np.array([0, 1, 0]),
    inst_fcn=None,
    scattered_power=False,
) -> Tuple[Union[np.floating, np.ndarray], np.ndarray]:
    r"""
    Calculate the spectral density function for Thomson scattering of a
    probe laser beam by a multi-species Maxwellian plasma.
    This function calculates the spectral density function for Thomson
    scattering of a probe laser beam by a plasma consisting of one or more ion
    species and a one or more thermal electron populations (the entire plasma
    is assumed to be quasi-neutral)
    .. math::
        S(k,\omega) = \sum_e \frac{2\pi}{k}
        \bigg |1 - \frac{\chi_e}{\epsilon} \bigg |^2
        f_{e0,e} \bigg (\frac{\omega}{k} \bigg ) +
        \sum_i \frac{2\pi Z_i}{k}
        \bigg |\frac{\chi_e}{\epsilon} \bigg |^2 f_{i0,i}
        \bigg ( \frac{\omega}{k} \bigg )
    where :math:`\chi_e` is the electron component suscetorchibility of the
    plasma and :math:`\epsilon = 1 + \sum_e \chi_e + \sum_i \chi_i` is the total
    plasma dielectric  function (with :math:`\chi_i` being the ion component
    of the suscetorchibility), :math:`Z_i` is the charge of each ion, :math:`k`
    is the scattering wavenumber, :math:`\omega` is the scattering frequency,
    and :math:`f_{e0,e}` and :math:`f_{i0,i}` are the electron and ion velocity
    distribution functions respectively. In this function the electron and ion
    velocity distribution functions are assumed to be Maxwellian, making this
    function equivalent to Eq. 3.4.6 in `Sheffield`_.
    Parameters
    ----------
    wavelengths : `~astropy.units.Quantity`
        Array of wavelengths over which the spectral density function
        will be calculated. (convertible to nm)
    probe_wavelength : `~astropy.units.Quantity`
        Wavelength of the probe laser. (convertible to nm)
    n : `~astropy.units.Quantity`
        Mean (0th order) density of all plasma components combined.
        (convertible to cm^-3.)
    Te : `~astropy.units.Quantity`, shape (Ne, )
        Temperature of each electron component. Shape (Ne, ) must be equal to the
        number of electron populations Ne. (in K or convertible to eV)
    Ti : `~astropy.units.Quantity`, shape (Ni, )
        Temperature of each ion component. Shape (Ni, ) must be equal to the
        number of ion populations Ni. (in K or convertible to eV)
    efract : array_like, shape (Ne, ), otorchional
        An array-like object where each element represents the fraction (or ratio)
        of the electron population number density to the total electron number density.
        Must sum to 1.0. Default is a single electron component.
    ifract : array_like, shape (Ni, ), otorchional
        An array-like object where each element represents the fraction (or ratio)
        of the ion population number density to the total ion number density.
        Must sum to 1.0. Default is a single ion species.
    ion_species : str or `~plasmapy.particles.Particle`, shape (Ni, ), otorchional
        A list or single instance of `~plasmapy.particles.Particle`, or strings
        convertible to `~plasmapy.particles.Particle`. Default is ``'H+'``
        corresponding to a single species of hydrogen ions.
    electron_vel : `~astropy.units.Quantity`, shape (Ne, 3), otorchional
        Velocity of each electron population in the rest frame. (convertible to m/s)
        If set, overrides electron_vdir and electron_speed.
        Defaults to a stationary plasma [0, 0, 0] m/s.
    ion_vel : `~astropy.units.Quantity`, shape (Ni, 3), otorchional
        Velocity vectors for each electron population in the rest frame
        (convertible to m/s). If set, overrides ion_vdir and ion_speed.
        Defaults zero drift for all specified ion species.
    probe_vec : float `~numpy.ndarray`, shape (3, )
        Unit vector in the direction of the probe laser. Defaults to
        ``[1, 0, 0]``.
    scatter_vec : float `~numpy.ndarray`, shape (3, )
        Unit vector pointing from the scattering volume to the detector.
        Defaults to [0, 1, 0] which, along with the default `probe_vec`,
        corresponds to a 90 degree scattering angle geometry.
    inst_fcn : function
        A function representing the instrument function that takes an `~astropy.units.Quantity`
        of wavelengths (centered on zero) and returns the instrument point
        spread function. The resulting array will be convolved with the
        spectral density function before it is returned.
    Returns
    -------
    alpha : float
        Mean scattering parameter, where `alpha` > 1 corresponds to collective
        scattering and `alpha` < 1 indicates non-collective scattering. The
        scattering parameter is calculated based on the total plasma density n.
    Skw : `~astropy.units.Quantity`
        Computed spectral density function over the input `wavelengths` array
        with units of s/rad.
    Notes
    -----
    For details, see "Plasma Scattering of Electromagnetic Radiation" by
    Sheffield et al. `ISBN 978\\-0123748775`_. This code is a modified version
    of the program described therein.
    For a concise summary of the relevant physics, see Chatorcher 5 of Derek
    Schaeffer's thesis, DOI: `10.5281/zenodo.3766933`_.
    .. _`ISBN 978\\-0123748775`: https://www.sciencedirect.com/book/9780123748775/plasma-scattering-of-electromagnetic-radiation
    .. _`10.5281/zenodo.3766933`: https://doi.org/10.5281/zenodo.3766933
    .. _`Sheffield`: https://doi.org/10.1016/B978-0-12-374877-5.00003-8
    """

    # Validate efract
    if efract is None:
        efract = np.ones(1)
    else:
        efract = np.asarray(efract, dtype=np.float64)

    # Validate ifract
    if ifract is None:
        ifract = np.ones(1)
    else:
        ifract = np.asarray(ifract, dtype=np.float64)

    if electron_vel is None:
        electron_vel = np.zeros([efract.size, 3]) * u.m / u.s

    # Condition the electron velocity keywords
    if ion_vel is None:
        ion_vel = np.zeros([ifract.size, 3]) * u.m / u.s
    
    if notches is None:
        notches = [(0, 0)] * u.nm

    # Condition ion_species
    if isinstance(ion_species, (str, Particle)):
        ion_species = [ion_species]
    if len(ion_species) == 0:
        raise ValueError("At least one ion species needs to be defined.")
    for ii, ion in enumerate(ion_species):
        if isinstance(ion, Particle):
            continue
        ion_species[ii] = Particle(ion)

    # Condition Ti
    if Ti.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If Ti.size != len(ion_species), assume same temp. for all species
        Ti = [Ti.value] * len(ion_species) * Ti.unit
    elif Ti.size != len(ion_species):
        raise ValueError(
            f"Got {Ti.size} ion temperatures and expected {len(ion_species)}."
        )

    # Make sure the sizes of ion_species, ifract, ion_vel, and Ti all match
    if (
        (len(ion_species) != ifract.size)
        or (ion_vel.shape[0] != ifract.size)
        or (Ti.size != ifract.size)
    ):
        raise ValueError(
            f"Inconsistent number of species in ifract ({ifract}), "
            f"ion_species ({len(ion_species)}), Ti ({Ti.size}), "
            f"and/or ion_vel ({ion_vel.shape[0]})."
        )

    # Condition Te
    if Te.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If Te.size != len(efract), assume same temp. for all species
        Te = [Te.value] * len(efract) * Te.unit
    elif Te.size != len(efract):
        raise ValueError(
            f"Got {Te.size} electron temperatures and expected {len(efract)}."
        )

    # Make sure the sizes of efract, electron_vel, and Te all match
    if (electron_vel.shape[0] != efract.size) or (Te.size != efract.size):
        raise ValueError(
            f"Inconsistent number of electron populations in efract ({efract.size}), "
            f"Te ({Te.size}), or electron velocity ({electron_vel.shape[0]})."
        )

    # Create arrays of ion Z and mass from particles given
    ion_z = np.zeros(len(ion_species))
    ion_m = np.zeros(len(ion_species))
    for i, particle in enumerate(ion_species):
        ion_z[i] = particle.charge_number
        ion_m[i] = particle.mass_number

    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)

    # Apply the insturment function
    if inst_fcn is not None and callable(inst_fcn):
        # Create an array of wavelengths of the same size as wavelengths
        # but centered on zero
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        inst_fcn_arr = inst_fcn(eval_w)
        inst_fcn_arr *= 1 / np.sum(inst_fcn_arr)
    else:
        inst_fcn_arr = None

    alpha, Skw = fast_spectral_density_maxwellian(
        wavelengths.to(u.m).value,
        probe_wavelength.to(u.m).value,
        n.to(u.m ** -3).value,
        Te.to(u.K).value,
        Ti.to(u.K).value,
        notches = notches.to(u.m).value,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_m = ion_m,
        ion_vel=ion_vel.to(u.m / u.s).value,
        electron_vel=electron_vel.to(u.m / u.s).value,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        inst_fcn_arr=inst_fcn_arr,
        scattered_power=scattered_power,
    )

    # Return output as PyTorch tensors
    alpha = torch.as_tensor(alpha)
    Skw = torch.as_tensor(Skw)

    print("Maxwellian S(k,w):", Skw)

    return torch.mean(alpha), Skw # * u.s / u.rad


# ***************************************************************************
# These functions are necessary to interface scalar Parameter objects with
# the array inputs of spectral_density
# ***************************************************************************


def _count_populations_in_params(params, prefix, allow_emtorchy = False):
    """
    Counts the number of entries matching the pattern prefix_i in a
    list of keys
    """
    
    keys = list(params.keys())
    prefixLength = len(prefix)
    
    if allow_emtorchy:
        nParams = 0
        for myKey in keys:
            if myKey[:prefixLength] == prefix:
                nParams = max(nParams, int(myKey[prefixLength + 1:]) + 1)
        
        return nParams
    
    else:
        return len(re.findall(prefix, ",".join(keys)))


def _params_to_array(params, prefix, vector=False, allow_emtorchy = False):
    """
    Takes a list of parameters and returns an array of the values corresponding
    to a key, based on the following naming convention:
    Each parameter should be named prefix_i
    Where i is an integer (starting at 0)
    This function allows lmfit.Parameter inputs to be converted into the
    array-type inputs required by the spectral density function
    """

    if vector:
        npop = _count_populations_in_params(params, prefix + "_x", allow_emtorchy = allow_emtorchy)
        output = np.zeros([npop, 3])
        for i in range(npop):
            for j, ax in enumerate(["x", "y", "z"]):
                if (prefix + f"_{ax}_{i}") in params:
                    output[i, j] = params[prefix + f"_{ax}_{i}"].value
                else:
                    output[i, j] = None

    else:
        npop = _count_populations_in_params(params, prefix, allow_emtorchy = allow_emtorchy)
        output = np.zeros([npop])
        for i in range(npop):
            if prefix + f"_{i}" in params:
                output[i] = params[prefix + f"_{i}"]

    return output

# ***************************************************************************
# Fitting functions
# ***************************************************************************


def _scattered_power_model_arbdist(wavelengths, settings=None, **params):
    """
    Non user-facing function for the lmfit model
    wavelengths: list of wavelengths over which scattered power is computed over
    settings: settings for the scattered power function
    eparams: parameters to put into emodel to generate a VDF
    iparams: parameters to put into imodel to generate a VDF
    """
    
    
    # check number of ion species
    
    if "ion_m" in settings:
        nSpecies = len(settings["ion_m"])
    else:
        nSpecies = 1
    
    # Separate params into electron params and ion params
    # Electron params must take the form e_paramName, where paramName is the name of the param in emodel
    # Ion params must take the form i_paramName, where paramName is the name of the param in imodel
    # The electron density n is just passed in as "n" and is treated separately from the other params
    # Velocity array is passed into settings
    eparams = {}
    iparams = [{} for _ in range(nSpecies)]

    # Extract crucial settings of emodel, imodel first
    emodel = settings["emodel"]
    imodel = settings["imodel"]
    
    ifract = _params_to_array(params, "ifract")
    
    
    #ion charges follow params if given, otherwise they are fixed at default values
    ion_z = np.zeros(nSpecies)
    for i in range(nSpecies):
        if "q_" + str(i) in params:
            ion_z[i] = params["q_" + str(i)]
        else:
            ion_z[i] = settings["ion_z"][i]
    
    for myParam in params.keys():
        
        myParam_split = myParam.split("_")
        
        if myParam_split[0] == "e":
            eparams[myParam_split[1]] = params[myParam]        
        elif (len(myParam_split[0])>0) and (myParam_split[0][0] == "i"):
            if myParam_split[0][1:].isnumeric():
                iparams[int(myParam_split[0][1:])][myParam_split[1]] = params[myParam]
        elif myParam_split[0] == "n":
            n = params[myParam]
        
    # Create VDFs from model functions
    ve = settings["e_velocity_axes"]
    vi = settings["i_velocity_axes"]

    fe = emodel(ve, **eparams)
    fi = [None] * nSpecies
    for i in range(nSpecies):
        fi[i] = imodel[i](vi[i], **(iparams[i]))

    # Remove emodel, imodel temporarily to put settings into the scattered power
    settings.pop("emodel")
    settings.pop("imodel")
    settings.pop("ion_z")

    # Call scattered power function
    alpha, model_Pw = fast_spectral_density_arbdist(
        wavelengths=wavelengths,
        n=n * 1e6, #this is so it accetorchs cm^-3 values by default
        efn=fe,
        ifn=fi,
        scattered_power=True,
        ifract = ifract,
        ion_z = ion_z,
        **settings,
    )

    # Put settings back now
    # this is necessary to avoid changing the settings array globally
    settings["emodel"] = emodel
    settings["imodel"] = imodel
    settings["ion_z"] = ion_z 

    return model_Pw

def _scattered_power_model_maxwellian(wavelengths, settings=None, **params):
    """
    lmfit Model function for fitting Thomson spectra
    For descritorchions of arguments, see the `thomson_model` function.
    """

    wavelengths_unitless = wavelengths.to(u.m).value

    # LOAD FROM SETTINGS
    notches = settings["notches"]
    ion_z = settings["ion_z"]
    ion_m = settings["ion_m"]
    probe_vec = settings["probe_vec"]
    scatter_vec = settings["scatter_vec"]
    electron_vdir = settings["electron_vdir"]
    ion_vdir = settings["ion_vdir"]
    probe_wavelength = settings["probe_wavelength"]
    inst_fcn_arr = settings["inst_fcn_arr"]

    # LOAD FROM PARAMS
    n = params["n"]
    Te = _params_to_array(params, "Te")
    Ti = _params_to_array(params, "Ti")
    efract = _params_to_array(params, "efract")
    ifract = _params_to_array(params, "ifract")

    electron_speed = _params_to_array(params, "electron_speed")
    ion_speed = _params_to_array(params, "ion_speed")

    electron_vel = electron_speed[:, np.newaxis] * electron_vdir
    ion_vel = ion_speed[:, np.newaxis] * ion_vdir

    # Convert temperatures from eV to Kelvin (required by fast_spectral_density)
    Te *= 11605
    Ti *= 11605
    
    #Convert density from cm^-3 to m^-3
    n *= 1e6

    alpha, model_Pw = fast_spectral_density_maxwellian(
        wavelengths_unitless,
        probe_wavelength,
        n,
        Te,
        Ti,
        notches = notches,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_m=ion_m,
        electron_vel=electron_vel,
        ion_vel=ion_vel,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        inst_fcn_arr=inst_fcn_arr,
        scattered_power=True,
    )

    return model_Pw

def scattered_power_model_arbdist(wavelengths, settings, params):
    """
    User facing fitting function, calls _scattered_power_model_arbdist to obtain lmfit model
    """

    

    # Extract crucial settings of emodel, imodel first

    if "emodel" in settings:
        emodel = settings["emodel"]
        print("emodel:", emodel)                # INSERTED PRINT STATEMENT HERE
    else:
        raise ValueError("Missing electron VDF model in settings")

    if "imodel" in settings:
        imodel = settings["imodel"]
        print("imodel:", imodel)                # INSERTED PRINT STATEMENT HERE
    else:
        raise ValueError("Missing ion VDF model in settings")
    
    if "ion_species" in settings:
        nSpecies = len(settings["ion_species"])
    else:
        settings["ion species"] = ["p"]
        nSpecies = 1
        
    if "ifract_0" not in list(params.keys()):
        params.add("ifract_0", value=1.0, vary=False)
        
    
    num_i = _count_populations_in_params(params, "ifract")
    
    if num_i > 1:
        nums = ["ifract_" + str(i) for i in range(num_i - 1)]
        nums.insert(0, "1.0")
        params["ifract_" + str(num_i - 1)].expr = " - ".join(nums)
        
    
    
    
    # Separate params into electron params and ion params
    # Electron params must take the form e_paramName, where paramName is the name of the param in emodel
    # Ion params must take the form i_paramName, where paramName is the name of the param in imodel
    # The electron density n is just passed in as "n" and is treated separately from the other params
    # Velocity array is passed into settings
    eparams = {}
    iparams = [{} for _ in range(nSpecies)]
        
    
    
    for myParam in params.keys():
        
        myParam_split = myParam.split("_")
        
        if myParam_split[0] == "e":
            eparams[myParam_split[1]] = params[myParam]
        elif myParam_split[0][0] == "i":
            if myParam_split[0][1:].isnumeric():
                iparams[int(myParam_split[0][1:])][myParam_split[1]] = params[myParam]
        elif myParam_split[0] == "n":
            n = params[myParam]
        elif myParam_split[0] == "z":
            z = params[myParam] #this line does nothing and is just to not have the next else trigger
        else:
            raise ValueError("Param name " + myParam + " invalid, must start with e or i")
        
    # Check that models have correct params as inputs
    
    

    # Param names from the model functions
    emodel_param_names = set(inspect.getfullargspec(emodel)[0])

    # Check if models take in velocity as an input -- this is ignored as a param
    if not ("v" in emodel_param_names):
        raise ValueError("Electron VDF model does not take velocity as input")
    
    emodel_param_names.remove("v")
    eparam_names = set(eparams.keys())
    # Raise errors if params are wrong
    if emodel_param_names != eparam_names:
        raise ValueError("Electron parameters do not match")
    
      
    for i in range(nSpecies):
        imodel_param_names = set(inspect.getfullargspec(imodel[i])[0])
        if not ("v" in imodel_param_names):
            raise ValueError("Ion VDF model does not take velocity as input")
        print("imodel_param_names:", imodel_param_names)            # INSERTED PRINT STATEMENT HERE
        imodel_param_names.remove("v")
        iparam_names = set(iparams[i].keys())
        
        if imodel_param_names != iparam_names:
            raise ValueError("Ion species " + str(i) + " parameters do not match")

    
    ion_species = settings["ion_species"]
    # Create arrays of ion Z and mass from particles given
    ion_z = np.zeros(nSpecies)
    ion_m = np.zeros(nSpecies)
    for i, species in enumerate(ion_species):
        particle = Particle(species)
        ion_z[i] = particle.charge_number
        ion_m[i] = particle.mass_number
    settings["ion_z"] = ion_z
    settings["ion_m"] = ion_m
    
    # Remove the ion_species from settings
    
    settings.pop("ion_species")

    model = Model(
        _scattered_power_model_arbdist,
        independent_vars=["wavelengths"],
        nan_policy="omit",
        settings=settings.copy(),
    )
    
    settings["ion_species"] = ion_species

    return model

def scattered_power_model_maxwellian(wavelengths, settings, params):
    """
    Returns a `lmfit.Model` function for Thomson spectral density function
    Parameters
    ----------
    wavelengths : u.Quantity
        Wavelength array
    settings : dict
        A dictionary of non-variable inputs to the spectral density function
        which must include the following:
            - probe_wavelength: Probe wavelength in nm
            - probe_vec : (3,) unit vector in the probe direction
            - scatter_vec: (3,) unit vector in the scattering direction
            - ion_species : list of Particle strings describing each ion species
        and may contain the following otorchional variables
            - electron_vdir : (e#, 3) array of electron velocity unit vectors
            - ion_vdir : (e#, 3) array of ion velocity unit vectors
            - inst_fcn : A function that takes a wavelength array and represents
                    a spectrometer insturment function.
        These quantities cannot be varied during the fit.
    params : `lmfit.Parameters` object
        A Parameters object that must contains the following variables
            - n: 0th order density in cm^-3
            - Te_e# : Temperature in eV
            - Ti_i# : Temperature in eV
        and may contain the following otorchional variables
            - efract_e# : Fraction of each electron population (must sum to 1) (otorchional)
            - ifract_i# : Fraction of each ion population (must sum to 1) (otorchional)
            - electron_speed_e# : Electron speed in m/s (otorchional)
            - ion_speed_i# : Ion speed in m/s (otorchional)
        where i# and e# are the number of electron and ion populations,
        zero-indexed, respectively (eg. 0,1,2...).
        These quantities can be either fixed or varying.
    Returns
    -------
    Spectral density (otorchimization function)
    """

    # **********************
    # Required settings and parameters
    # **********************
    req_settings = ["probe_wavelength", "probe_vec", "scatter_vec", "ion_species"]
    for k in req_settings:
        if k not in list(settings.keys()):
            raise KeyError(f"{k} was not provided in settings, but is required.")

    req_params = ["n"]
    for k in req_params:
        if k not in list(params.keys()):
            raise KeyError(f"{k} was not provided in parameters, but is required.")

    # **********************
    # Count number of populations
    # **********************
    if "efract_0" not in list(params.keys()):
        params.add("efract_0", value=1.0, vary=False)

    if "ifract_0" not in list(params.keys()):
        params.add("ifract_0", value=1.0, vary=False)

    num_e = _count_populations_in_params(params, "efract")
    num_i = _count_populations_in_params(params, "ifract")

    # **********************
    # Required settings and parameters per population
    # **********************
    req_params = ["Te"]
    for p in req_params:
        for e in range(num_e):
            k = p + "_" + str(e)
            if k not in list(params.keys()):
                raise KeyError(f"{p} was not provided in parameters, but is required.")

    req_params = ["Ti"]
    for p in req_params:
        for i in range(num_i):
            k = p + "_" + str(i)
            if k not in list(params.keys()):
                raise KeyError(f"{p} was not provided in parameters, but is required.")


    # Create arrays of ion Z and mass from particles given
    ion_z = np.zeros(num_i)
    ion_m= np.zeros(num_i)
    for i, species in enumerate(settings["ion_species"]):
        particle = Particle(species)
        ion_z[i] = particle.charge_number
        ion_m[i] = particle.mass_number
    settings["ion_z"] = ion_z
    settings["ion_m"] = ion_m
    

    # Automatically add an expression to the last efract parameter to
    # indicate that it depends on the others (so they sum to 1.0)
    # The resulting expression for the last of three will look like
    # efract_2.expr = "1.0 - efract_0 - efract_1"
    if num_e > 1:
        nums = ["efract_" + str(i) for i in range(num_e - 1)]
        nums.insert(0, "1.0")
        params["efract_" + str(num_e - 1)].expr = " - ".join(nums)

    if num_i > 1:
        nums = ["ifract_" + str(i) for i in range(num_i - 1)]
        nums.insert(0, "1.0")
        params["ifract_" + str(num_i - 1)].expr = " - ".join(nums)

    # **************
    # Electron velocity
    # **************
    electron_speed = np.zeros([num_e])
    for e in range(num_e):
        k = "electron_speed_" + str(e)
        if k in list(params.keys()):
            electron_speed[e] = params[k].value
        else:
            # electron_speed[e] = 0 already
            params.add(k, value=0, vary=False)

    if "electron_vdir" not in list(settings.keys()):
        if np.all(electron_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["electron_vdir"] = np.ones([num_e, 3])
        else:
            raise ValueError(
                "electron_vdir must be set if electron_speeds " "are not all zero."
            )
    # Normalize vdir
    norm = np.linalg.norm(settings["electron_vdir"], axis=-1)
    settings["electron_vdir"] = settings["electron_vdir"] / norm[:, np.newaxis]

    # **************
    # Ion velocity
    # **************
    ion_speed = np.zeros([num_i])
    for i in range(num_i):
        k = "ion_speed_" + str(i)
        if k in list(params.keys()):
            ion_speed[i] = params[k].value
        else:
            # ion_speed[i] = 0 already
            params.add(k, value=0, vary=False)

    if "ion_vdir" not in list(settings.keys()):
        if np.all(ion_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["ion_vdir"] = np.ones([num_i, 3])
        else:
            raise ValueError("ion_vdir must be set if ion_speeds " "are not all zero.")
    # Normalize vdir
    norm = np.linalg.norm(settings["ion_vdir"], axis=-1)
    settings["ion_vdir"] = settings["ion_vdir"] / norm[:, np.newaxis]

    if "inst_fcn" not in list(settings.keys()):
        settings["inst_fcn_arr"] = None
    else:
        # Create inst fcn array from inst_fcn
        inst_fcn = settings["inst_fcn"]
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        inst_fcn_arr = inst_fcn(eval_w)
        inst_fcn_arr *= 1 / np.sum(inst_fcn_arr)
        settings["inst_fcn_arr"] = inst_fcn_arr

    # Convert and strip units from settings if necessary
    val = {"probe_wavelength": u.m}
    for k, unit in val.items():
        if hasattr(settings[k], "unit"):
            settings[k] = settings[k].to(unit).value

    # TODO: raise an excetorchion if the number of any of the ion or electron
    # quantities isn't consistent with the number of that species defined
    # by ifract or efract.

    # Create a lmfit.Model
    # nan_policy='omit' automatically ignores NaN values in data, allowing those
    # to be used to represnt regions of missing data
    # the "settings" dict is an additional kwarg that will be passed to the model function on every call
    model = Model(
        _scattered_power_model_maxwellian,
        independent_vars=["wavelengths"],
        nan_policy="omit",
        settings=settings,
    )

    return model
