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
torch.set_default_dtype(torch.float64)

_c = const.c.si.value  # Make sure C is in SI units
_e = const.e.si.value
_m_p = const.m_p.si.value
_m_e = const.m_e.si.value

@torch.jit.script
def derivative(f: torch.Tensor, x: torch.Tensor, derivative_matrices: Tuple[torch.Tensor, torch.Tensor], order: int):
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
# Original interpolation function from Lars Du (end of thread): https://github.com/pytorch/pytorch/issues/1552
def torch_1d_interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    # left: Optional[float] = None, #| None = None,
    # right: Optional[float] = None #| None = None,
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
    
    left = fp[0]
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
    
    with torch.no_grad():
        outer_frac = torch.tensor([1.]) - inner_frac

        m_inner = torch.linspace(0, inner_range, int(torch.floor(torch.tensor([nPoints / 2 * inner_frac]))))
        p_inner = torch.linspace(0, inner_range, int(torch.ceil(torch.tensor([nPoints / 2 * inner_frac]))))
        m_outer = torch.linspace(inner_range, 1, int(torch.floor(torch.tensor([nPoints / 2 * outer_frac]))))
        p_outer = torch.linspace(inner_range, 1, int(torch.ceil(torch.tensor([nPoints / 2 * outer_frac]))))

        m = torch.cat((m_inner, m_outer))
        p = torch.cat((p_inner, p_outer))

        # Generate integration sample points that avoid the singularity
        # Create empty arrays of the correct size
        zm = torch.zeros((len(xi), len(m)))
        zp = torch.zeros((len(xi), len(p)))
    
        # Compute maximum width of integration range based on the size of the input array of normalized velocities
        deltauMax = max(u_axis) - min(u_axis)
        # print("deltauMax:", deltauMax)

        # Compute arrays of offsets to add to the central points in xi
        m_point_array = phi + m * deltauMax
        p_point_array = phi + p * deltauMax

        m_deltas = torch.cat((m_point_array[1:] - m_point_array[:-1], torch.tensor([0.])))
        p_deltas = torch.cat((p_point_array[1:] - p_point_array[:-1], torch.tensor([0.])))

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
    k_vec = scatter_vec - probe_vec
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

        electron_vel = torch.cat((electron_vel, bulk_velocity * k_vec / torch.linalg.norm(k_vec)))
        electron_vel_1d = torch.cat((electron_vel_1d, torch.tensor([bulk_velocity])))
        vTe = torch.cat((vTe, torch.tensor([torch.sqrt(torch.trapz(moment2_integrand, v_axis))])))

    electron_vel = torch.reshape(electron_vel, (len(efn), 3))

    ion_vel = torch.tensor([])
    ion_vel_1d = torch.tensor([])
    vTi = torch.tensor([])

    for i, fn in enumerate(ifn):
        v_axis = i_velocity_axes[i]
        moment1_integrand = torch.multiply(fn, v_axis)
        bulk_velocity = torch.trapz(moment1_integrand, v_axis)
        moment2_integrand = torch.multiply(fn, torch.square(v_axis - bulk_velocity))

        ion_vel = torch.cat((ion_vel, bulk_velocity * k_vec / torch.linalg.norm(k_vec)))
        ion_vel_1d = torch.cat((ion_vel_1d, torch.tensor([bulk_velocity])))
        vTi = torch.cat((vTi, torch.tensor([torch.sqrt(torch.trapz(moment2_integrand, v_axis))])))

    ion_vel = torch.reshape(ion_vel, (len(ifn), 3))

    # Define some constants
    C = torch.tensor([299792458], dtype = torch.float64)  # speed of light

    # Calculate plasma parameters
    zbar = torch.sum(ifract * ion_z)
    ne = efract * n
    ni = ifract * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    # wpe = plasma_frequency(n=n, particle="e-").to(u.rad / u.s).value
    n = n * 3182.60735
    wpe = torch.sqrt(n)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = 2 * torch.pi * C / wavelengths
    wl = 2 * torch.pi * C / probe_wavelength

    # Compute the frequency shift (required by energy conservation)
    w = ws - wl
    
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

    # Make a for loop to calculate and interpolate necessary arguments ahead of time
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

    if ifract is None:
        ifract = torch.ones(1)
        
    #Check for notches
    if notches is None:
        notches = torch.tensor([[520, 540]]) # * u.nm
    
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
