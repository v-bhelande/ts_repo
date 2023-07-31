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
import torch as pt
import torch.nn.functional as F

import astropy.constants as const
import astropy.units as u
import inspect
import numpy as np
import re
import warnings

from lmfit import Model
# from numba import jit
from typing import List, Tuple, Union

from plasmapy.formulary.dielectric import fast_permittivity_1D_Maxwellian
from plasmapy.formulary.parameters import fast_plasma_frequency, fast_thermal_speed
from plasmapy.particles import Particle, particle_mass
from plasmapy.utils.decorators import validate_quantities

_c = const.c.si.value  # Make sure C is in SI units
_e = const.e.si.value
_m_p = const.m_p.si.value
_m_e = const.m_e.si.value

import torch as pt

def derivative(f, x, order):
    dx = x[1]-x[0]
    # print("dx:", dx)

    # Assumes f is 1D
    finDiffMat = pt.zeros(len(f), len(f), dtype = pt.float64)

    # Fill in finite difference matrix

    # Convert input to torch tensor if needed
    if not pt.is_tensor(f):
      f = pt.as_tensor(f)
      # print(f)
    f.requires_grad = True

    if order == 1:
        # Forward difference elements
        finDiffMat[0][0] = -25./12.
        finDiffMat[0][1] = 4.
        finDiffMat[0][2] = -3.
        finDiffMat[0][3] = 4./3.
        finDiffMat[0][4] = -1./4.

        finDiffMat[1][1] = -25./12.
        finDiffMat[1][2] = 4.
        finDiffMat[1][3] = -3.
        finDiffMat[1][4] = 4./3.
        finDiffMat[1][5] = -1./4.

        # Backward difference elements
        finDiffMat[-1][-1] = 25./12.
        finDiffMat[-1][-2] = -4.
        finDiffMat[-1][-3] = 3.
        finDiffMat[-1][-4] = -4./3.
        finDiffMat[-1][-5] = 1./4.

        finDiffMat[-2][-2] = 25./12.
        finDiffMat[-2][-3] = -4.
        finDiffMat[-2][-4] = 3.
        finDiffMat[-2][-5] = -4./3.
        finDiffMat[-2][-6] = 1./4.

        # Centered difference elements
        for i in range(2, len(finDiffMat)-2):
          # print(i)
          finDiffMat[i][i-2] = 1./12.
          finDiffMat[i][i-1] = -8./12.
          finDiffMat[i][i] = 0.
          finDiffMat[i][i+1] = 8./12.
          finDiffMat[i][i+2] = -1./12.

        # Make sparse matrix
        # finDiffMat = finDiffMat.to_sparse()

        # print("finDiffMat:", finDiffMat)

        f = (1./dx)*pt.matmul(finDiffMat, f)

        return f

        # Returns f as np.array
        # return f.detach().numpy()

    elif order == 2:
        # Forward difference elements
        finDiffMat[0][0] = 15./4.
        finDiffMat[0][1] = -77./6.
        finDiffMat[0][2] = 107./6.
        finDiffMat[0][3] = -13.
        finDiffMat[0][4] = 61./12.
        finDiffMat[0][5] = -5./6.

        finDiffMat[1][1] = 15./4.
        finDiffMat[1][2] = -77./6.
        finDiffMat[1][3] = 107./6.
        finDiffMat[1][4] = -13.
        finDiffMat[1][5] = 61./12.
        finDiffMat[1][6] = -5./6.

        # Backward difference elements
        finDiffMat[-1][-1] = 15./4.
        finDiffMat[-1][-2] = -77./6.
        finDiffMat[-1][-3] = 107./6.
        finDiffMat[-1][-4] = -13.
        finDiffMat[-1][-5] = 61./12.
        finDiffMat[-1][-6] = -5./6.

        finDiffMat[-2][-2] = 15./4.
        finDiffMat[-2][-3] = -77./6.
        finDiffMat[-2][-4] = 107./6.
        finDiffMat[-2][-5] = -13.
        finDiffMat[-2][-6] = 61./12.
        finDiffMat[-2][-7] = -5./6.

        # Centered difference elements
        for i in range(2, len(finDiffMat)-2):
          # print(i)
          finDiffMat[i][i-2] = -1./12.
          finDiffMat[i][i-1] = 4./3.
          finDiffMat[i][i] = -5./2.
          finDiffMat[i][i+1] = 4./3.
          finDiffMat[i][i+2] = -1./12.

        # finDiffMat = finDiffMat.to_sparse()
        # print("finDiffMat:", finDiffMat)

        f = (1./dx**2)*pt.matmul(finDiffMat, f)

        return f
        
        # Returns f as np.array
        # return f.detach().numpy()
    else:
        print("You can only choose an order of 1 or 2...")

# Interpolation Function from Lars Du (end of thread): https://github.com/pytorch/pytorch/issues/1552

def torch_1d_interp(
    x: pt.Tensor,
    xp: pt.Tensor,
    fp: pt.Tensor,
    left: float | None = None,
    right: float | None = None,
) -> pt.Tensor:

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

    # MY CHANGES

    if not pt.is_tensor(fp):
      # print("Converting fp to tensor")
      fp = pt.as_tensor(fp)    # Convert to tensor
      # fp = fp.reshape(len(fp))    # TODO: Temporary fix, change this later
    # fp = fp.expand(3,-1)
    # fp = fp.reshape(len(fp))
    # print("fp:", fp)

    if not pt.is_tensor(xp):
      # print("Converting xp to tensor")
      xp = pt.as_tensor(xp)    # Convert to tensor
    # xp = xp.expand(3,-1)
    # print("xp:", xp)

    if not pt.is_tensor(x):
      # print("Converting x to tensor")
      x = pt.as_tensor(x)    # Convert to tensor
    # print("x:", x)

    if left is None:
        left = fp[0]

    if right is None:
        right = fp[-1]

    i = pt.clip(pt.searchsorted(xp, x, right=True), 1, len(xp) - 1)

    answer = pt.where(
        x < xp[0],
        left,
        (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1]),
    )

    answer = pt.where(x > xp[-1], right, answer)
    return answer

def chi(
    f,
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
    fPrime = derivative(f=f, x=u_axis, order=1)
    # print("fPrime:", fPrime)

    fDoublePrime = derivative(f=f, x=u_axis, order=2)
    # print("fDoublePrime:", fDoublePrime)

    # Interpolate f' and f" onto xi
    g = torch_1d_interp(xi, u_axis, fPrime)
    # print("g Interpolation:", g)

    gPrime = torch_1d_interp(xi, u_axis, fDoublePrime)
    # print("g' Interpolation:", gPrime)

    # CONVERT INPUTS TO TENSORS
    k = pt.tensor(k, dtype=pt.float64)
    v_th = pt.tensor(v_th, dtype=pt.float64)
    n = pt.tensor(n, dtype=pt.float64)
    particle_m = pt.tensor(particle_m, dtype=pt.float64)
    particle_q = pt.tensor(particle_q, dtype=pt.float64)
    phi=pt.tensor(phi, dtype=pt.float64)
    nPoints=pt.tensor(nPoints, dtype=pt.float64)
    inner_range=pt.tensor(inner_range, dtype=pt.float64)
    inner_frac=pt.tensor(inner_frac, dtype=pt.float64)

    # Set up integration ranges and spacing
    # We need fine divisions near the asymptote, but not at infinity

    """
    #the fractional range of the inner fine divisions near the asymptote
    inner_range = 0.1
    #the fraction of total divisions used in the inner range; should be > inner_range
    inner_frac = 0.8
    """

    outer_frac = pt.tensor(1.) - inner_frac

    m_inner = pt.linspace(0, inner_range, int(pt.floor(nPoints / 2 * inner_frac))) # Specify data type to pt.float64
    p_inner = pt.linspace(0, inner_range, int(pt.ceil(nPoints / 2 * inner_frac)))
    m_outer = pt.linspace(inner_range, 1, int(pt.floor(nPoints / 2 * outer_frac)))
    p_outer = pt.linspace(inner_range, 1, int(pt.ceil(nPoints / 2 * outer_frac)))

    m = pt.concatenate((m_inner, m_outer))
    p = pt.concatenate((p_inner, p_outer))

    # print("m.size:", m.size())
    # print("m:", m)
    # print("p.size:", p.size())
    # print("p:", p)

    # Generate integration sample points that avoid the singularity
    # Create empty arrays of the correct size
    zm = pt.zeros((len(xi), len(m)))            # TODO: Might have to rewrite this to extract length from Torch.Size
    zp = pt.zeros((len(xi), len(p)))

    # print("zm Shape:", zm.size())
    # print("zp Shape:", zp.size())

    # Compute maximum width of integration range based on the size of the input array of normalized velocities
    deltauMax = max(u_axis) - min(u_axis)
    # print("deltauMax:", deltauMax)      # Loses degrees of precision once it becomes a tensor, why?

    # Compute arrays of offsets to add to the central points in xi
    m_point_array = pt.tensor(phi + m * deltauMax)
    # print("m_point_array:", m_point_array)
    p_point_array = pt.tensor(phi + p * deltauMax)
    # print("m_point_array:", m_point_array)

    m_deltas = pt.concatenate((pt.tensor(pt.tensor(m_point_array[1:]) - pt.tensor(m_point_array[:-1])),
                               pt.tensor([0.])))
    # print("m_deltas shape:", m_deltas.size())
    # print("m_deltas:", m_deltas)

    p_deltas = pt.concatenate((pt.tensor(pt.tensor(p_point_array[1:]) - pt.tensor(p_point_array[:-1])),
                               pt.tensor([0.])))
    # print("p_deltas shape:", p_deltas.size())
    # print("p_deltas:", p_deltas)

    # The integration points on u
    for i in range(len(xi)):
        zm[i, :] = xi[i] + m_point_array
        zp[i, :] = xi[i] - p_point_array

    # Get sizes of zm and zp
    zmDims = list(zm.size())
    zpDims = list(zp.size())

    gm = torch_1d_interp(zm, u_axis, fPrime)
    gp = torch_1d_interp(zp, u_axis, fPrime)

    # print("gm Interpolation:", gm)
    # print("gp Interpolation:", gp)

    # Evaluate integral (df/du / (u - xi)) du
    M_array = m_deltas * gm / m_point_array
    # print("M_array:", M_array)
    P_array = p_deltas * gp / p_point_array
    # print("M_array:", M_array)

    integral = (
        pt.sum(M_array, axis=1)
        - pt.sum(P_array, axis=1)
        + 1j * pt.pi * g
        + 2 * phi * gPrime
    )

    # Convert mass and charge to SI units
    m_SI = particle_m * 1.6605e-27
    q_SI = particle_q * 1.6022e-19

    # Compute plasma frequency squared
    wpl2 = n * q_SI ** 2 / (m_SI * 8.8541878e-12)

    # Coefficient
    coefficient = -wpl2 / k ** 2 / (pt.sqrt(pt.tensor(2)) * v_th)

    # print("Coefficient:", coefficient)

    return coefficient * integral

def fast_spectral_density_arbdist(
    wavelengths,
    probe_wavelength,
    e_velocity_axes,
    i_velocity_axes,
    efn,    # efn = electron vdf
    ifn,
    n,
    notches: u.nm = None,  
    efract: np.ndarray = np.array([1.0]),
    ifract: np.ndarray = np.array([1.0]),
    ion_z=np.array([1]),
    ion_m=np.array([1]),
    probe_vec=np.array([1, 0, 0]),
    scatter_vec=np.array([0, 1, 0]),
    scattered_power=False,
    inner_range=0.1,
    inner_frac=0.8,
) -> Tuple[Union[np.floating, np.ndarray], np.ndarray]:
    
    # Convert arguments passed to tensors
    wavelengths = pt.as_tensor(wavelengths, dtype = pt.float64)
    e_velocity_axes = pt.as_tensor(e_velocity_axes, dtype = pt.float64)
    i_velocity_axes = pt.as_tensor(i_velocity_axes, dtype = pt.float64)
    efn = pt.as_tensor(efn, dtype = pt.float64)
    ifn = pt.as_tensor(ifn, dtype = pt.float64)
    efract = pt.as_tensor(efract, dtype = pt.float64)
    ifract = pt.as_tensor(ifract, dtype = pt.float64)
    ion_z = pt.as_tensor(ion_z, dtype = pt.float64)
    ion_m = pt.as_tensor(ion_m, dtype = pt.float64)
    probe_vec=pt.as_tensor(probe_vec, dtype = pt.float64)
    scatter_vec=pt.as_tensor(scatter_vec, dtype = pt.float64)

    # Ensure unit vectors are normalized
    probe_vec = probe_vec / pt.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / pt.linalg.norm(scatter_vec)

    # Normal vector along k, assume all velocities lie in this direction
    k_vec = pt.tensor(scatter_vec - probe_vec)
    k_vec = k_vec / pt.linalg.norm(k_vec)  # normalization

    # Compute drift velocities and thermal speeds for all electrons and ion species
    electron_vel = pt.tensor([])  # drift velocities (vector)
    electron_vel_1d = pt.tensor([]) # 1D drift velocities (scalar)
    vTe = pt.tensor([])  # thermal speeds (scalar)

    # Note that we convert to SI, strip units, then reintroduce them outside the loop to get the correct objects
    for i, fn in enumerate(efn):
        v_axis = e_velocity_axes[i]
        moment1_integrand = pt.multiply(fn, v_axis)
        bulk_velocity = pt.trapz(moment1_integrand, v_axis)       # Integrate along the given axis using the composite trapezoidal rule
        moment2_integrand = pt.multiply(fn, pt.square(v_axis - bulk_velocity))

        electron_vel = pt.concatenate((electron_vel, bulk_velocity * k_vec / pt.linalg.norm(k_vec)))
        electron_vel_1d = pt.concatenate((electron_vel_1d, pt.tensor([bulk_velocity])))
        vTe = pt.concatenate((vTe, pt.tensor([pt.sqrt(pt.trapz(moment2_integrand, v_axis))])))

    # print("electron_vel:", electron_vel)
    # print("electron_vel_1d:", electron_vel_1d)
    # print("vTe:", vTe)

    ion_vel = pt.tensor([])
    ion_vel_1d = pt.tensor([])
    vTi = pt.tensor([])

    for i, fn in enumerate(ifn):
        v_axis = i_velocity_axes[i]
        moment1_integrand = pt.multiply(fn, v_axis)
        bulk_velocity = pt.trapz(moment1_integrand, v_axis)
        moment2_integrand = pt.multiply(fn, pt.square(v_axis - bulk_velocity))

        ion_vel = pt.concatenate((ion_vel, bulk_velocity * k_vec / pt.linalg.norm(k_vec)))
        ion_vel_1d = pt.concatenate((ion_vel_1d, pt.tensor([bulk_velocity])))
        vTi = pt.concatenate((vTi, pt.tensor([pt.sqrt(pt.trapz(moment2_integrand, v_axis))])))

    # print("ion_vel:", ion_vel)
    # print("ion_vel_1d:", ion_vel_1d)
    # print("vTi:", vTi)

    # Define some constants
    C = pt.tensor([299792458], dtype = pt.float64)  # speed of light

    # Calculate plasma parameters
    zbar = pt.sum(ifract * ion_z)     
    ne = efract * n
    ni = ifract * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    # wpe = plasma_frequency(n=n, particle="e-").to(u.rad / u.s).value
    n = pt.tensor([n * 3182.60735], dtype = pt.float64)
    wpe = pt.sqrt(n)
    # print("wpe:", wpe)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = pt.tensor(2 * pt.pi * C / wavelengths)
    wl = pt.tensor(2 * pt.pi * C / probe_wavelength)

    # print("ws:", ws)
    # print("wl:", wl)

    # Compute the frequency shift (required by energy conservation)
    w = pt.tensor(ws - wl)
    # print("w:", w)

    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = pt.sqrt((ws ** 2 - wpe ** 2)) / C
    # print("ws ** 2 - wpe ** 2:", ws ** 2 - wpe ** 2)
    kl = pt.sqrt((wl ** 2 - wpe ** 2)) / C

    # print("ks:", ks)
    # print("kl:", kl)

    # Compute the wavenumber shift (required by momentum conservation)
    scattering_angle = pt.arccos(pt.dot(probe_vec, scatter_vec))
    # Eq. 1.7.10 in Sheffield
    k = pt.sqrt((ks ** 2 + kl ** 2 - 2 * ks * kl * pt.cos(scattering_angle)))
    # print("k:", k)

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot product over all wavelengths
    # and ion components
    w_e = w -pt.matmul(electron_vel, pt.outer(k, k_vec).T)
    w_i = w - pt.matmul(ion_vel, pt.outer(k, k_vec).T)

    # print("w_e:", w_e)
    # print("w_i:", w_i)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = pt.sqrt(pt.tensor(2)) * wpe / pt.outer(k, vTe)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xie = (pt.outer(1 / vTe, 1 / k) * w_e) / pt.sqrt(pt.tensor([2]))
    xii = (pt.outer(1 / vTi, 1 / k) * w_i) / pt.sqrt(pt.tensor([2]))

    # Calculate the susceptibilities
    # Apply Sheffield (3.3.9) with the following substitutions
    # xi = w / (sqrt2 k v_th), u = v / (sqrt2 v_th)
    # Then chi = -w_pl ** 2 / (2 v_th ** 2 k ** 2) integral (df/du / (u - xi)) du

    # Electron susceptibilities
    # print("efract.size():", efract.size())
    # print("efract.size():", len(efract))
    # print("w.size():", w.size())
    # print("w.size():", len(w))
    chiE = pt.zeros((len(efract), len(w)), dtype=pt.complex128)
    for i in range(len(efract)):
        chiE[i, :] = chi(
            f=efn[i],
            u_axis=(
                e_velocity_axes[i] - electron_vel_1d[i]
            )
            / (pt.sqrt(pt.tensor(2)) * vTe[i]),
            k=k,
            xi=xie[i],
            v_th=vTe[i],
            n=ne[i],
            particle_m=5.4858e-4,
            particle_q=-1,
            inner_range = inner_range,
            inner_frac = inner_frac
        )

    # print("chiE:", chiE)

    # Ion susceptibilities
    chiI = pt.zeros((len(ifract), len(w)), dtype=pt.complex128)
    for i in range(len(ifract)):
        chiI[i, :] = chi(
            f=ifn[i],
            u_axis=(i_velocity_axes[i] - ion_vel_1d[i])
            / (pt.sqrt(pt.tensor(2)) * vTi[i]),
            k=k,
            xi=xii[i],
            v_th=vTi[i],
            n=ni[i],
            particle_m=ion_m[i],
            particle_q=ion_z[i],
            inner_range = inner_range,
            inner_frac = inner_frac
        )

    # print("chiI:", chiI)

    # Calculate the longitudinal dielectric function
    epsilon = 1 + pt.sum(chiE, axis=0) + pt.sum(chiI, axis=0)
    # print("epsilon:", epsilon)

    xie = pt.flatten(xie)
    longArgE = (e_velocity_axes - electron_vel_1d) / (pt.sqrt(pt.tensor(2)) * vTe)
    longArgE = pt.flatten(longArgE)
    efn = pt.flatten(efn)

    eInterp = torch_1d_interp(xie, longArgE, efn)
    
    # Resize eInterp
    eInterp = pt.reshape(eInterp, (1, len(eInterp)))
    # print("eInterp:", eInterp)

    # Electron component of Skw from Sheffield 5.1.2
    econtr = pt.zeros((len(efract), len(w)), dtype=pt.complex128)
    for m in range(len(efract)):
        # print("EFIOEFEWO:", longArgE[m])
        econtr[m] = efract[m] * (
            2
            * pt.pi
            / k
            * pt.pow(pt.abs(1 - pt.sum(chiE, axis=0) / epsilon), 2)
            * eInterp[m]
            )

    # print("econtr:", econtr)

    xii = pt.flatten(xii)
    longArgI = (i_velocity_axes - ion_vel_1d) / (pt.sqrt(pt.tensor(2)) * vTi)
    longArgI = pt.flatten(longArgI)
    ifn = pt.flatten(ifn)

    iInterp = torch_1d_interp(xii, longArgI, ifn)
    
    # Resize eInterp
    inspeciInterp = pt.reshape(iInterp, (1, len(iInterp)))
    # print("iInterp:", iInterp)

    # print("iInterp:", iInterp)

    # ion component
    icontr = pt.zeros((len(ifract), len(w)), dtype=pt.complex128)
    for m in range(len(ifract)):
        icontr[m] = ifract[m] * (
            2
            * pt.pi
            * ion_z[m]
            / k
            * pt.pow(pt.abs(pt.sum(chiE, axis=0) / epsilon), 2)
            * iInterp[m]
        )

    # print("icontr:", icontr)

    # Recast as real: imaginary part is already zero
    Skw = pt.real(pt.sum(econtr, axis=0) + pt.sum(icontr, axis=0))

    # Convert to power spectrum if option is enabled
    if scattered_power:
        # Conversion factor
        Skw = Skw * (1 + 2 * w / wl) * 2 / (wavelengths ** 2)
        #this is to convert from S(frequency) to S(wavelength), there is an
        #extra 2 * pi * c here but that should be removed by normalization

    # print("S(k,w) before normalization:", Skw)

    # Account for notch(es)
    for myNotch in notches:
        if len(myNotch) != 2:
            raise ValueError("Notches must be pairs of values")

        x0 = pt.argmin(pt.abs(wavelengths - myNotch[0]))
        x1 = pt.argmin(pt.abs(wavelengths - myNotch[1]))
        Skw[x0:x1] = 0

    # Normalize result to have integral 1
    Skw = Skw / pt.trapz(Skw, wavelengths)

    # print("S(k,w) after normalization:", Skw)

    # Convert to np and return
    alpha = pt.mean(alpha)
    alpha = alpha.detach().numpy()
    Skw = Skw.detach().numpy()

    # print("alpha:", alpha)
    # print("S(k,w):", Skw)

    return alpha, Skw

def spectral_density_arbdist(
    wavelengths: u.nm,
    probe_wavelength: u.nm,
    e_velocity_axes: u.m / u.s,
    i_velocity_axes: u.m / u.s,
    efn: u.nm ** -1,
    ifn: u.nm ** -1,
    n: u.m ** -3,
    notches: u.nm = None,
    efract: np.ndarray = None,
    ifract: np.ndarray = None,
    ion_species: Union[str, List[str], Particle, List[Particle]] = "p",
    probe_vec=np.array([1, 0, 0]),
    scatter_vec=np.array([0, 1, 0]),
    scattered_power=False,
    inner_range=0.1,
    inner_frac=0.8,
) -> Tuple[Union[np.floating, np.ndarray], np.ndarray]:
    
    if efract is None:
        efract = np.ones(1)
    else:
        efract = np.asarray(efract, dtype=np.float64)

    if ifract is None:
        ifract = np.ones(1)
    else:
        ifract = np.asarray(ifract, dtype=np.float64)
        
    #Check for notches
    if notches is None:
        notches = [(0, 0)] * u.nm

    # Convert everything to SI, strip units
    wavelengths = wavelengths.to(u.m).value
    notches = notches.to(u.m).value
    probe_wavelength = probe_wavelength.to(u.m).value
    e_velocity_axes = e_velocity_axes.to(u.m / u.s).value
    i_velocity_axes = i_velocity_axes.to(u.m / u.s).value
    efn = efn.to(u.s / u.m).value
    ifn = ifn.to(u.s / u.m).value
    n = n.to(u.m ** -3).value
    
    
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
    ion_z = np.zeros(len(ion_species))
    ion_m = np.zeros(len(ion_species))
    for i, particle in enumerate(ion_species):
        ion_z[i] = particle.charge_number
        ion_m[i] = ion_species[i].mass_number
        
    
    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)
    
    
    return fast_spectral_density_arbdist(
        wavelengths, 
        probe_wavelength, 
        e_velocity_axes, 
        i_velocity_axes, 
        efn, 
        ifn,
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
    where :math:`\chi_e` is the electron component susceptibility of the
    plasma and :math:`\epsilon = 1 + \sum_e \chi_e + \sum_i \chi_i` is the total
    plasma dielectric  function (with :math:`\chi_i` being the ion component
    of the susceptibility), :math:`Z_i` is the charge of each ion, :math:`k`
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
    efract : array_like, shape (Ne, ), optional
        An array-like object where each element represents the fraction (or ratio)
        of the electron population number density to the total electron number density.
        Must sum to 1.0. Default is a single electron component.
    ifract : array_like, shape (Ni, ), optional
        An array-like object where each element represents the fraction (or ratio)
        of the ion population number density to the total ion number density.
        Must sum to 1.0. Default is a single ion species.
    ion_species : str or `~plasmapy.particles.Particle`, shape (Ni, ), optional
        A list or single instance of `~plasmapy.particles.Particle`, or strings
        convertible to `~plasmapy.particles.Particle`. Default is ``'H+'``
        corresponding to a single species of hydrogen ions.
    electron_vel : `~astropy.units.Quantity`, shape (Ne, 3), optional
        Velocity of each electron population in the rest frame. (convertible to m/s)
        If set, overrides electron_vdir and electron_speed.
        Defaults to a stationary plasma [0, 0, 0] m/s.
    ion_vel : `~astropy.units.Quantity`, shape (Ni, 3), optional
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
    For a concise summary of the relevant physics, see Chapter 5 of Derek
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

    return alpha, Skw * u.s / u.rad


# ***************************************************************************
# These functions are necessary to interface scalar Parameter objects with
# the array inputs of spectral_density
# ***************************************************************************


def _count_populations_in_params(params, prefix, allow_empty = False):
    """
    Counts the number of entries matching the pattern prefix_i in a
    list of keys
    """
    
    keys = list(params.keys())
    prefixLength = len(prefix)
    
    if allow_empty:
        nParams = 0
        for myKey in keys:
            if myKey[:prefixLength] == prefix:
                nParams = max(nParams, int(myKey[prefixLength + 1:]) + 1)
        
        return nParams
    
    else:
        return len(re.findall(prefix, ",".join(keys)))


def _params_to_array(params, prefix, vector=False, allow_empty = False):
    """
    Takes a list of parameters and returns an array of the values corresponding
    to a key, based on the following naming convention:
    Each parameter should be named prefix_i
    Where i is an integer (starting at 0)
    This function allows lmfit.Parameter inputs to be converted into the
    array-type inputs required by the spectral density function
    """

    if vector:
        npop = _count_populations_in_params(params, prefix + "_x", allow_empty = allow_empty)
        output = np.zeros([npop, 3])
        for i in range(npop):
            for j, ax in enumerate(["x", "y", "z"]):
                if (prefix + f"_{ax}_{i}") in params:
                    output[i, j] = params[prefix + f"_{ax}_{i}"].value
                else:
                    output[i, j] = None

    else:
        npop = _count_populations_in_params(params, prefix, allow_empty = allow_empty)
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
        n=n * 1e6, #this is so it accepts cm^-3 values by default
        efn=fe,
        ifn=fi,
        scattered_power=True,
        ifract = ifract,
        ion_z = ion_z,
        **settings,
    )

    print("alpha:", alpha)
    print("S(k,w):", model_Pw)

    # Put settings back now
    # this is necessary to avoid changing the settings array globally
    settings["emodel"] = emodel
    settings["imodel"] = imodel
    settings["ion_z"] = ion_z 

    return model_Pw

def _scattered_power_model_maxwellian(wavelengths, settings=None, **params):
    """
    lmfit Model function for fitting Thomson spectra
    For descriptions of arguments, see the `thomson_model` function.
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
    else:
        raise ValueError("Missing electron VDF model in settings")

    if "imodel" in settings:
        imodel = settings["imodel"]
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
        and may contain the following optional variables
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
        and may contain the following optional variables
            - efract_e# : Fraction of each electron population (must sum to 1) (optional)
            - ifract_i# : Fraction of each ion population (must sum to 1) (optional)
            - electron_speed_e# : Electron speed in m/s (optional)
            - ion_speed_i# : Ion speed in m/s (optional)
        where i# and e# are the number of electron and ion populations,
        zero-indexed, respectively (eg. 0,1,2...).
        These quantities can be either fixed or varying.
    Returns
    -------
    Spectral density (optimization function)
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

    # TODO: raise an exception if the number of any of the ion or electron
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
