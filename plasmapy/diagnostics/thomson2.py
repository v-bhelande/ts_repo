"""
Defines the Thomson scattering analysis module as part of
`plasmapy.diagnostics`.
"""

__all__ = [
    "spectral_density",
    "spectral_density_model",
    "spectral_power"
]
__lite_funcs__ = ["spectral_density_lite",
                  "spectral_power_lite",
                  "spectral_power_lite2"]

import astropy.constants as const
import astropy.units as u
import numbers
import numpy as np
import warnings

from lmfit import Model
from typing import Any, Callable, Optional, Union

from plasmapy.formulary import (
    permittivity_1D_Maxwellian_lite,
    plasma_frequency_lite,
    thermal_speed_coefficients,
    thermal_speed_lite,
)
from plasmapy.particles import Particle, ParticleLike
from plasmapy.particles.exceptions import ChargeError
from plasmapy.particles.particle_collections import ParticleList
from plasmapy.utils.decorators import (
    bind_lite_func,
    preserve_signature,
    validate_quantities,
)

__all__ += __lite_funcs__

c_si_unitless = const.c.si.value
e_si_unitless = const.e.si.value
m_p_si_unitless = const.m_p.si.value
m_e_si_unitless = const.m_e.si.value


# TODO: interface for inputting a multi-species configuration could be
#     simplified using the plasmapy.classes.plasma_base class if that class
#     included ion and electron drift velocities and information about the ion
#     atomic species.


@preserve_signature
def spectral_density_lite(
    wavelengths,
    probe_wavelength: numbers.Real,
    n: numbers.Real,
    T_e: np.ndarray,
    T_i: np.ndarray,
    efract: np.ndarray,
    ifract: np.ndarray,
    ion_z: np.ndarray,
    ion_mass: np.ndarray,
    electron_vel: np.ndarray,
    ion_vel: np.ndarray,
    probe_vec: np.ndarray,
    scatter_vec: np.ndarray,
    instr_func_arr: Optional[np.ndarray] = None,
) -> tuple[Union[np.floating, np.ndarray], np.ndarray]:
    r"""
    The :term:`lite-function` version of
    `~plasmapy.diagnostics.thomson.spectral_density`.  Performs the same
    thermal speed calculations as
    `~plasmapy.diagnostics.thomson.spectral_density`, but is intended
    for computational use and thus has data conditioning safeguards
    removed.

    Parameters
    ----------
    wavelengths : (Nλ,) `~numpy.ndarray`
        The wavelengths in meters over which the spectral density
        function will be calculated.

    probe_wavelength : real number
        Wavelength of the probe laser in meters.

    n : `~numpy.ndarray`
        Total combined number density of all electron populations in
        m\ :sup:`-3`\ .

    T_e : (Ne,) `~numpy.ndarray`
        Temperature of each electron population in kelvin, where Ne is
        the number of electron populations.

    T_i : (Ni,) `~numpy.ndarray`
        Temperature of each ion population in kelvin, where Ni is the
        number of ion populations.

    efract : (Ne,) `~numpy.ndarray`
        An `~numpy.ndarray` where each element represents the fraction
        (or ratio) of the electron population number density to the
        total electron number density. Must sum to 1.0. Default is a
        single electron population.

    ifract : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` object where each element represents the
        fraction (or ratio) of the ion population number density to the
        total ion number density. Must sum to 1.0. Default is a single
        ion species.

    ion_z : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` of the charge number :math:`Z` of each ion
        species.

    ion_mass : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` of the mass of each ion species in kg.

    electron_vel : (Ne, 3) `~numpy.ndarray`
        Velocity of each electron population in the rest frame (in m/s).
        If set, overrides ``electron_vdir`` and ``electron_speed``.
        Defaults to a stationary plasma ``[0, 0, 0]`` m/s.

    ion_vel : (Ni, 3) `~numpy.ndarray`
        Velocity vectors for each electron population in the rest frame
        (in  m/s). If set, overrides ``ion_vdir`` and ``ion_speed``.
        Defaults to zero drift for all specified ion species.

    probe_vec : (3,) float `~numpy.ndarray`
        Unit vector in the direction of the probe laser. Defaults to
        ``[1, 0, 0]``.

    scatter_vec : (3,) float `~numpy.ndarray`
        Unit vector pointing from the scattering volume to the detector.
        Defaults to [0, 1, 0] which, along with the default ``probe_vec``,
        corresponds to a 90 degree scattering angle geometry.

    instr_func_arr : `~numpy.ndarray`, shape (Nwavelengths,) optional
        The instrument function evaluated at a linearly spaced range of
        wavelengths ranging from :math:`-W` to :math:`W`, where

        .. math::
            W = 0.5*(\max{λ} - \min{λ})

        Here :math:`λ` is the ``wavelengths`` array. This array will be
        convolved with the spectral density function before it is
        returned.

    Returns
    -------
    alpha : float
        Mean scattering parameter, where ``alpha`` > 1 corresponds to
        collective scattering and ``alpha`` < 1 indicates non-collective
        scattering. The scattering parameter is calculated based on the
        total plasma density :math:`n`.

    Skw : `~numpy.ndarray`
        Computed spectral density function over the input
        ``wavelengths`` array with units of s/rad.
    """

    scattering_angle = np.arccos(np.dot(probe_vec, scatter_vec))

    # Calculate plasma parameters
    # Temperatures here in K!
    coefs = thermal_speed_coefficients("most_probable", 3)
    vT_e = thermal_speed_lite(T_e, m_e_si_unitless, coefs)
    vT_i = thermal_speed_lite(T_i, ion_mass, coefs)

    # Compute electron and ion densities
    ne = efract * n
    zbar = np.sum(ifract * ion_z)
    ni = ifract * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    wpe = plasma_frequency_lite(n, m_e_si_unitless, 1)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = 2 * np.pi * c_si_unitless / wavelengths
    wl = 2 * np.pi * c_si_unitless / probe_wavelength

    # Compute the frequency shift (required by energy conservation)
    w = ws - wl

    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = np.sqrt(ws**2 - wpe**2) / c_si_unitless
    kl = np.sqrt(wl**2 - wpe**2) / c_si_unitless

    # Compute the wavenumber shift (required by momentum conservation)
    # Eq. 1.7.10 in Sheffield
    k = np.sqrt(ks**2 + kl**2 - 2 * ks * kl * np.cos(scattering_angle))
    # Normal vector along k
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / np.linalg.norm(k_vec)

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot products over all wavelengths
    # and ion populations
    w_e = w - np.matmul(electron_vel, np.outer(k, k_vec).T)
    w_i = w - np.matmul(ion_vel, np.outer(k, k_vec).T)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = np.sqrt(2) * wpe / np.outer(k, vT_e)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xe = np.outer(1 / vT_e, 1 / k) * w_e
    xi = np.outer(1 / vT_i, 1 / k) * w_i

    # Calculate the susceptibilities
    chiE = np.zeros([efract.size, w.size], dtype=np.complex128)
    for i, _fract in enumerate(efract):  # noqa: B007
        wpe = plasma_frequency_lite(ne[i], m_e_si_unitless, 1)
        chiE[i, :] = permittivity_1D_Maxwellian_lite(w_e[i, :], k, vT_e[i], wpe)

    # Treatment of multiple species is an extension of the discussion in
    # Sheffield Sec. 5.1
    chiI = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for i, _fract in enumerate(ifract):  # noqa: B007
        wpi = plasma_frequency_lite(ni[i], ion_mass[i], ion_z[i])
        chiI[i, :] = permittivity_1D_Maxwellian_lite(w_i[i, :], k, vT_i[i], wpi)

    # Calculate the longitudinal dielectric function
    epsilon = 1 + np.sum(chiE, axis=0) + np.sum(chiI, axis=0)

    econtr = np.zeros([efract.size, w.size], dtype=np.complex128)
    for m in range(efract.size):
        econtr[m, :] = efract[m] * (
            2
            * np.sqrt(np.pi)
            / k
            / vT_e[m]
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
            / vT_i[m]
            * np.power(np.abs(np.sum(chiE, axis=0) / epsilon), 2)
            * np.exp(-xi[m, :] ** 2)
        )

    # Recast as real: imaginary part is already zero
    Skw = np.real(np.sum(econtr, axis=0) + np.sum(icontr, axis=0))

    # Apply an instrument function if one is provided
    if instr_func_arr is not None:
        Skw = np.convolve(Skw, instr_func_arr, mode="same")
    return np.mean(alpha), Skw



def spectral_power_lite(
    wavelengths,
    probe_wavelength: numbers.Real,
    n: numbers.Real,
    T_e: np.ndarray,
    T_i: np.ndarray,
    efract: np.ndarray,
    ifract: np.ndarray,
    ion_z: np.ndarray,
    ion_mass: np.ndarray,
    electron_vel: np.ndarray,
    ion_vel: np.ndarray,
    probe_vec: np.ndarray,
    scatter_vec: np.ndarray,
    instr_func_arr: Optional[np.ndarray] = None,
) -> tuple[Union[np.floating, np.ndarray], np.ndarray]:
    r"""
    Calculates the spectral power by correcting the spectral density function
    with a geometric factor G(k,w) from relativistic effects (Doppler shift 
    and light-cone effect), and multiplies by differential 
    frequency-to-wavelength factor 1/lambda^2. This is key to calculate the
    peak intensity in EPW
    
    Legacy annotation from copying from:
    The :term:`lite-function` version of
    `~plasmapy.diagnostics.thomson.spectral_density`.  Performs the same
    thermal speed calculations as
    `~plasmapy.diagnostics.thomson.spectral_density`, but is intended
    for computational use and thus has data conditioning safeguards
    removed.

    Parameters
    ----------
    wavelengths : (Nλ,) `~numpy.ndarray`
        The wavelengths in meters over which the spectral density
        function will be calculated.

    probe_wavelength : real number
        Wavelength of the probe laser in meters.

    n : `~numpy.ndarray`
        Total combined number density of all electron populations in
        m\ :sup:`-3`\ .

    T_e : (Ne,) `~numpy.ndarray`
        Temperature of each electron population in kelvin, where Ne is
        the number of electron populations.

    T_i : (Ni,) `~numpy.ndarray`
        Temperature of each ion population in kelvin, where Ni is the
        number of ion populations.

    efract : (Ne,) `~numpy.ndarray`
        An `~numpy.ndarray` where each element represents the fraction
        (or ratio) of the electron population number density to the
        total electron number density. Must sum to 1.0. Default is a
        single electron population.

    ifract : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` object where each element represents the
        fraction (or ratio) of the ion population number density to the
        total ion number density. Must sum to 1.0. Default is a single
        ion species.

    ion_z : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` of the charge number :math:`Z` of each ion
        species.

    ion_mass : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` of the mass of each ion species in kg.

    electron_vel : (Ne, 3) `~numpy.ndarray`
        Velocity of each electron population in the rest frame (in m/s).
        If set, overrides ``electron_vdir`` and ``electron_speed``.
        Defaults to a stationary plasma ``[0, 0, 0]`` m/s.

    ion_vel : (Ni, 3) `~numpy.ndarray`
        Velocity vectors for each electron population in the rest frame
        (in  m/s). If set, overrides ``ion_vdir`` and ``ion_speed``.
        Defaults to zero drift for all specified ion species.

    probe_vec : (3,) float `~numpy.ndarray`
        Unit vector in the direction of the probe laser. Defaults to
        ``[1, 0, 0]``.

    scatter_vec : (3,) float `~numpy.ndarray`
        Unit vector pointing from the scattering volume to the detector.
        Defaults to [0, 1, 0] which, along with the default ``probe_vec``,
        corresponds to a 90 degree scattering angle geometry.

    instr_func_arr : `~numpy.ndarray`, shape (Nwavelengths,) optional
        The instrument function evaluated at a linearly spaced range of
        wavelengths ranging from :math:`-W` to :math:`W`, where

        .. math::
            W = 0.5*(\max{λ} - \min{λ})

        Here :math:`λ` is the ``wavelengths`` array. This array will be
        convolved with the spectral density function before it is
        returned.

    Returns
    -------
    alpha : float
        Mean scattering parameter, where ``alpha`` > 1 corresponds to
        collective scattering and ``alpha`` < 1 indicates non-collective
        scattering. The scattering parameter is calculated based on the
        total plasma density :math:`n`.

    Skw : `~numpy.ndarray`
        Computed spectral density function over the input
        ``wavelengths`` array with units of s/rad.
    """

    scattering_angle = np.arccos(np.dot(probe_vec, scatter_vec))

    # Calculate plasma parameters
    # Temperatures here in K!
    coefs = thermal_speed_coefficients("most_probable", 3)
    vT_e = thermal_speed_lite(T_e, m_e_si_unitless, coefs)
    vT_i = thermal_speed_lite(T_i, ion_mass, coefs)

    # Compute electron and ion densities
    ne = efract * n
    zbar = np.sum(ifract * ion_z)
    ni = ifract * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    wpe = plasma_frequency_lite(n, m_e_si_unitless, 1)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = 2 * np.pi * c_si_unitless / wavelengths
    wl = 2 * np.pi * c_si_unitless / probe_wavelength

    # Compute the frequency shift (required by energy conservation)
    w = ws - wl

    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = np.sqrt(ws**2 - wpe**2) / c_si_unitless
    kl = np.sqrt(wl**2 - wpe**2) / c_si_unitless

    # Compute the wavenumber shift (required by momentum conservation)
    # Eq. 1.7.10 in Sheffield
    k = np.sqrt(ks**2 + kl**2 - 2 * ks * kl * np.cos(scattering_angle))
    # Normal vector along k
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / np.linalg.norm(k_vec)

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot products over all wavelengths
    # and ion populations
    w_e = w - np.matmul(electron_vel, np.outer(k, k_vec).T)
    w_i = w - np.matmul(ion_vel, np.outer(k, k_vec).T)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = np.sqrt(2) * wpe / np.outer(k, vT_e)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xe = np.outer(1 / vT_e, 1 / k) * w_e
    xi = np.outer(1 / vT_i, 1 / k) * w_i

    # Calculate the susceptibilities
    chiE = np.zeros([efract.size, w.size], dtype=np.complex128)
    for i, _fract in enumerate(efract):  # noqa: B007
        wpe = plasma_frequency_lite(ne[i], m_e_si_unitless, 1)
        chiE[i, :] = permittivity_1D_Maxwellian_lite(w_e[i, :], k, vT_e[i], wpe)

    # Treatment of multiple species is an extension of the discussion in
    # Sheffield Sec. 5.1
    chiI = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for i, _fract in enumerate(ifract):  # noqa: B007
        wpi = plasma_frequency_lite(ni[i], ion_mass[i], ion_z[i])
        chiI[i, :] = permittivity_1D_Maxwellian_lite(w_i[i, :], k, vT_i[i], wpi)

    # Calculate the longitudinal dielectric function
    epsilon = 1 + np.sum(chiE, axis=0) + np.sum(chiI, axis=0)

    econtr = np.zeros([efract.size, w.size], dtype=np.complex128)
    for m in range(efract.size):
        econtr[m, :] = efract[m] * (
            2
            * np.sqrt(np.pi)
            / k
            / vT_e[m]
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
            / vT_i[m]
            * np.power(np.abs(np.sum(chiE, axis=0) / epsilon), 2)
            * np.exp(-xi[m, :] ** 2)
        )

    # Recast as real: imaginary part is already zero
    Skw = np.real(np.sum(econtr, axis=0) + np.sum(icontr, axis=0))
    
    # Apply an instrument function if one is provided    
    if instr_func_arr is not None:
        Skw = np.convolve(Skw, instr_func_arr, mode="same")
    
    
    """ up until then, its the same as spectral_density_lite
    now we calculate the geometric factor gw and divide by lambda_s
    squared. This is especially important for the epw channel.

    Output is Pkw = Skw *gw /lambda_s^2    
    """
    for i, _fract in enumerate(efract):  # noqa: B007
        wpe = plasma_frequency_lite(ne[i], m_e_si_unitless, 1)
        k_i = 2 *np.pi /probe_wavelength
        k_s = 2 *np.pi /wavelengths
        omega_i = np.sqrt( k_i**2 *c_si_unitless**2 + wpe**2)
        omega_s = np.sqrt( k_s**2 *c_si_unitless**2 + wpe**2)
        omega = omega_s - omega_i
        gw = 1 +2*(omega/omega_i)
        Pkw = Skw *gw

    Pkw /= np.power(wavelengths,2)
    Pkw /= np.nanmax(Pkw)
    
    """ The model needs to adapt to the wave spectrum """
    " If the spectrum we are fitting is epw, we need to remove the notch"
    
    
    
    
    """If the spectrum is iaw, then we need to calculate the f-number
    correction that adds broadening to the feature"""
    
    return np.mean(alpha), Pkw #Skw



def spectral_power_lite2(
    wavelengths,
    probe_wavelength: numbers.Real,
    n: numbers.Real,
    T_e: np.ndarray,
    T_i: np.ndarray,
    efract: np.ndarray,
    ifract: np.ndarray,
    ion_z: np.ndarray,
    ion_mass: np.ndarray,
    electron_vel: np.ndarray,
    ion_vel: np.ndarray,
    probe_vec: np.ndarray,
    scatter_vec: np.ndarray,
    fNumberCorrection: bool,
    scatteringAngle: numbers.Real,
    deltaAngle: numbers.Real,
    angleStep: numbers.Real,
    removeNotch: bool,
    notch_minimum: numbers.Real,
    notch_maximum: numbers.Real,
    instr_func_arr: Optional[np.ndarray] = None,
) -> tuple[Union[np.floating, np.ndarray], np.ndarray]:
    r"""
    spectral_power_lite2 calls spectral_power_lite
    but adds modules to remove notch (useful for epw)
    and applies f-number broadening correction (important for iaw)
    
    this function adds as 'settings' for f-number broadening:
        - fNumberCorrection: True or False (boolean)
        - scatteringAngle:
        - deltaAngle:
        - angleStep:
    
    and for notch removal:
        - notch_minimum:
        - notch_maximum:
    
    
    Legacy annotation from copying from:
    The :term:`lite-function` version of
    `~plasmapy.diagnostics.thomson.spectral_density`.  Performs the same
    thermal speed calculations as
    `~plasmapy.diagnostics.thomson.spectral_density`, but is intended
    for computational use and thus has data conditioning safeguards
    removed.

    Parameters
    ----------
    wavelengths : (Nλ,) `~numpy.ndarray`
        The wavelengths in meters over which the spectral density
        function will be calculated.

    probe_wavelength : real number
        Wavelength of the probe laser in meters.

    n : `~numpy.ndarray`
        Total combined number density of all electron populations in
        m\ :sup:`-3`\ .

    T_e : (Ne,) `~numpy.ndarray`
        Temperature of each electron population in kelvin, where Ne is
        the number of electron populations.

    T_i : (Ni,) `~numpy.ndarray`
        Temperature of each ion population in kelvin, where Ni is the
        number of ion populations.

    efract : (Ne,) `~numpy.ndarray`
        An `~numpy.ndarray` where each element represents the fraction
        (or ratio) of the electron population number density to the
        total electron number density. Must sum to 1.0. Default is a
        single electron population.

    ifract : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` object where each element represents the
        fraction (or ratio) of the ion population number density to the
        total ion number density. Must sum to 1.0. Default is a single
        ion species.

    ion_z : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` of the charge number :math:`Z` of each ion
        species.

    ion_mass : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` of the mass of each ion species in kg.

    electron_vel : (Ne, 3) `~numpy.ndarray`
        Velocity of each electron population in the rest frame (in m/s).
        If set, overrides ``electron_vdir`` and ``electron_speed``.
        Defaults to a stationary plasma ``[0, 0, 0]`` m/s.

    ion_vel : (Ni, 3) `~numpy.ndarray`
        Velocity vectors for each electron population in the rest frame
        (in  m/s). If set, overrides ``ion_vdir`` and ``ion_speed``.
        Defaults to zero drift for all specified ion species.

    probe_vec : (3,) float `~numpy.ndarray`
        Unit vector in the direction of the probe laser. Defaults to
        ``[1, 0, 0]``.

    scatter_vec : (3,) float `~numpy.ndarray`
        Unit vector pointing from the scattering volume to the detector.
        Defaults to [0, 1, 0] which, along with the default ``probe_vec``,
        corresponds to a 90 degree scattering angle geometry.

    instr_func_arr : `~numpy.ndarray`, shape (Nwavelengths,) optional
        The instrument function evaluated at a linearly spaced range of
        wavelengths ranging from :math:`-W` to :math:`W`, where

        .. math::
            W = 0.5*(\max{λ} - \min{λ})

        Here :math:`λ` is the ``wavelengths`` array. This array will be
        convolved with the spectral density function before it is
        returned.

    Returns
    -------
    alpha : float
        Mean scattering parameter, where ``alpha`` > 1 corresponds to
        collective scattering and ``alpha`` < 1 indicates non-collective
        scattering. The scattering parameter is calculated based on the
        total plasma density :math:`n`.

    Skw : `~numpy.ndarray`
        Computed spectral density function over the input
        ``wavelengths`` array with units of s/rad.
    """

    
    if removeNotch == False and fNumberCorrection == False:
        alpha, Pkw = spectral_power_lite(
            wavelengths,
            probe_wavelength,
            n,
            T_e,
            T_i,
            efract=efract,
            ifract=ifract,
            ion_z=ion_z,
            ion_mass=ion_mass,
            electron_vel=electron_vel,
            ion_vel=ion_vel,
            probe_vec=probe_vec, # this needs to change with angle
            scatter_vec=scatter_vec, # this needs to change with angle
            instr_func_arr=instr_func_arr,
            )
        
        Pkw /= np.nanmax(Pkw)
        return alpha, Pkw
    
   
    
    """If the spectrum is iaw, then we need to calculate the f-number
    correction that adds broadening to the feature"""
    
    if fNumberCorrection == True: # Iterate spectral_density_lite through
                        # the several TS geometries
        # scatteringAngle := main scattering angle
        # deltaAngle := maximum angles around main scattering angle
        # angleStep  := size of angular slices around main scattering
                        # angle
        
        angles = np.arange(scatteringAngle-deltaAngle,
                           scatteringAngle+deltaAngle+angleStep,
                           angleStep)
        counter = 0
        for angle in angles:
            probe_vec2 = np.array([1, 0, 0])
            scattering_angle2 = np.deg2rad(angle)
            scatter_vec2 = np.array([np.cos(scattering_angle2),
                                     np.sin(scattering_angle2),
                                     0])
            
            alpha_aux, Pkw_aux = spectral_power_lite(
                wavelengths,
                probe_wavelength,
                n,
                T_e,
                T_i,
                efract=efract,
                ifract=ifract,
                ion_z=ion_z,
                ion_mass=ion_mass,
                electron_vel=electron_vel,
                ion_vel=ion_vel,
                probe_vec=probe_vec2, # this needs to change with angle
                scatter_vec=scatter_vec2, # this needs to change with angle
                instr_func_arr=instr_func_arr,
                )
            Pkw_aux /= np.nanmax(Pkw_aux)
            if counter == 0:
                total_Pkw = Pkw_aux
                counter += 1
            else:
                total_Pkw += Pkw_aux
            total_Pkw/= np.nanmax(total_Pkw)
        return alpha_aux,total_Pkw
    
    """ The model needs to adapt to the wave spectrum """
    " If the spectrum we are fitting is epw, we need to remove the notch"
    
    if removeNotch == True:
        if removeNotch == True:
            Pkw = total_Pkw
        else:
            alpha_aux, Pkw = spectral_power_lite(
                wavelengths,
                probe_wavelength,
                n,
                T_e,
                T_i,
                efract=efract,
                ifract=ifract,
                ion_z=ion_z,
                ion_mass=ion_mass,
                electron_vel=electron_vel,
                ion_vel=ion_vel,
                probe_vec=probe_vec, # this needs to change with angle
                scatter_vec=scatter_vec, # this needs to change with angle
                instr_func_arr=instr_func_arr,
                )
        
        print('Removing notch...')
        # measure the resolution of wavelength axis
        # to calculate position in the wavelengths array 
        # that define the notch
        res = np.abs(wavelengths[1] - wavelengths[0])
        # Converts lower limit of notch to an array position
        Min = int((notch_minimum - wavelengths[0])/res)  
                  
        # Converts upper limit of notch to an array position          
        Max = int((notch_maximum - wavelengths[0])/res)
                   
        # Replace values inside notch with NaN
        Pkw[Min:Max] = np.nan
        return alpha_aux,Pkw


@validate_quantities(
    wavelengths={"can_be_negative": False, "can_be_zero": False},
    probe_wavelength={"can_be_negative": False, "can_be_zero": False},
    n={"can_be_negative": False, "can_be_zero": False},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
@bind_lite_func(spectral_density_lite)
def spectral_density(  # noqa: C901, PLR0912, PLR0915
    wavelengths: u.nm,
    probe_wavelength: u.nm,
    n: u.m**-3,
    *,
    T_e: u.K,
    T_i: u.K,
    efract=None,
    ifract=None,
    ions: ParticleLike = "p+",
    electron_vel: u.m / u.s = None,
    ion_vel: u.m / u.s = None,
    probe_vec=None,
    scatter_vec=None,
    instr_func: Optional[Callable] = None,
) -> tuple[Union[np.floating, np.ndarray], np.ndarray]:
    r"""Calculate the spectral density function for Thomson scattering of
    a probe laser beam by a multi-species Maxwellian plasma.

    Parameters
    ----------
    wavelengths : `~astropy.units.Quantity`
        The wavelengths over which the spectral density function will be
        calculated, in units convertible to m.

    probe_wavelength : `~astropy.units.Quantity`
        Wavelength of the probe laser, in units convertible to m.

    n : `~astropy.units.Quantity`
        Total combined number density of all electron populations, in
        units convertible to m\ :sup:`-3`\ .

    T_e : (Ne,) `~astropy.units.Quantity`, |keyword-only|
        Temperature of each electron population in units convertible to
        K or eV, where Ne is the number of electron populations.

    T_i : (Ni,) `~astropy.units.Quantity`, |keyword-only|
        Temperature of each ion population in units convertible to K or
        eV, where Ni is the number of ion populations.

    efract : (Ne,) |array_like|, |keyword-only|, optional
        The ratio of the number density of each electron population to
        the total electron number density, denoted by :math:`F_e` below.
        Must sum to one. The default corresponds to a single electron
        population.

    ifract : (Ni,) |array_like|, |keyword-only|, optional
        The fractional number densities of each ion population, denoted
        by :math:`F_i` below. Must sum to one. The default corresponds
        to a single ion population.

    ions : (Ni,) |particle-like|, |keyword-only|, default: "p+"
        One or more positively charged ions representing each ion
        population.

    electron_vel : (Ne, 3) `~astropy.units.Quantity`, |keyword-only|, optional
        Velocity vectors for each electron population in the rest frame,
        in units convertible to m/s. If set, overrides ``electron_vdir``
        and ``electron_speed``.  Defaults to a stationary plasma at
        :math:`[0, 0, 0]` m/s.

    ion_vel : (Ni, 3) `~astropy.units.Quantity`, |keyword-only|, optional
        Velocity vectors for each ion population in the rest frame, in
        units convertible to m/s. If set, overrides ``ion_vdir`` and
        ``ion_speed``. Defaults to zero drift for all specified ion
        species.

    probe_vec : (3,) |array_like|, |keyword-only|, default: [1, 0, 0]
        Unit vector in the direction of the probe laser.

    scatter_vec : (3,) |array_like|, |keyword-only|, default: [0, 1, 0]
        Unit vector pointing from the scattering volume to the
        detector. The default, along with the default for ``probe_vec``,
        corresponds to a 90° scattering angle geometry.

    instr_func : function

        A function representing the instrument function that takes a
        `~astropy.units.Quantity` of wavelengths (centered on zero)
        and returns the instrument point spread function. The
        resulting array will be convolved with the spectral density
        function before it is returned.

    Returns
    -------
    alpha : `float`
        Mean scattering parameter, where ``alpha`` > 1 corresponds to
        collective scattering and ``alpha`` < 1 indicates
        non-collective scattering. The scattering parameter is
        calculated based on the total plasma density ``n``.

    Skw : `~astropy.units.Quantity`
        Computed spectral density function over the input
        ``wavelengths`` array with units of s/rad.

    Notes
    -----
    This function calculates the spectral density function for Thomson
    scattering of a probe laser beam by a plasma consisting of one or
    more ion species and one or more thermal electron populations (the
    entire plasma is assumed to be quasi-neutral):

    .. math::
        S(k,ω) = \sum_e \frac{2π}{k}
        \bigg |1 - \frac{χ_e}{ε} \bigg |^2
        f_{e0,e} \bigg (\frac{ω}{k} \bigg ) +
        \sum_i \frac{2π Z_i}{k}
        \bigg |\frac{χ_e}{ε} \bigg |^2 f_{i0,i}
        \bigg ( \frac{ω}{k} \bigg )

    where :math:`χ_e` is the electron population susceptibility of the
    plasma and :math:`ε = 1 + ∑_e χ_e + ∑_i χ_i` is the total plasma
    dielectric function (with :math:`χ_i` being the ion population of
    the susceptibility), :math:`Z_i` is the charge of each ion,
    :math:`k` is the scattering wavenumber, :math:`ω` is the scattering
    frequency, and :math:`f_{e0,e}` and :math:`f_{i0,i}` are the
    electron and ion velocity distribution functions, respectively. In
    this function, the electron and ion velocity distribution functions
    are assumed to be Maxwellian, making this function equivalent to Eq.
    3.4.6 in :cite:t:`sheffield:2011`\ .

    The number density of the e\ :sup:`th` electron populations is
    defined as

    .. math::
        n_e = F_e n

    where :math:`n` is the total number density of all electron
    populations combined and :math:`F_e` is the fractional number
    density of each electron population such that

    .. math::
        \sum_e n_e = n

    .. math::
        \sum_e F_e = 1

    The plasma is assumed to be quasineutral, and therefore the number
    density of the i\ :sup:`th` ion population is

    .. math::
        n_i = \frac{F_i n}{∑_i F_i Z_i}

    with :math:`F_i` defined in the same way as :math:`F_e`.

    For details, see "Plasma Scattering of Electromagnetic Radiation"
    by :cite:t:`sheffield:2011`. This code is a modified version of
    the program described therein.

    For a summary of the relevant physics, see Chapter 5 of the
    :cite:t:`schaeffer:2014` thesis.

    """

    # Validate efract
    if efract is None:
        efract = np.ones(1)
    else:
        efract = np.asarray(efract, dtype=np.float64)
        if np.sum(efract) != 1:
            raise ValueError(f"The provided efract does not sum to 1: {efract}")

    # Validate ifract
    if ifract is None:
        ifract = np.ones(1)
    else:
        ifract = np.asarray(ifract, dtype=np.float64)
        if np.sum(ifract) != 1:
            raise ValueError(f"The provided ifract does not sum to 1: {ifract}")

    if probe_vec is None:
        probe_vec = np.array([1, 0, 0])

    if scatter_vec is None:
        scatter_vec = np.array([0, 1, 0])

    # If electron velocity is not specified, create an array corresponding
    # to zero drift
    if electron_vel is None:
        electron_vel = np.zeros([efract.size, 3]) * u.m / u.s

    # Condition the electron velocity keywords
    if ion_vel is None:
        ion_vel = np.zeros([ifract.size, 3]) * u.m / u.s

    # Condition ions
    # If a single value is provided, turn into a particle list
    if isinstance(ions, ParticleList):
        pass
    elif isinstance(ions, str):
        ions = ParticleList([Particle(ions)])
    # If a list is provided, ensure all values are Particles, then convert
    # to a ParticleList
    elif isinstance(ions, list):
        for ii, ion in enumerate(ions):
            if isinstance(ion, Particle):
                continue
            ions[ii] = Particle(ion)
        ions = ParticleList(ions)
    else:
        raise TypeError(
            "The type of object provided to the ``ions`` keyword "
            f"is not supported: {type(ions)}"
        )

    # Validate ions
    if len(ions) == 0:
        raise ValueError("At least one ion species needs to be defined.")

    try:
        if sum(ion.charge_number <= 0 for ion in ions):
            raise ValueError("All ions must be positively charged.")
    # Catch error if charge information is missing
    except ChargeError as ex:
        raise ValueError("All ions must be positively charged.") from ex

    # Condition T_i
    if T_i.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If T_i.size != len(ions), assume same temp. for all species
        T_i = np.array([T_i.value]) * T_i.unit

    # Make sure the sizes of ions, ifract, ion_vel, and T_i all match
    if (
        (len(ions) != ifract.size)
        or (ion_vel.shape[0] != ifract.size)
        or (T_i.size != ifract.size)
    ):
        raise ValueError(
            f"Inconsistent number of ion species in ifract ({ifract}), "
            f"ions ({len(ions)}), T_i ({T_i.size}), "
            f"and/or ion_vel ({ion_vel.shape[0]})."
        )

    # Condition T_e
    if T_e.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If T_e.size != len(efract), assume same temp. for all species
        T_e = np.array([T_e.value]) * T_e.unit

    # Make sure the sizes of efract, electron_vel, and T_e all match
    if (electron_vel.shape[0] != efract.size) or (T_e.size != efract.size):
        raise ValueError(
            f"Inconsistent number of electron populations in efract ({efract.size}), "
            f"T_e ({T_e.size}), or electron velocity ({electron_vel.shape[0]})."
        )

    # Create arrays of ion Z and mass from particles given
    ion_z = ions.charge_number
    ion_mass = ions.mass

    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)

    # Apply the instrument function
    if instr_func is not None and callable(instr_func):
        # Create an array of wavelengths of the same size as wavelengths
        # but centered on zero
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        instr_func_arr = instr_func(eval_w)

        if type(instr_func_arr) != np.ndarray:
            raise ValueError(
                "instr_func must be a function that returns a "
                "np.ndarray, but the provided function returns "
                f" a {type(instr_func_arr)}"
            )

        if wavelengths.shape != instr_func_arr.shape:
            raise ValueError(
                "The shape of the array returned from the "
                f"instr_func ({instr_func_arr.shape}) "
                "does not match the shape of the wavelengths "
                f"array ({wavelengths.shape})."
            )

        instr_func_arr /= np.sum(instr_func_arr)
    else:
        instr_func_arr = None

    alpha, Skw = spectral_density_lite(
        wavelengths.to(u.m).value,
        probe_wavelength.to(u.m).value,
        n.to(u.m**-3).value,
        T_e.to(u.K).value,
        T_i.to(u.K).value,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_mass=ion_mass.to(u.kg).value,
        ion_vel=ion_vel.to(u.m / u.s).value,
        electron_vel=electron_vel.to(u.m / u.s).value,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        instr_func_arr=instr_func_arr,
    )

    return alpha, Skw * u.s / u.rad



@validate_quantities(
    wavelengths={"can_be_negative": False, "can_be_zero": False},
    probe_wavelength={"can_be_negative": False, "can_be_zero": False},
    n={"can_be_negative": False, "can_be_zero": False},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
@bind_lite_func(spectral_power_lite)
def spectral_power(  # noqa: C901, PLR0912, PLR0915
    wavelengths: u.nm,
    probe_wavelength: u.nm,
    n: u.m**-3,
    *,
    T_e: u.K,
    T_i: u.K,
    efract=None,
    ifract=None,
    ions: ParticleLike = "p+",
    electron_vel: u.m / u.s = None,
    ion_vel: u.m / u.s = None,
    probe_vec=None,
    scatter_vec=None,
    instr_func: Optional[Callable] = None,
) -> tuple[Union[np.floating, np.ndarray], np.ndarray]:
    r"""
    Calculates the spectral power by correcting the spectral density function
    with a geometric factor G(k,w) from relativistic effects (Doppler shift 
    and light-cone effect), and multiplies by differential 
    frequency-to-wavelength factor 1/lambda^2. This is key to calculate the
    peak intensity in EPW
    
    Legacy annotation from copying from spectral_density_lite:
    
    Calculate the spectral density function for Thomson scattering of
    a probe laser beam by a multi-species Maxwellian plasma.

    Parameters
    ----------
    wavelengths : `~astropy.units.Quantity`
        The wavelengths over which the spectral density function will be
        calculated, in units convertible to m.

    probe_wavelength : `~astropy.units.Quantity`
        Wavelength of the probe laser, in units convertible to m.

    n : `~astropy.units.Quantity`
        Total combined number density of all electron populations, in
        units convertible to m\ :sup:`-3`\ .

    T_e : (Ne,) `~astropy.units.Quantity`, |keyword-only|
        Temperature of each electron population in units convertible to
        K or eV, where Ne is the number of electron populations.

    T_i : (Ni,) `~astropy.units.Quantity`, |keyword-only|
        Temperature of each ion population in units convertible to K or
        eV, where Ni is the number of ion populations.

    efract : (Ne,) |array_like|, |keyword-only|, optional
        The ratio of the number density of each electron population to
        the total electron number density, denoted by :math:`F_e` below.
        Must sum to one. The default corresponds to a single electron
        population.

    ifract : (Ni,) |array_like|, |keyword-only|, optional
        The fractional number densities of each ion population, denoted
        by :math:`F_i` below. Must sum to one. The default corresponds
        to a single ion population.

    ions : (Ni,) |particle-like|, |keyword-only|, default: "p+"
        One or more positively charged ions representing each ion
        population.

    electron_vel : (Ne, 3) `~astropy.units.Quantity`, |keyword-only|, optional
        Velocity vectors for each electron population in the rest frame,
        in units convertible to m/s. If set, overrides ``electron_vdir``
        and ``electron_speed``.  Defaults to a stationary plasma at
        :math:`[0, 0, 0]` m/s.

    ion_vel : (Ni, 3) `~astropy.units.Quantity`, |keyword-only|, optional
        Velocity vectors for each ion population in the rest frame, in
        units convertible to m/s. If set, overrides ``ion_vdir`` and
        ``ion_speed``. Defaults to zero drift for all specified ion
        species.

    probe_vec : (3,) |array_like|, |keyword-only|, default: [1, 0, 0]
        Unit vector in the direction of the probe laser.

    scatter_vec : (3,) |array_like|, |keyword-only|, default: [0, 1, 0]
        Unit vector pointing from the scattering volume to the
        detector. The default, along with the default for ``probe_vec``,
        corresponds to a 90° scattering angle geometry.

    instr_func : function

        A function representing the instrument function that takes a
        `~astropy.units.Quantity` of wavelengths (centered on zero)
        and returns the instrument point spread function. The
        resulting array will be convolved with the spectral density
        function before it is returned.

    Returns
    -------
    alpha : `float`
        Mean scattering parameter, where ``alpha`` > 1 corresponds to
        collective scattering and ``alpha`` < 1 indicates
        non-collective scattering. The scattering parameter is
        calculated based on the total plasma density ``n``.

    Skw : `~astropy.units.Quantity`
        Computed spectral density function over the input
        ``wavelengths`` array with units of s/rad.

    Notes
    -----
    This function calculates the spectral density function for Thomson
    scattering of a probe laser beam by a plasma consisting of one or
    more ion species and one or more thermal electron populations (the
    entire plasma is assumed to be quasi-neutral):

    .. math::
        S(k,ω) = \sum_e \frac{2π}{k}
        \bigg |1 - \frac{χ_e}{ε} \bigg |^2
        f_{e0,e} \bigg (\frac{ω}{k} \bigg ) +
        \sum_i \frac{2π Z_i}{k}
        \bigg |\frac{χ_e}{ε} \bigg |^2 f_{i0,i}
        \bigg ( \frac{ω}{k} \bigg )

    where :math:`χ_e` is the electron population susceptibility of the
    plasma and :math:`ε = 1 + ∑_e χ_e + ∑_i χ_i` is the total plasma
    dielectric function (with :math:`χ_i` being the ion population of
    the susceptibility), :math:`Z_i` is the charge of each ion,
    :math:`k` is the scattering wavenumber, :math:`ω` is the scattering
    frequency, and :math:`f_{e0,e}` and :math:`f_{i0,i}` are the
    electron and ion velocity distribution functions, respectively. In
    this function, the electron and ion velocity distribution functions
    are assumed to be Maxwellian, making this function equivalent to Eq.
    3.4.6 in :cite:t:`sheffield:2011`\ .

    The number density of the e\ :sup:`th` electron populations is
    defined as

    .. math::
        n_e = F_e n

    where :math:`n` is the total number density of all electron
    populations combined and :math:`F_e` is the fractional number
    density of each electron population such that

    .. math::
        \sum_e n_e = n

    .. math::
        \sum_e F_e = 1

    The plasma is assumed to be quasineutral, and therefore the number
    density of the i\ :sup:`th` ion population is

    .. math::
        n_i = \frac{F_i n}{∑_i F_i Z_i}

    with :math:`F_i` defined in the same way as :math:`F_e`.

    For details, see "Plasma Scattering of Electromagnetic Radiation"
    by :cite:t:`sheffield:2011`. This code is a modified version of
    the program described therein.

    For a summary of the relevant physics, see Chapter 5 of the
    :cite:t:`schaeffer:2014` thesis.

    """

    # Validate efract
    if efract is None:
        efract = np.ones(1)
    else:
        efract = np.asarray(efract, dtype=np.float64)
        if np.sum(efract) != 1:
            raise ValueError(f"The provided efract does not sum to 1: {efract}")

    # Validate ifract
    if ifract is None:
        ifract = np.ones(1)
    else:
        ifract = np.asarray(ifract, dtype=np.float64)
        if np.sum(ifract) != 1:
            raise ValueError(f"The provided ifract does not sum to 1: {ifract}")

    if probe_vec is None:
        probe_vec = np.array([1, 0, 0])

    if scatter_vec is None:
        scatter_vec = np.array([0, 1, 0])

    # If electron velocity is not specified, create an array corresponding
    # to zero drift
    if electron_vel is None:
        electron_vel = np.zeros([efract.size, 3]) * u.m / u.s

    # Condition the electron velocity keywords
    if ion_vel is None:
        ion_vel = np.zeros([ifract.size, 3]) * u.m / u.s

    # Condition ions
    # If a single value is provided, turn into a particle list
    if isinstance(ions, ParticleList):
        pass
    elif isinstance(ions, str):
        ions = ParticleList([Particle(ions)])
    # If a list is provided, ensure all values are Particles, then convert
    # to a ParticleList
    elif isinstance(ions, list):
        for ii, ion in enumerate(ions):
            if isinstance(ion, Particle):
                continue
            ions[ii] = Particle(ion)
        ions = ParticleList(ions)
    else:
        raise TypeError(
            "The type of object provided to the ``ions`` keyword "
            f"is not supported: {type(ions)}"
        )

    # Validate ions
    if len(ions) == 0:
        raise ValueError("At least one ion species needs to be defined.")

    try:
        if sum(ion.charge_number <= 0 for ion in ions):
            raise ValueError("All ions must be positively charged.")
    # Catch error if charge information is missing
    except ChargeError as ex:
        raise ValueError("All ions must be positively charged.") from ex

    # Condition T_i
    if T_i.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If T_i.size != len(ions), assume same temp. for all species
        T_i = np.array([T_i.value]) * T_i.unit

    # Make sure the sizes of ions, ifract, ion_vel, and T_i all match
    if (
        (len(ions) != ifract.size)
        or (ion_vel.shape[0] != ifract.size)
        or (T_i.size != ifract.size)
    ):
        raise ValueError(
            f"Inconsistent number of ion species in ifract ({ifract}), "
            f"ions ({len(ions)}), T_i ({T_i.size}), "
            f"and/or ion_vel ({ion_vel.shape[0]})."
        )

    # Condition T_e
    if T_e.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If T_e.size != len(efract), assume same temp. for all species
        T_e = np.array([T_e.value]) * T_e.unit

    # Make sure the sizes of efract, electron_vel, and T_e all match
    if (electron_vel.shape[0] != efract.size) or (T_e.size != efract.size):
        raise ValueError(
            f"Inconsistent number of electron populations in efract ({efract.size}), "
            f"T_e ({T_e.size}), or electron velocity ({electron_vel.shape[0]})."
        )

    # Create arrays of ion Z and mass from particles given
    ion_z = ions.charge_number
    ion_mass = ions.mass

    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)

    # Apply the instrument function
    if instr_func is not None and callable(instr_func):
        # Create an array of wavelengths of the same size as wavelengths
        # but centered on zero
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        instr_func_arr = instr_func(eval_w)

        if type(instr_func_arr) != np.ndarray:
            raise ValueError(
                "instr_func must be a function that returns a "
                "np.ndarray, but the provided function returns "
                f" a {type(instr_func_arr)}"
            )

        if wavelengths.shape != instr_func_arr.shape:
            raise ValueError(
                "The shape of the array returned from the "
                f"instr_func ({instr_func_arr.shape}) "
                "does not match the shape of the wavelengths "
                f"array ({wavelengths.shape})."
            )

        instr_func_arr /= np.sum(instr_func_arr)
    else:
        instr_func_arr = None

    alpha, Skw = spectral_power_lite(
        wavelengths.to(u.m).value,
        probe_wavelength.to(u.m).value,
        n.to(u.m**-3).value,
        T_e.to(u.K).value,
        T_i.to(u.K).value,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_mass=ion_mass.to(u.kg).value,
        ion_vel=ion_vel.to(u.m / u.s).value,
        electron_vel=electron_vel.to(u.m / u.s).value,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        instr_func_arr=instr_func_arr,
    )
    Skw /= Skw.max()

    return alpha, Skw * u.s / u.rad


# ***************************************************************************
# These functions are necessary to interface scalar Parameter objects with
# the array inputs of spectral_density
# ***************************************************************************


def _count_populations_in_params(params: dict[str, Any], prefix: str) -> int:
    """
    Counts the number of electron or ion populations in a ``params``
    `dict`.

    The number of populations is determined by counting the number of
    items in the ``params`` `dict` with a key that starts with the
    string defined by ``prefix``.
    """
    return len([key for key in params if key.startswith(prefix)])


def _params_to_array(
    params: dict[str, Any], prefix: str, vector: bool = False
) -> np.ndarray:
    """
    Constructs an array from the values contained in the dictionary
    ``params`` associated with keys starting with the prefix defined by
    ``prefix``.

    If ``vector == False``, then values for keys matching the expression
    ``prefix_[0-9]+`` are gathered into a 1D array.

    If ``vector == True``, then values for keys matching the expression
    ``prefix_[xyz]_[0-9]+`` are gathered into a 2D array of shape
    ``(N, 3)``.

    Notes
    -----
    This function allows `lmfit.parameter.Parameter` inputs to be
    converted into the array-type inputs required by the spectral
    density function.
    """

    if vector:
        npop = _count_populations_in_params(params, f"{prefix}_x")
        output = np.zeros([npop, 3])
        for i in range(npop):
            for j, ax in enumerate(["x", "y", "z"]):
                output[i, j] = params[f"{prefix}_{ax}_{i}"].value

    else:
        npop = _count_populations_in_params(params, prefix)
        output = np.zeros([npop])
        for i in range(npop):
            output[i] = params[f"{prefix}_{i}"]

    return output


# ***************************************************************************
# Fitting functions
# ***************************************************************************


def _spectral_density_model(wavelengths, settings=None, **params):
    """
    lmfit Model function for fitting Thomson spectra.

    For descriptions of arguments, see the `thomson_model` function.
    """

    # LOAD FROM SETTINGS
    ion_z = settings["ion_z"]
    ion_mass = settings["ion_mass"]
    probe_vec = settings["probe_vec"]
    scatter_vec = settings["scatter_vec"]
    electron_vdir = settings["electron_vdir"]
    ion_vdir = settings["ion_vdir"]
    probe_wavelength = settings["probe_wavelength"]
    instr_func_arr = settings["instr_func_arr"]

    # LOAD FROM PARAMS
    n = params["n"]
    T_e = _params_to_array(params, "T_e")
    T_i = _params_to_array(params, "T_i")
    efract = _params_to_array(params, "efract")
    ifract = _params_to_array(params, "ifract")

    electron_speed = _params_to_array(params, "electron_speed")
    ion_speed = _params_to_array(params, "ion_speed")

    electron_vel = electron_speed[:, np.newaxis] * electron_vdir
    ion_vel = ion_speed[:, np.newaxis] * ion_vdir

    # Convert temperatures from eV to kelvin (required by fast_spectral_density)
    T_e *= 11604.51812155
    T_i *= 11604.51812155

    alpha, model_Skw = spectral_density_lite(
        wavelengths,
        probe_wavelength,
        n,
        T_e,
        T_i,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_mass=ion_mass,
        electron_vel=electron_vel,
        ion_vel=ion_vel,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        instr_func_arr=instr_func_arr,
    )

    model_Skw *= 1 / np.max(model_Skw)

    return model_Skw


def _spectral_power_model(wavelengths, settings=None, **params):
    """
    lmfit Model function for fitting Thomson spectra.

    For descriptions of arguments, see the `thomson_model` function.
    """

    # LOAD FROM SETTINGS
    ion_z = settings["ion_z"]
    ion_mass = settings["ion_mass"]
    probe_vec = settings["probe_vec"]
    scatter_vec = settings["scatter_vec"]
    electron_vdir = settings["electron_vdir"]
    ion_vdir = settings["ion_vdir"]
    probe_wavelength = settings["probe_wavelength"]
    instr_func_arr = settings["instr_func_arr"]

    # LOAD FROM PARAMS
    n = params["n"]
    T_e = _params_to_array(params, "T_e")
    T_i = _params_to_array(params, "T_i")
    efract = _params_to_array(params, "efract")
    ifract = _params_to_array(params, "ifract")

    electron_speed = _params_to_array(params, "electron_speed")
    ion_speed = _params_to_array(params, "ion_speed")

    electron_vel = electron_speed[:, np.newaxis] * electron_vdir
    ion_vel = ion_speed[:, np.newaxis] * ion_vdir

    # Convert temperatures from eV to kelvin (required by fast_spectral_density)
    T_e *= 11604.51812155
    T_i *= 11604.51812155

    alpha, model_Pkw = spectral_power_lite(
        wavelengths,
        probe_wavelength,
        n,
        T_e,
        T_i,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_mass=ion_mass,
        electron_vel=electron_vel,
        ion_vel=ion_vel,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        instr_func_arr=instr_func_arr,
    )

    model_Pkw *= 1 / np.max(model_Pkw)

    return model_Pkw


def _spectral_power_model2(wavelengths, settings=None, **params):
    """
    lmfit Model function for fitting Thomson spectra.

    For descriptions of arguments, see the `thomson_model` function.
    """
    
    """
    
    This version uses spectral_power_lite2 which does 2 important
    things:
        1) Removes the notch filter (a setting is set to True when we
                                     want to remove the notch; when
                                     fitting epw, for example.)
        2) Calculates the f-number broadening (a setting is set to True
                                               when we want to do this;
                                               when fitting iaw,
                                               for example)
    
    
    """

    # LOAD FROM SETTINGS
    ion_z = settings["ion_z"]
    ion_mass = settings["ion_mass"]
    probe_vec = settings["probe_vec"]
    scatter_vec = settings["scatter_vec"]
    electron_vdir = settings["electron_vdir"]
    ion_vdir = settings["ion_vdir"]
    probe_wavelength = settings["probe_wavelength"]
    fNumberCorrection = settings["fNumberCorrection"]
    scatteringAngle = settings["scatteringAngle"]
    deltaAngle = settings["deltaAngle"]
    angleStep = settings["angleStep"]
    removeNotch = settings["removeNotch"]
    notch_minimum = settings["notch_minimum"]
    notch_maximum = settings["notch_maximum"]
    instr_func_arr = settings["instr_func_arr"]

    # LOAD FROM PARAMS
    n = params["n"]
    T_e = _params_to_array(params, "T_e")
    T_i = _params_to_array(params, "T_i")
    efract = _params_to_array(params, "efract")
    ifract = _params_to_array(params, "ifract")

    electron_speed = _params_to_array(params, "electron_speed")
    ion_speed = _params_to_array(params, "ion_speed")

    electron_vel = electron_speed[:, np.newaxis] * electron_vdir
    ion_vel = ion_speed[:, np.newaxis] * ion_vdir

    # Convert temperatures from eV to kelvin (required by fast_spectral_density)
    T_e *= 11604.51812155
    T_i *= 11604.51812155

    alpha, model_Pkw = spectral_power_lite2(
        wavelengths,
        probe_wavelength,
        n,
        T_e,
        T_i,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_mass=ion_mass,
        electron_vel=electron_vel,
        ion_vel=ion_vel,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        fNumberCorrection=fNumberCorrection,
        scatteringAngle=scatteringAngle,
        deltaAngle=deltaAngle,
        angleStep=angleStep,
        removeNotch=removeNotch,
        notch_minimum=notch_minimum,
        notch_maximum=notch_maximum,
        instr_func_arr=instr_func_arr,
        )

    model_Pkw *= 1 / np.max(model_Pkw)

    return model_Pkw


def spectral_density_model(  # noqa: C901, PLR0912, PLR0915
    wavelengths, settings, params
):
    r"""
    Returns a `lmfit.model.Model` function for Thomson spectral density
    function.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Wavelength array, in meters.

    settings : dict
        A dictionary of non-variable inputs to the spectral density
        function which must include the following keys:

        - ``"probe_wavelength"``: Probe wavelength in meters
        - ``"probe_vec"`` : (3,) unit vector in the probe direction
        - ``"scatter_vec"``: (3,) unit vector in the scattering
          direction
        - ``"ions"`` : list of particle strings,
          `~plasmapy.particles.particle_class.Particle` objects, or a
          `~plasmapy.particles.particle_collections.ParticleList`
          describing each ion species. All ions must be positive.

        and may contain the following optional variables:

        - ``"electron_vdir"`` : (e#, 3) array of electron velocity unit
          vectors
        - ``"ion_vdir"`` : (e#, 3) array of ion velocity unit vectors
        - ``"instr_func"`` : A function that takes a wavelength
          |Quantity| array and returns a spectrometer instrument
          function as an `~numpy.ndarray`.

        These quantities cannot be varied during the fit.

    params : `~lmfit.parameter.Parameters` object
        A `~lmfit.parameter.Parameters` object that must contain the
        following variables:

        - n: Total combined density of the electron populations in
          m\ :sup:`-3`
        - :samp:`T_e_{e#}` : Temperature in eV
        - :samp:`T_i_{i#}` : Temperature in eV

        where where :samp:`{i#}` and where :samp:`{e#}` are replaced by
        the number of electron and ion populations, zero-indexed,
        respectively (e.g., 0, 1, 2, ...). The
        `~lmfit.parameter.Parameters` object may also contain the
        following optional variables:

        - :samp:`"efract_{e#}"` : Fraction of each electron population
          (must sum to 1)
        - :samp:`"ifract_{i#}"` : Fraction of each ion population (must
          sum to 1)
        - :samp:`"electron_speed_{e#}"` : Electron speed in m/s
        - :samp:`"ion_speed_{ei}"` : Ion speed in m/s

        These quantities can be either fixed or varying.

    Returns
    -------
    model : `lmfit.model.Model`
        An `lmfit.model.Model` of the spectral density function for the
        provided settings and parameters that can be used to fit Thomson
        scattering data.

    Notes
    -----
    If an instrument function is included, the data should not include
    any `numpy.nan` values — instead regions with no data should be
    removed from both the data and wavelength arrays using
    `numpy.delete`.
    """

    required_settings = {
        "probe_wavelength",
        "probe_vec",
        "scatter_vec",
        "ions",
    }

    if missing_settings := required_settings - set(settings):
        raise ValueError(
            f"The following required settings were not provided in the "
            f"'settings' argument: {missing_settings}"
        )

    required_params = {"n"}
    if missing_params := required_params - set(params):
        raise ValueError(
            f"The following required parameters were not provided in the "
            f"'params': {missing_params}"
        )

    # **********************
    # Count number of populations
    # **********************
    if "efract_0" not in params:
        params.add("efract_0", value=1.0, vary=False)

    if "ifract_0" not in params:
        params.add("ifract_0", value=1.0, vary=False)

    num_e = _count_populations_in_params(params, "efract")
    num_i = _count_populations_in_params(params, "ifract")

    # **********************
    # Required settings and parameters per population
    # **********************
    for p, nums in zip(["T_e", "T_i"], [num_e, num_i]):
        for num in range(nums):
            key = f"{p}_{num!s}"
            if key not in params:
                raise ValueError(
                    f"{p} was not provided in kwarg 'parameters', but is required."
                )

    # **************
    # ions
    # **************

    ions = settings["ions"]
    # Condition ions
    # If a single value is provided, turn into a particle list
    if isinstance(ions, ParticleList):
        pass
    elif isinstance(ions, str):
        ions = ParticleList([Particle(ions)])
    # If a list is provided, ensure all values are Particles, then convert
    # to a ParticleList
    elif isinstance(ions, list):
        for ii, ion in enumerate(ions):
            if isinstance(ion, Particle):
                continue
            ions[ii] = Particle(ion)
        ions = ParticleList(ions)
    else:
        raise TypeError(
            "The type of object provided to the ``ions`` keyword "
            f"is not supported: {type(ions)}"
        )

    # Validate ions
    if len(ions) == 0:
        raise ValueError("At least one ion species needs to be defined.")

    try:
        if sum(ion.charge_number <= 0 for ion in ions):
            raise ValueError("All ions must be positively charged.")
    # Catch error if charge information is missing
    except ChargeError as ex:
        raise ValueError("All ions must be positively charged.") from ex

    # Create arrays of ion Z and mass from particles given
    settings["ion_z"] = ions.charge_number
    settings["ion_mass"] = ions.mass

    # **************
    # efract and ifract
    # **************

    # Automatically add an expression to the last efract parameter to
    # indicate that it depends on the others (so they sum to 1.0)
    # The resulting expression for the last of three will look like
    # efract_2.expr = "1.0 - efract_0 - efract_1"
    if num_e > 1:
        nums = ["1.0"] + [f"efract_{i}" for i in range(num_e - 1)]
        params[f"efract_{num_e - 1}"].expr = " - ".join(nums)

    if num_i > 1:
        nums = ["1.0"] + [f"ifract_{i}" for i in range(num_i - 1)]
        params[f"ifract_{num_i - 1}"].expr = " - ".join(nums)

    # **************
    # Electron velocity
    # **************
    electron_speed = np.zeros(num_e)
    for num in range(num_e):
        k = f"electron_speed_{num}"
        if k in params:
            electron_speed[num] = params[k].value
        else:
            # electron_speed[e] = 0 already
            params.add(k, value=0, vary=False)

    if "electron_vdir" not in settings:
        if np.all(electron_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["electron_vdir"] = np.ones([num_e, 3])
        else:
            raise ValueError(
                "Key 'electron_vdir' must be defined in kwarg 'settings' if "
                "any electron population has a non-zero speed (i.e. any "
                "params['electron_speed_<#>'] is non-zero)."
            )
    norm = np.linalg.norm(settings["electron_vdir"], axis=-1)
    settings["electron_vdir"] = settings["electron_vdir"] / norm[:, np.newaxis]

    # **************
    # Ion velocity
    # **************
    ion_speed = np.zeros(num_i)
    for num in range(num_i):
        k = f"ion_speed_{num}"
        if k in params:
            ion_speed[num] = params[k].value
        else:
            # ion_speed[i] = 0 already
            params.add(k, value=0, vary=False)

    if "ion_vdir" not in list(settings.keys()):
        if np.all(ion_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["ion_vdir"] = np.ones([num_i, 3])
        else:
            raise ValueError(
                "Key 'ion_vdir' must be defined in kwarg 'settings' if "
                "any ion population has a non-zero speed (i.e. any "
                "params['ion_speed_<#>'] is non-zero)."
            )
    norm = np.linalg.norm(settings["ion_vdir"], axis=-1)
    settings["ion_vdir"] = settings["ion_vdir"] / norm[:, np.newaxis]

    if "instr_func" not in settings or settings["instr_func"] is None:
        settings["instr_func_arr"] = None
    else:
        # Create instr_func array from instr_func
        instr_func = settings["instr_func"]
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        instr_func_arr = instr_func(eval_w * u.m)

        if type(instr_func_arr) != np.ndarray:
            raise ValueError(
                "instr_func must be a function that returns a "
                "np.ndarray, but the provided function returns "
                f" a {type(instr_func_arr)}"
            )

        if wavelengths.shape != instr_func_arr.shape:
            raise ValueError(
                "The shape of the array returned from the "
                f"instr_func ({instr_func_arr.shape}) "
                "does not match the shape of the wavelengths "
                f"array ({wavelengths.shape})."
            )

        instr_func_arr *= 1 / np.sum(instr_func_arr)
        settings["instr_func_arr"] = instr_func_arr

        warnings.warn(
            "If an instrument function is included, the data "
            "should not include any `numpy.nan` values. "
            "Instead regions with no data should be removed from "
            "both the data and wavelength arrays using "
            "`numpy.delete`."
        )

    # TODO: raise an exception if the number of any of the ion or electron
    #       quantities isn't consistent with the number of that species defined
    #       by ifract or efract.

    # Create and return the lmfit.Model
    return Model(
        _spectral_density_model,
        independent_vars=["wavelengths"],
        nan_policy="omit",
        settings=settings,
    )

def spectral_power_model(  # noqa: C901, PLR0912, PLR0915
    wavelengths, settings, params
):
    r"""
    Calculates the spectral power by correcting the spectral density function
    with a geometric factor G(k,w) from relativistic effects (Doppler shift 
    and light-cone effect), and multiplies by differential 
    frequency-to-wavelength factor 1/lambda^2. This is key to calculate the
    peak intensity in EPW
    
    Legacy annotation from copying from spectral_density_model:
    
    Returns a `lmfit.model.Model` function for Thomson spectral density
    function.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Wavelength array, in meters.

    settings : dict
        A dictionary of non-variable inputs to the spectral density
        function which must include the following keys:

        - ``"probe_wavelength"``: Probe wavelength in meters
        - ``"probe_vec"`` : (3,) unit vector in the probe direction
        - ``"scatter_vec"``: (3,) unit vector in the scattering
          direction
        - ``"ions"`` : list of particle strings,
          `~plasmapy.particles.particle_class.Particle` objects, or a
          `~plasmapy.particles.particle_collections.ParticleList`
          describing each ion species. All ions must be positive.

        and may contain the following optional variables:

        - ``"electron_vdir"`` : (e#, 3) array of electron velocity unit
          vectors
        - ``"ion_vdir"`` : (e#, 3) array of ion velocity unit vectors
        - ``"instr_func"`` : A function that takes a wavelength
          |Quantity| array and returns a spectrometer instrument
          function as an `~numpy.ndarray`.

        These quantities cannot be varied during the fit.

    params : `~lmfit.parameter.Parameters` object
        A `~lmfit.parameter.Parameters` object that must contain the
        following variables:

        - n: Total combined density of the electron populations in
          m\ :sup:`-3`
        - :samp:`T_e_{e#}` : Temperature in eV
        - :samp:`T_i_{i#}` : Temperature in eV

        where where :samp:`{i#}` and where :samp:`{e#}` are replaced by
        the number of electron and ion populations, zero-indexed,
        respectively (e.g., 0, 1, 2, ...). The
        `~lmfit.parameter.Parameters` object may also contain the
        following optional variables:

        - :samp:`"efract_{e#}"` : Fraction of each electron population
          (must sum to 1)
        - :samp:`"ifract_{i#}"` : Fraction of each ion population (must
          sum to 1)
        - :samp:`"electron_speed_{e#}"` : Electron speed in m/s
        - :samp:`"ion_speed_{ei}"` : Ion speed in m/s

        These quantities can be either fixed or varying.

    Returns
    -------
    model : `lmfit.model.Model`
        An `lmfit.model.Model` of the spectral density function for the
        provided settings and parameters that can be used to fit Thomson
        scattering data.

    Notes
    -----
    If an instrument function is included, the data should not include
    any `numpy.nan` values — instead regions with no data should be
    removed from both the data and wavelength arrays using
    `numpy.delete`.
    """

    required_settings = {
        "probe_wavelength",
        "probe_vec",
        "scatter_vec",
        "ions",
    }

    if missing_settings := required_settings - set(settings):
        raise ValueError(
            f"The following required settings were not provided in the "
            f"'settings' argument: {missing_settings}"
        )

    required_params = {"n"}
    if missing_params := required_params - set(params):
        raise ValueError(
            f"The following required parameters were not provided in the "
            f"'params': {missing_params}"
        )

    # **********************
    # Count number of populations
    # **********************
    if "efract_0" not in params:
        params.add("efract_0", value=1.0, vary=False)

    if "ifract_0" not in params:
        params.add("ifract_0", value=1.0, vary=False)

    num_e = _count_populations_in_params(params, "efract")
    num_i = _count_populations_in_params(params, "ifract")

    # **********************
    # Required settings and parameters per population
    # **********************
    for p, nums in zip(["T_e", "T_i"], [num_e, num_i]):
        for num in range(nums):
            key = f"{p}_{num!s}"
            if key not in params:
                raise ValueError(
                    f"{p} was not provided in kwarg 'parameters', but is required."
                )

    # **************
    # ions
    # **************

    ions = settings["ions"]
    # Condition ions
    # If a single value is provided, turn into a particle list
    if isinstance(ions, ParticleList):
        pass
    elif isinstance(ions, str):
        ions = ParticleList([Particle(ions)])
    # If a list is provided, ensure all values are Particles, then convert
    # to a ParticleList
    elif isinstance(ions, list):
        for ii, ion in enumerate(ions):
            if isinstance(ion, Particle):
                continue
            ions[ii] = Particle(ion)
        ions = ParticleList(ions)
    else:
        raise TypeError(
            "The type of object provided to the ``ions`` keyword "
            f"is not supported: {type(ions)}"
        )

    # Validate ions
    if len(ions) == 0:
        raise ValueError("At least one ion species needs to be defined.")

    try:
        if sum(ion.charge_number <= 0 for ion in ions):
            raise ValueError("All ions must be positively charged.")
    # Catch error if charge information is missing
    except ChargeError as ex:
        raise ValueError("All ions must be positively charged.") from ex

    # Create arrays of ion Z and mass from particles given
    settings["ion_z"] = ions.charge_number
    settings["ion_mass"] = ions.mass

    # **************
    # efract and ifract
    # **************

    # Automatically add an expression to the last efract parameter to
    # indicate that it depends on the others (so they sum to 1.0)
    # The resulting expression for the last of three will look like
    # efract_2.expr = "1.0 - efract_0 - efract_1"
    if num_e > 1:
        nums = ["1.0"] + [f"efract_{i}" for i in range(num_e - 1)]
        params[f"efract_{num_e - 1}"].expr = " - ".join(nums)

    if num_i > 1:
        nums = ["1.0"] + [f"ifract_{i}" for i in range(num_i - 1)]
        params[f"ifract_{num_i - 1}"].expr = " - ".join(nums)

    # **************
    # Electron velocity
    # **************
    electron_speed = np.zeros(num_e)
    for num in range(num_e):
        k = f"electron_speed_{num}"
        if k in params:
            electron_speed[num] = params[k].value
        else:
            # electron_speed[e] = 0 already
            params.add(k, value=0, vary=False)

    if "electron_vdir" not in settings:
        if np.all(electron_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["electron_vdir"] = np.ones([num_e, 3])
        else:
            raise ValueError(
                "Key 'electron_vdir' must be defined in kwarg 'settings' if "
                "any electron population has a non-zero speed (i.e. any "
                "params['electron_speed_<#>'] is non-zero)."
            )
    norm = np.linalg.norm(settings["electron_vdir"], axis=-1)
    settings["electron_vdir"] = settings["electron_vdir"] / norm[:, np.newaxis]

    # **************
    # Ion velocity
    # **************
    ion_speed = np.zeros(num_i)
    for num in range(num_i):
        k = f"ion_speed_{num}"
        if k in params:
            ion_speed[num] = params[k].value
        else:
            # ion_speed[i] = 0 already
            params.add(k, value=0, vary=False)

    if "ion_vdir" not in list(settings.keys()):
        if np.all(ion_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["ion_vdir"] = np.ones([num_i, 3])
        else:
            raise ValueError(
                "Key 'ion_vdir' must be defined in kwarg 'settings' if "
                "any ion population has a non-zero speed (i.e. any "
                "params['ion_speed_<#>'] is non-zero)."
            )
    norm = np.linalg.norm(settings["ion_vdir"], axis=-1)
    settings["ion_vdir"] = settings["ion_vdir"] / norm[:, np.newaxis]

    if "instr_func" not in settings or settings["instr_func"] is None:
        settings["instr_func_arr"] = None
    else:
        # Create instr_func array from instr_func
        instr_func = settings["instr_func"]
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        instr_func_arr = instr_func(eval_w * u.m)

        if type(instr_func_arr) != np.ndarray:
            raise ValueError(
                "instr_func must be a function that returns a "
                "np.ndarray, but the provided function returns "
                f" a {type(instr_func_arr)}"
            )

        if wavelengths.shape != instr_func_arr.shape:
            raise ValueError(
                "The shape of the array returned from the "
                f"instr_func ({instr_func_arr.shape}) "
                "does not match the shape of the wavelengths "
                f"array ({wavelengths.shape})."
            )

        instr_func_arr *= 1 / np.sum(instr_func_arr)
        settings["instr_func_arr"] = instr_func_arr

        warnings.warn(
            "If an instrument function is included, the data "
            "should not include any `numpy.nan` values. "
            "Instead regions with no data should be removed from "
            "both the data and wavelength arrays using "
            "`numpy.delete`."
        )

    # TODO: raise an exception if the number of any of the ion or electron
    #       quantities isn't consistent with the number of that species defined
    #       by ifract or efract.

    # Create and return the lmfit.Model
    return Model(
        _spectral_power_model,
        independent_vars=["wavelengths"],
        nan_policy="omit",
        settings=settings,
    )
