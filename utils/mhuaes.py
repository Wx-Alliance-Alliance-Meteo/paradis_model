"""
Thermodynamic function for dew point depression calculation.

This module is a Python conversion of the Fortran subroutine mhuaes3 from
the tdpack library (src/mhuaes.F90), with gemdyn-specific constraints applied.

Source: Environment Canada - Atmospheric Science and Technology
Original Author: N. Brunet (Jan 1991)
Fortran file: src/mhuaes.F90
gemdyn usage: src/base/out_thm.F90 and src/base/out_thm_hlt.F90
"""

import numpy as np

# Thermodynamic constants from tdpack_const.h
# Alduchov and Eskridge (1995) coefficients for saturation vapor pressure
AERK1W = 610.94  # Pa - coefficient for water phase
AERK1I = 611.21  # Pa - coefficient for ice phase
AERK2W = 17.625  # dimensionless - coefficient for water phase
AERK2I = 22.587  # dimensionless - coefficient for ice phase
AERK3W = 30.11   # K - coefficient for water phase
AERK3I = -0.71   # K - coefficient for ice phase

# Other thermodynamic constants
TRPL = 273.16    # K - triple point temperature of water
EPS1 = 0.6219800221014  # rgasd/rgasv - ratio of gas constants
EPS2 = 0.3780199778986  # 1 - eps1

# gemdyn model constraints
ES_MAX = 30.0    # K - maximum dew point depression (from gemdyn)


def foefq(qqq, prs):
    """
    Calculate vapor pressure from specific humidity and pressure.

    This implements the FOEFQ macro from tdpack_func.h.

    Parameters
    ----------
    qqq : Specific humidity [kg/kg]
    prs : Pressure [Pa]

    Returns
    -------
    array_like
        Vapor pressure [Pa]

    Notes
    -----
    Formula: e = min(prs, (qqq * prs) / (EPS1 + EPS2 * qqq))
    """
    return np.minimum(prs, (qqq * prs) / (EPS1 + EPS2 * qqq))


def mhuaes3(hu, tt, ps):
    """
    Calculate dew point depression from specific humidity, temperature, and pressure.

    This vectorized function converts specific humidity to dew point depression,
    applying gemdyn model constraints (water phase only, 30 K cap).

    Parameters
    ----------
    hu : Specific humidity [kg/kg]
    tt : Temperature [K]
    ps : Pressure [Pa]
    
    Returns
    -------
    es : Dew point depression [K], capped at 30 K
        
    References
    ----------
    Original Fortran code: src/mhuaes.F90 from tdpack library
    """
    
    # Small positive value to prevent log of negative numbers
    petit = 1e-10

    # Calculate intermediate value (log of normalized vapor pressure)
    hu_safe = np.maximum(petit, hu)
    vapor_pressure = foefq(hu_safe, ps)
    cte = np.log(vapor_pressure / AERK1W)

    # Calculate dew point temperature using water phase formula
    # (gemdyn uses water phase only: satues_L = .false.)
    td = (AERK3W * cte - AERK2W * TRPL) / (cte - AERK2W)

    # Dew point depression = temperature - dew point temperature
    es = tt - td

    # Apply gemdyn maximum constraint (ES_MAX = 30.0)
    es = np.minimum(es, ES_MAX)

    return es
