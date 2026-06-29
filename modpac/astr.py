import numpy as np
import datetime

# All calculations based on Wikipedia Position of the Sun article retrieved 11 Nov 2016
# In all functions n is the number of days since 1 Jan 2000

# Conversion factors to convert degrees to radians or hours to radians
d2r = np.pi / 180.
h2r = np.pi / 12.

def date_to_n(date):
# {{{
   ''' Returns n, the number of days since 2000-01-01. Expects 
   date in ISO 8601 format 'YYYY-MM-DD'; e.g. '2010-10-23'. '''

   dtref = datetime.datetime.fromisoformat('2000-01-01')
   dt = datetime.datetime.fromisoformat(date)
   return (dt - dtref).days
# }}}

def mean_longitude(n):
# {{{
   ''' Returns the mean longitude of the sun (in degrees). '''
   return 280.460 + 0.9856474*n
# }}}

def mean_anomaly(n):
# {{{
   ''' Returns the mean anomaly of the sun (in degrees). '''
   return 357.528 + 0.9856003*n
# }}}

def ecliptic_longitude(n):
# {{{
   ''' Returns the ecliptic longitude (in degrees). '''
   g = d2r * mean_anomaly(n) # Mean anomaly in radians
   return mean_longitude(n) + 1.915*np.sin(g) + 0.020 * np.sin(2*g)
# }}}

def sun_earth_distance(n):
# {{{
   ''' Returns the distance of the sun from the earth. '''
   g = d2r * mean_anomaly(n) # Mean anomaly in radians
   return 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2*g)
# }}}

def obliquity(n):
# {{{
   ''' Returns the obliquity of the ecliptic (in degrees). '''
   return 23.439 - 0.0000004*n
# }}}

def right_ascension(n):
# {{{
   ''' Returns the right ascension (in degrees). '''
   eps = d2r * obliquity(n)
   lam = d2r * ecliptic_longitude(n)
   return np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)) / d2r
# }}}

def declination(n):
# {{{
   ''' Returns the declination (in degrees). '''
   eps = d2r * obliquity(n)
   lam = d2r * ecliptic_longitude(n)
   return np.arcsin(np.sin(eps) * np.sin(lam)) / d2r
# }}}

def sunrise(lat, n):
# {{{
   ''' Returns the sunset/sunrise hour angle. '''
   return sunrise_from_declination(lat, declination(n))
# }}}

def sunrise_from_declination(lat, dec):
# {{{
   ''' Returns the sunset/sunrise hour angle given latitude and declination. '''
   dc = d2r * dec
   lt = d2r * lat

   ch = -np.tan(dc) * np.tan(lt)
   hd = np.arccos(np.clip(ch, -1, 1))
   return hd / h2r
# }}}
def zenith(lat, lon, h, n):
# {{{
   ''' Returns the solar zenith angle (in degrees) at a given latitude, 
   longitude, hour (UTC), and date (days since 2000-01-01).'''
   # h is hour of the day, UTC
   dec = declination(n)
   h = h + d2r * lon / h2r

   return zenith_from_declination(lat, dec, h)
# }}}

def zenith_from_declination(lat, dec, h):
# {{{
   ''' Returns the solar zenith angle (in degrees) at a given latitude and 
   local hour for fixed declination (in degrees). '''

   dc = d2r * dec
   lt = d2r * lat
   hr = h2r * h

   cosz = np.sin(dc) * np.sin(lt) - np.cos(dc) * np.cos(lt) * np.cos(hr)
   return np.arccos(cosz) / d2r
# }}}

def zenith_dailymean_equivalent(lat, n):
# {{{
   # Return zenith angle whose cosine is equal to the daily mean of the
   # cosine of the zenith angle
   dec = declination(n)

   dc = d2r * dec
   lt = d2r * lat

   sh = h2r * sunrise_from_declination(lat, dec)

   A = np.sin(dc) * np.sin(lt)
   B = np.cos(dc) * np.cos(lt)

   cosz_dm = (A * sh + B * np.sin(sh)) / np.pi
   return np.arccos(cosz_dm) / d2r
# }}}

def get_quadrature_angles_and_weights_cheb_asym(lat, dec, nquad):
# {{{
   ''' Returns the zenith angles and quadrature weights to compute
   daily averaged radiative transfer using Chebychev quadrature. 
   Does not assume that the quantity being averaged over is symmetric
   respect to local noon (i.e. a function of the cosine of the zenith
   angle only).'''

   dc = d2r * dec
   lt = d2r * lat

   sh = h2r * sunrise_from_declination(lat, dec)

   A = np.sin(dc) * np.sin(lt)
   B = np.cos(dc) * np.cos(lt)

   #czm = np.maximum(A + B, A - B)
   #print(f"A: {A:.3f}, B: {B:.3f}")
   #print(f"day length: {sh / h2r:.2f}, minimum zenith angle: {np.arccos(czm) / d2r:.2f}")

   j = np.arange(nquad) + 1
   xj = np.cos((2*j - 1)*np.pi / (2. * nquad))

   angles = np.arccos(A + B * np.cos(sh * xj)) / d2r
   weights = sh / (2 * nquad) * np.sqrt(1 - xj**2)

   return angles, weights
# }}}

def get_quadrature_angles_and_weights_gauss_asym(lat, dec, nquad):
# {{{
   ''' Returns the zenith angles and quadrature weights to compute
   daily averaged radiative transfer using Gaussian quadrature. 
   Does not assume that the quantity being averaged over is symmetric
   respect to local noon (i.e. a function of the cosine of the zenith
   angle only).'''

   from scipy.special import roots_legendre

   dc = d2r * dec
   lt = d2r * lat

   sh = h2r * sunrise_from_declination(lat, dec)

   A = np.sin(dc) * np.sin(lt)
   B = np.cos(dc) * np.cos(lt)

   xj, wj = roots_legendre(nquad)

   angles = np.arccos(A + B * np.cos(sh * xj)) / d2r
   weights = sh / (2 * np.pi) * wj

   return angles, weights
# }}}

def get_quadrature_angles_and_weights_cheb(lat, dec, nquad):
# {{{
   ''' Returns the zenith angles and quadrature weights to compute
   daily averaged radiative transfer using Chebychev quadrature. '''

   dc = d2r * dec
   lt = d2r * lat

   sh = h2r * sunrise_from_declination(lat, dec)

   A = np.sin(dc) * np.sin(lt)
   B = np.cos(dc) * np.cos(lt)

   j = np.arange(nquad) + 1
   xj = np.cos((2*j - 1)*np.pi / (4. * nquad))

   angles = np.arccos(A + B * np.cos(sh * xj)) / d2r
   weights = sh / (2 * nquad) * np.sqrt(1 - xj**2)

   return angles, weights
# }}}

def get_quadrature_angles_and_weights_gauss(lat, dec, nquad):
# {{{
   ''' Returns the zenith angles and quadrature weights to compute
   daily averaged radiative transfer using Gaussian quadrature.''' 

   from scipy.special import roots_legendre

   dc = d2r * dec
   lt = d2r * lat

   sh = h2r * sunrise_from_declination(lat, dec)

   A = np.sin(dc) * np.sin(lt)
   B = np.cos(dc) * np.cos(lt)

   xj, wj = roots_legendre(2*nquad)

   angles = np.arccos(A + B * np.cos(sh * xj[:nquad])) / d2r
   weights = sh / np.pi * wj[:nquad]

   return angles, weights
# }}}
