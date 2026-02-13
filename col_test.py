import numpy as np
from matplotlib import pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve

import pygeode as pyg

import column

# import adiabat
from rrtm import astr

import xarray as xr

def interpolate_matrix(x_new, x_old, method = 'linear'):
# {{{
   ''' Constructs a CSR sparse matrix that, when applied to a vector
   of quantities defined at locations x_old, yields interpolated values
   at x_new. x_old and x_new must be sorted. '''

   #print(x_new)
   #print(x_old)

   ip = x_old.searchsorted(x_new)
   iL = np.where(ip == 0.)[0]
   iR = np.where(ip == len(x_old))[0]

   ip[iL] = 1
   ip[iR] = len(x_old) - 1

   i0 = ip - 1

   dx = x_old[ip] - x_old[i0]
   tn = (x_new - x_old[i0]) / dx
   tn[iL] = 0.
   tn[iR] = 1.

   otn = 1 - tn

   #print(i0, tn)
   #print(x_new, x_old[i0] + tn * dx)

   N = len(x_new)
   M = len(x_old)

   if method == 'linear':
      order = 2
      entries = np.zeros(N * order, 'd')
      indptr = order * np.arange(N + 1)
      cols = np.zeros(N * order, 'i')
      cols[::order]  = i0
      cols[1::order] = ip

      entries[::order] = otn
      entries[1::order] = tn

      return sparse.csr_array((entries, cols, indptr), shape = (N, M))
   elif 'cubic' in method:
      Dl = np.zeros(M - 1)
      Dc = np.zeros(M)
      Dr = np.zeros(M - 1)

      #Dr[0] = 1 / (x_old[-1] - x_old[-2])
      #Dc[-1] = 1 / (x_old[1] - x_old[0])
      #Dr[1:] = 1 / (x_old[2:] - x_old[:-2])
      #Dl[:-1] = -Dr[1:]
      #Dc[0] = -Dr[0]
      #Dl[-1] = -Dc[-1]

      Dx = np.diff(x_old)

      Dr[1:  ] =  1 / (2 * Dx[:-1])
      Dc[1:-1] = (Dx[1:] - Dx[:-1]) / (2 * Dx[1:] * Dx[:-1])
      Dl[ :-1] = -1 / (2 * Dx[1:])

      Del = sparse.diags_array([Dl, Dc, Dr], offsets = [-1, 0, 1], shape = (M,M), format = 'csr')

      order = 2
      Cdata = np.zeros(N * order, 'd')
      Ddata = np.zeros(N * order, 'd')
      indptr = order * np.arange(N + 1)
      cols = np.zeros(N * order, 'i')
      cols[::order]  = i0
      cols[1::order] = ip

      Cdata[::order] = otn**3 + 3*otn**2*tn
      Cdata[1::order] = tn**3 + 3*tn**2*otn

      Ddata[::order] = dx*otn**2*tn
      Ddata[1::order] = -dx*tn**2*otn

      C = sparse.csr_array((Cdata, cols, indptr), shape = (N, M))
      D = sparse.csr_array((Ddata, cols, indptr), shape = (N, M))

      if 'monotone' in method:
         return C, D, Del
      else:
         return C + D @ Del
   else:
      raise ValueError(f'Unrecognized method {method}.')

   return L
# }}}

def test_oscillation(C = 0.2, shape = 'gaussian', periods = 3, onestep = False):
# {{{
   c = column.Configuration('adv.json', 'configs/')
   col = column.Column(c)

   z0 = 20000.
   Dz = 2000.

   if shape == 'gaussian':
      col.H2O[:] = np.exp(-(col.zfull - z0)**2 / (2 * Dz**2))
   elif shape == 'hanning':
      iz0 = np.argmin((col.zfull - z0 - Dz)**2)
      iz1 = np.argmin((col.zfull - z0 + Dz)**2)
      col.H2O[iz0:iz1] = np.hanning(iz1 - iz0)
   elif shape == 'box':
      iz0 = np.argmin((col.zfull - z0 - Dz)**2)
      iz1 = np.argmin((col.zfull - z0 + Dz)**2)
      col.H2O[iz0:iz1] = 1.
   else:
      raise ValueError(f'Unrecognized shape "{shape}".')

   # 30 day period for testing
   period = 30. * 86400.

   wp = 2e-2

   col.w[1:-1] = 0e-2# + 1e-2 * (col.zhalf[1:-1] / col.z_top)
   col.wp[1:-1] = wp
   col.omega = 2*np.pi / period
   dz = np.min(col.zfull[:-1] - col.zfull[1:])
   
   dt = C * dz / wp

   if onestep:
      Tf = dt
      nsteps = 1
   else:
      Tf = periods * period
      nsteps = int(Tf / dt)

   dt = Tf / nsteps
   C = wp * dt / dz

   print(f'Integrating {Tf / 86400.} days, {nsteps} steps.')

   ts, o0 = col.solve(nsteps, dt)

   ds = column.to_pyg(col, ts, o0)

   # Lower boundary
   z_low = (z0 - Dz) + wp / col.omega * pyg.sin(col.omega * ds.time * 86400.)
   z_upp = (z0 + Dz) + wp / col.omega * pyg.sin(col.omega * ds.time * 86400.)

   plt.ioff()

   cm = plt.cm.Blues
   cm.set_over('0.1')
   cm.set_under('0.9')

   ch = pyg.clfdict(cdelt = 0.2, min = 0., style = 'seq', \
                    ndiv=5, nl = 0, extend = 'both', cmap = plt.cm.Blues)

   ax = pyg.showvar(ds.H2O, **ch)

   pyg.vplot(z_low, fmt = 'C1', axes = ax.axes[0], lbly=False)
   pyg.vplot(z_upp, fmt = 'C1', axes = ax.axes[0], lbly=False)

   ax.axes[0].setp(title = 'Tracer')
   #pyg.showlines([z_low, z_upp], fig=2)
   #pyg.showlines([ds.H2O(i_time = i, zfull=(4000, 18000)) for i in [0,1,2,-1]], fig=2)

   plt.ion()
   ax.render(6)

   plt.ioff()

   axs = pyg.showlines([ds.H2O(i_time = 0), ds.H2O(i_time = -1)], \
                       labels = ['Initial Time', 'Final Time'])

   plt.ion()

   axs.render(7)

   plt.ioff()
   sm = ds.H2O.sum('zfull').load()
   st = ds.H2O.stdev('zfull').load()
   mn = ds.H2O.min('zfull').load()
   mx = ds.H2O.max('zfull').load()

   axs = pyg.showlines([v - v(s_time = 0) for v in [sm, st, mn, mx]], \
               labels = ['Sum %.2f' % sm[:][0], 'Std. Dev. %.2f' % st[:][0], 
                         'Minimum %.2f' % mn[:][0], 'Maximum %.2f' % mx[:][0]])
   axs.setp(title = 'Tracer statistic relative to initial value', ylabel = '')
   plt.ion()

   axs.render(8)
   #pyg.showvar(ds.H2O, fig=1, nozero=True)

   return col, ds
# }}} 

def plot_origins(dt = 2000.):
# {{{
   c = column.Configuration('adv.json', 'configs/')
   col = column.Column(c)

   col.w[1:-1] = 2e-2 + 2e-2 * col.zhalf[1:-1] / col.z_top
   col.wp[1:-1] = 2e-2

   s0 = col.get_internal_state(n = 2)
   col.update_externals(s0, 0, 0)
   col.update_externals(s0, 1, dt/86400.)

   print(s0.w[:, 15], dt)

   zorg = col.get_origins(s0, 0, 1, dt, I=2)

   zs = np.concatenate([[col.z_bot], col.zfull[::-1], [col.z_top]])
   ip = zs.searchsorted(zorg[::-1])
   iL = np.where(ip == 0.)[0]
   iR = np.where(ip == len(zs))[0]

   #return ip, iL, iR

   # Half-point origins
   fg = plt.figure(1, (2., 4.))
   fg.clf()

   ax = fg.add_subplot(111)

   print(zorg[15] - col.zfull[15], -s0.w[0, 15] * dt)

   ax.plot(zorg - col.zfull, col.zfull, 'k+')
   #ax.plot(ip, zorg, 'k+')
# }}}

def test_advection_step(C = 0.2, Nz = 11, w0 = 0.02):
# {{{
   c = column.Configuration('adv.json', 'configs/')
   c.grid['Nz'] = Nz
   col = column.Column(c)

   z0 = 20000.
   #Dz = 2000.
   Dz = 5000.

   col.H2O[:] = np.exp(-(col.zfull - z0)**2 / (2 * Dz**2))
   #col.H2O[1] = 0.1
   #col.H2O[2] = 0.1
   #col.H2O[3] = 1.
   #col.H2O[4] = 1.
   #col.H2O[8] = -0.98
   #col.H2O[9] = -1.
   #col.H2O[10] = -1.
   #col.H2O[11] = -0.9
   #col.H2O[12] = -0.1
   #col.H2O[14] = 0.4
   #col.H2O[15] = 0.41
   #col.H2O[15] = 0.01

   #w0 = 0.0

   col.w[1:-1] = w0# + 1e-2 * (col.zhalf[1:-1] / col.z_top)
   #col.wp[1:-1] = wp
   #col.omega = 2*np.pi / period
   dz = np.min(col.zfull[:-1] - col.zfull[1:])
   
   #dt = C * dz / np.absolute(w0)
   dt = C * dz / np.sqrt(w0**2 + 0.01**2)

   s0 = col.get_internal_state(n = 2)

   z_org = col.get_origins(s0, 0, 1, dt)
   #z_org = np.linspace(col.z_top, col.z_bot, 1001)

   #print('Internal method')
   C, D, Del = col.build_advection_matrix(z_org[::-1])

   #print('Stand alone method')
   #Cp, Dp, Delp = interpolate_matrix(z_org[::-1], col.zadv, method = 'cubic monotone')

   #W = col.advect_quantity(C, D, Del, s0.H2O[0, ::-1])[::-1]
   #return W
   #print(W)

   s0.H2O[1, :] = col.advect_quantity(C, D, Del, s0.H2O[0, ::-1])[::-1]
   #H2On = col.advect_quantity(C, D, Del, s0.H2O[0, ::-1])[::-1]

   print(np.diff(np.max(s0.H2O, 1)))

   #print(C)
   #print(Cp)
   #print(D)
   #print(Dp)
   #print(Del)
   #print(Delp)

   f = plt.figure(1)
   f.clf()
   ax = f.add_subplot(111)

   ax.plot(s0.H2O[0,:], col.zfull, 'ko')
   ax.plot(s0.H2O[1,:], z_org, 'r+')
   #ax.plot(H2On, z_org, 'r+')

   #return s0, col
# }}} 

def test_reaction():
# {{{
   c = column.Configuration('mcm_test.json', 'configs/')
   c.radiation['active'] = False
   col = column.Column(c)

   tau_o3 = 1. * 86400. # 10 day decay time
   col.w[:] = 0.0
   col.w[1:-1] = 0.01
   col.O3[150:] = 10e-6

   taus = np.ones(col.Nz) / tau_o3
   taus[150:] = 0.
   col.MICMstate.set_user_defined_rate_parameters({'LOSS.O3loss': taus})

   ts, o0 = col.solve(400, 2000.)

   ds = column.to_pyg(col, ts, o0)

   plt.ioff()
   ax = pyg.showlines([ds.O3(i_time = t) for t in [0, 100, 200, 400]])

   #pyg.vplot(10e-6 * pyg.exp(-ds.time * 86400. / tau_o3), ls = '--', axes = ax)

   plt.ion()
   ax.render(1)

   plt.ioff()
   ax = pyg.showvar(ds.O3)

   plt.ion()
   ax.render(2)

   return ds

   # Lower boundary
   z_low = (z0 - Dz) + wp / col.omega * pyg.sin(col.omega * ds.time * 86400.)
   z_upp = (z0 + Dz) + wp / col.omega * pyg.sin(col.omega * ds.time * 86400.)

   plt.ioff()

   cm = plt.cm.Blues
   cm.set_over('0.1')
   cm.set_under('0.9')

   ch = pyg.clfdict(cdelt = 0.2, min = 0., style = 'seq', \
                    ndiv=5, nl = 0, extend = 'both', cmap = plt.cm.Blues)

   ax = pyg.showvar(ds.H2O, **ch)

   pyg.vplot(z_low, fmt = 'C1', axes = ax.axes[0], lbly=False)
   pyg.vplot(z_upp, fmt = 'C1', axes = ax.axes[0], lbly=False)

   ax.axes[0].setp(title = 'Tracer')
   #pyg.showlines([z_low, z_upp], fig=2)
   #pyg.showlines([ds.H2O(i_time = i, zfull=(4000, 18000)) for i in [0,1,2,-1]], fig=2)

   plt.ion()
   ax.render(6)

   plt.ioff()

   axs = pyg.showlines([ds.H2O(i_time = 0), ds.H2O(i_time = -1)], \
                       labels = ['Initial Time', 'Final Time'])

   plt.ion()

   axs.render(7)

   plt.ioff()
   sm = ds.H2O.sum('zfull').load()
   st = ds.H2O.stdev('zfull').load()
   mn = ds.H2O.min('zfull').load()
   mx = ds.H2O.max('zfull').load()

   axs = pyg.showlines([v - v(s_time = 0) for v in [sm, st, mn, mx]], \
               labels = ['Sum %.2f' % sm[:][0], 'Std. Dev. %.2f' % st[:][0], 
                         'Minimum %.2f' % mn[:][0], 'Maximum %.2f' % mx[:][0]])
   axs.setp(title = 'Tracer statistic relative to initial value', ylabel = '')
   plt.ion()

   axs.render(8)
   #pyg.showvar(ds.H2O, fig=1, nozero=True)

   return ds
# }}} 

def test_radiation_profile():
# {{{
   dsr = pyg.open('/data/RO/sample_rrtm_profile.nc')

   c = column.Configuration('mcm_test.json', 'configs/')
   c.grid['spacing'] = 'specified_pressure'
   c.grid['Nz'] = len(dsr.phalf)
   c.grid['ptop'] = dsr.phalf[0]
   c.grid['phalf'] = dsr.phalf[:]
   c.grid['pfull'] = dsr.pfull[:]

   col = column.Column(c)

   col.T[:] = dsr.tf[:]
   col.CO2[:] = dsr.co2[:]
   col.O3[:] = dsr.o3[:]
   col.H2O[:] = dsr.h2o[:]

   col.Tsfc = dsr.tsfc[()]
   col.Emissivity = dsr.emis[()]
   col.Albedo = dsr.alb[()]
   col.cosz = dsr.cosz[()]
   #col.Latitude = dsr.lat[()]
   #col.Declination = 4.631605247595053

   s0 = col.get_internal_state()
   o0 = col.create_output_state()
   col.compute_radiation(s0, o0, 0, 0)

   swhr = pyg.Var((dsr.pfull,), name = 'swhr_col', values = o0.sw_hr[0, :])
   lwhr = pyg.Var((dsr.pfull,), name = 'lwhr_col', values = o0.lw_hr[0, :])

   sw_dflx = pyg.Var((dsr.phalf,), name = 'sw_dflx', values = o0.sw_dflx[0, :])
   sw_uflx = pyg.Var((dsr.phalf,), name = 'sw_uflx', values = o0.sw_uflx[0, :])

   pyg.showlines([swhr, lwhr, dsr.swhr, dsr.lwhr], 
          fmts = ['C0', 'C1', 'k--', 'k--'], size = (3.1, 4.2), fig=4)

   pyg.showlines([sw_dflx, sw_uflx, dsr.dflxsw, dsr.uflxsw], 
          fmts = ['C0', 'C1', 'k--', 'k--'], size = (3.1, 4.2), fig=5)

   return col, s0
# }}}

def test_chapman():
# {{{
   c = column.Configuration('chapman_rad.json', 'column/configs/')
   c.radiation['active'] = True
   c.chemistry['active'] = True
   c.photolysis['active'] = True
   col = column.Column(c)

   dsr = pyg.open('/local1/storage1/aph28/sample_rrtm_profile.nc')

    # mixing ratio or mol/m-3?
   col.CO2[:] = 400e-6
   col.O2[:] = 0.21
   col.O1D[:] = 0
   col.O[:] = 0
   col.O3[:] = 0   # np.interp(col.zfull[::-1], -col.cfg.H * np.log(dsr.pfull[::-1] / col.cfg.p0), dsr.o3[::-1])[::-1] 
   col.H2O[:] = 0.02 / 0.622 * np.exp(-col.zfull[:]/2000) # approximate profile with 2 km scale height
   col.w[:] = 0.001
   col.wp[:] =  0.
   col.M[:] = 1.

   # dst = pyg.open('/local1/storage1/alm334/tuv-x/sample_photolysis_rate_constants.nc')(time=0)
   # zt_in = dst.vertical_level.values*1e3 # meters
    
   # col.MICMstate.set_user_defined_rate_parameters({'PHOTO.jO2': np.interp(col.zfull,zt_in,dst.jo2_b[:][:,0])})
   # col.MICMstate.set_user_defined_rate_parameters({'PHOTO.jO3->O': np.interp(col.zfull,zt_in,dst.jo3_b[:][:,0])})
   # col.MICMstate.set_user_defined_rate_parameters({'PHOTO.jO3->O1D': np.interp(col.zfull,zt_in,dst.jo3_a[:][:,0])})

   # col.MICMstate.set_user_defined_rate_parameters({'PHOTO.jO2': col.M[:]*0.+1e-11})
   # col.MICMstate.set_user_defined_rate_parameters({'PHOTO.jO3->O': col.M[:]*0.+1e-3})
   col.MICMstate.set_user_defined_rate_parameters({'PHOTO.jO2': col.M[:]*.0})
   col.MICMstate.set_user_defined_rate_parameters({'PHOTO.jO3->O': col.M[:]*.0})
   col.MICMstate.set_user_defined_rate_parameters({'PHOTO.jO3->O1D': col.M[:]*.0})

   ts, o0 = col.solve(500, 3600)

   ds = column.to_pyg(col, ts, o0)

   #pyg.showvar(ds.T, fig=3)
   return ds
# }}}

def compare_chapman_rates():
# {{{
   ''' Compares rate coefficients k2 and k3 of Chapman reactions
       from TS1 and chapman MUSICA v1 configuration files as of 26 Jan 2026.'''

   # Form of reaction rate temperature dependence
   # Used by MUSICA (pressure dependent term is omitted)
   def arr(T, A, B, C, D):
      return A*np.exp(C / T) * (T / D)**B

   def k2(T):
      return arr(T, 7.9e37, -2.4, 0., 300.)
   def k3(T):
      return arr(T, 2.9e24, 0., -2060, 300)

   def k2_c(T):
      return arr(T, 2.9e19, 0., 0., 300.)
   def k3_c(T):
      return arr(T, 5.7e20, 0., 0., 300.)

   # Avogadro constant
   N_A = 6.022e23

   T = np.linspace(190., 310, 1001)

   fig = plt.figure(1)
   fig.clf()

   ax1 = fig.add_subplot(211)

   ax1.semilogy(T, k2(T), label = r'TS1 k$_2$')
   ax1.semilogy(T, N_A*k2_c(T)*1e-5, label = r'Chapman k$_2 \times N_A \times 10^{-5}$ ')

   ax1.set_title(r'O + O$_2$ + M $\to$ O$_3$ + M')
   ax1.set_xlabel('K')
   ax1.legend(loc='best', frameon=False)

   ax2 = fig.add_subplot(212)

   ax2.semilogy(T, k3(T), label = r'TS1 k$_3$')
   ax2.semilogy(T, k3_c(T), label = r'Chapman k$_3$')

   ax2.set_title(r'O + O$_3$ $\to$ 2O$_2$')
   ax2.legend(loc='best', frameon=False)
   ax2.set_xlabel('K')

   fig.tight_layout()
# }}}
