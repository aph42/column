import numpy as np
from matplotlib import pyplot as plt

import pygeode as pyg

import column

import adiabat
from rrtm import astr

def test_oscillation(C = 0.2, shape = 'gaussian', periods = 3):
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

   return ds
# }}} 

def plot_origins(dt = 2000.):
# {{{
   c = column.Configuration('adv.json', 'configs/')
   col = column.Column(c)

   col.w[1:-1] = 2e-2
   col.wp[1:-1] = 2e-2

   s0 = col.get_internal_state(n = 2)
   col.update_externals(s0, 0, 0)
   col.update_externals(s0, 1, dt/86400.)

   print(s0.w[:, 15], dt)

   zorg = col.get_origins(s0, 0, 1, dt, I=2)

   # Half-point origins
   fg = plt.figure(1, (2., 4.))
   fg.clf()

   ax = fg.add_subplot(111)

   print(zorg[15] - col.zfull[15], -s0.w[0, 15] * dt)

   ax.plot(zorg - col.zfull, col.zfull, 'k+')
# }}}

def test_adv():
# {{{
   c = Configuration('chapman_rad.json', 'configs/')
   col = Column(c)

   col.H2O[50:60] = 1.

   ts, o0 = col.solve(100, 0.01)

   ds = col.to_pyg(col, ts, o0)

   pyg.showvar(ds.H2O, fig=1)
   pyg.showvar(ds.H2O(i_time = 1), fig=2)

   return ds
# }}}
