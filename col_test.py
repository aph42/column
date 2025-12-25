import numpy as np
from matplotlib import pyplot as plt

import pygeode as pyg

import column

import adiabat
from rrtm import astr

def test_oscillation():
# {{{
   c = column.Configuration('chapman_rad.json', 'configs/')
   col = column.Column(c)

   col.H2O[140:160] = np.hanning(20)**2
   col.w[1:-1] = 0e-2# + 1e-2 * (col.zhalf[1:-1] / col.z_top)
   col.wp[1:-1] = 2e-2
   col.omega = 2*np.pi / 30.

   ts, o0 = col.solve(1000, 0.05)

   ds = column.to_pyg(col, ts, o0)

   # Lower boundary
   z_low = col.zfull[140] + 86400.*col.wp[140] / col.omega * pyg.sin(col.omega * ds.time)
   z_upp = col.zfull[160] + 86400.*col.wp[160] / col.omega * pyg.sin(col.omega * ds.time)

   plt.ioff()

   ax = pyg.showvar(ds.H2O, nozero=True)

   pyg.vplot(z_low, fmt = 'C0', axes = ax.axes[0], lblx=False)
   pyg.vplot(z_upp, fmt = 'C0', axes = ax.axes[0], lblx=False)
   #pyg.showlines([z_low, z_upp], fig=2)
   #pyg.showlines([ds.H2O(i_time = i, zfull=(4000, 18000)) for i in [0,1,2,-1]], fig=2)

   plt.ion()
   ax.render(1)

   sm = ds.H2O.sum('zfull')
   st = ds.H2O.stdev('zfull')
   mn = ds.H2O.min('zfull')
   mx = ds.H2O.max('zfull')
   axs = pyg.showlines([v - v(s_time = 0) for v in [sm, st, mn, mx]], \
                 labels = ['Sum', 'Std. Dev.', 'Minimum', 'Maximum'], fig=3)
   axs.setp(title = 'Tracer statistic relative to initial value', ylabel = '')
   #pyg.showvar(ds.H2O, fig=1, nozero=True)

   return ax, ds
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
