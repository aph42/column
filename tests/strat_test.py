import numpy as np
from matplotlib import pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve

import pygeode as pyg

import modpac

# import adiabat
#from rrtm import astr

import xarray as xr

def test_chapman_mechanism(ndays=200, Nz=200, daily_mean = False, kappa = 0.):
# {{{
   c = modpac.Configuration('chapman_rad')
   c.radiation['active'] = True
   c.chemistry['active'] = True
   c.photolysis['active'] = True
   c.grid['Nz'] = Nz

   c.dynamics['kappa_zz'] = kappa

   if daily_mean:
      c.radiation['zenith'] = 'daily_mean_computed'
      c.radiation['num_radiation_calls'] = 2
      ts = 7200
   else:
      c.radiation['zenith'] = 'diurnal_cycle'
      ts = 600

   col = modpac.ModPAC(c, output_path_template = '/data/QOSM/chapman/{rundate}/')

   pfull = 1000. * np.exp(-col.zfull / col.cfg.H)
   phalf = 1000. * np.exp(-col.zhalf / col.cfg.H)
   pf = pyg.Pres(pfull)
   ph = pyg.Pres(phalf)

   def to_col(var, pr):
      vi = var.interpolate('pres', pr, inx = pyg.log(var.pres), outx = pyg.log(pr), d_above = 0., d_below = 0.)
      return vi[:]

   ref = pyg.open('/data/QOSM/basic_state_from_rce.nc')
   conv = pyg.open('/data/QOSM/conv_profile.nc')

   # mixing ratios
   col.M[:] = 1.
   col.O2[:] = 0.21
   col.CO2[:] = 400e-6
   col.O3[:] = 1e-6
   col.H2O[:] = to_col(conv.H2O, pf)
   col.H2O.fixed = True

   col.w = 0.
   # Surface properties
   col.Tsfc = 300.
   col.emissivity = 0.99
   col.albedo  = 0.3

   #col.T_conv[:]   = to_col(conv.T  , pf)
   #col.H2O_conv[:] = to_col(conv.H2O, pf)

   tspd = int(86400 // ts)

   o0 = col.solve(tspd * ndays, ts, 1, write_output = False)

   ds = modpac.to_pyg(col, o0)

   return col, ds
# }}}

def test_strat_mechanism(ndays=200, Nz=200, kappa = 0.):
# {{{
   c = modpac.Configuration('strat_rad')
   c.radiation['active'] = True
   c.chemistry['active'] = True
   c.photolysis['active'] = True
   c.grid['Nz'] = Nz

   c.dynamics['kappa_zz'] = kappa#1e-2

   print(c.dynamics['kappa_zz'])

   #c.radiation['zenith'] = 'fixed_specified'
   #c.radiation['solar_zenith_angle'] = 54.

   col = modpac.ModPAC(c, output_path_template = 'data/QOSM/strat_full/{rundate}/')

   pfull = 1000. * np.exp(-col.zfull / col.cfg.H)
   phalf = 1000. * np.exp(-col.zhalf / col.cfg.H)
   pf = pyg.Pres(pfull)
   ph = pyg.Pres(phalf)

   ref = pyg.open('data/QOSM/basic_state_from_rce.nc')
   prk = pyg.open('data/QOSM/park_noy_plot.nc')
   conv = pyg.open('data/QOSM/conv_profile.nc')
   gps = pyg.open('data/QOSM/ref_gps_blended.nc')

   # mixing ratio 
   col.M[:] = 1.
   col.O2[:] = 0.21
   col.N2[:] = 0.78
   col.O1D[:] = 1e-15
   col.O[:] = 1e-15
   col.HO2[:] = 1e-10
   col.H[:] = 1e-10
   col.H2[:] = 1e-9
   col.H2O2[:] = 1e-11
   col.HO2NO2[:] = 1e-14
   col.NO2[:] = 50e-12 # 
   col.NO[:]  = 50e-12 # 
   # col.HCL[:] = 6e-11 # BS05 at 15 km

   def to_col(var, pr):
      vi = var.interpolate('pres', pr, inx = pyg.log(var.pres), outx = pyg.log(pr), d_above = 0., d_below = 0.)
      return vi[:]

   col.O3[:] = to_col(gps.O3,pf) # 20e-9
   col.H2O[:] = to_col(conv.H2O,pf)
    
   # Better idea is to solve analytically for N2O given w, a lower boundary value, and a calculation of jn2o
   col.w[:] = to_col(ref.w, ph) * 1e-3
   col.w[col.w < 0] = 6e-5

   col.w[:] = col.w[:] * (0.5 + 0.5*np.tanh((col.zhalf - 7000) / 1500))
   col.w[1] = 0.
   col.w[-1] = 0.

   #col.w[5:-5] = 0.0003
   col.wp[:] =  0. + 0j

   # col.O3[:] = to_col(ref.O3, pf)
   #col.variables['O3'].fixed = True

   col.N2O[:] = solve_steady_n2o(col)
   #col.N2O[:] = to_col(prk.N2O, pf) *1e-9
   #col.variables['N2O'].fixed = True

   #col.w[5:-5] = 0.

   #col.TSfc = 300.

   # dst = pyg.open('/local1/storage1/alm334/tuv-x/sample_photolysis_rate_constants.nc')(time=0)

   #dss = []
   #for c in [400e-6]:
   col.CO2[:] = 400e-6

   col.T_conv[:]   = to_col(conv.T  ,pf)
   col.H2O_conv[:] = to_col(conv.H2O,pf)
    
   #return col
   #ts, o0 = col.solve(26*6, 600)
   o0 = col.solve(24 * ndays, 1200, 1, write_output = True)
   #ts, o0 = col.solve(112, 3*600, 1)

   ds = modpac.to_pyg(col, o0)
   #dss.append(ds)

   #pyg.showvar(ds.T, fig=3)
   return col, ds
# }}}

def plot_noy(ds, fig=3):
# {{{
   pfull = pyg.Pres(1e3*pyg.exp(-ds.zfull / 7e3)[:])

   ds = ds.replace_axes(zfull = pfull)(pres = (300, 1))

   prk = pyg.open('data/QOSM/park_noy_plot.nc')

   NOx = ds.NO + ds.NO2
   NOy = ds.NO + ds.NO2 + ds.HNO3 + 2*ds.N2O5

   plt.ioff()

   axn = pyg.plot.AxesWrapper()
   ax  = pyg.plot.AxesWrapper()

   pyg.vplot(1e9*ds.N2O,  'o',  axes = axn, mew = 1., ms = 5., mec = '0.4', mfc = 'w', label = 'N2O')
   pyg.vplot(prk.N2O, 'o',  axes = axn, mew = 1., ms = 5., mec = '0.4', mfc = '0.4', label = 'N2O (Park et al.)')

   axn.setp(xscale = 'log', xlim = (4, 4e2))
   axn.setp_xaxis(major_formatter = plt.LogFormatter(), \
                  major_locator = plt.LogLocator(subs = [1.]))
   axn.legend(loc = 'lower center', frameon = False, ncols = 2)
                                         
   pyg.vplot(1e9*NOx,    '--', axes = ax, lw = 2., c = '0.6', label = 'NOx')
   pyg.vplot(1e9*NOy,    '-',  axes = ax, lw = 2., c = '0.8', label = 'NOy')
                                         
   pyg.vplot(1e9*ds.NO,   'o-', axes = ax, c = 'r',  ms = 4., label = 'NO')
   pyg.vplot(1e9*ds.NO2,  'd-', axes = ax, c = 'g',  ms = 4., label = 'NO2')
   pyg.vplot(1e9*ds.HNO3, 's-', axes = ax, c = 'b',  ms = 4., label = 'HNO3')
   pyg.vplot(2e9*ds.N2O5, 'p-', axes = ax, c = 'C1', ms = 4., label = '2xN2O5')

   pyg.vplot(prk.NOx, '--', axes = ax, lw = 1., c = '0.2')
   pyg.vplot(prk.NOy,  '-', axes = ax, lw = 1., c = '0.4')
                                     
   pyg.vplot(prk.NO,   'o', axes = ax, c = 'r',  ms = 6.)
   pyg.vplot(prk.NO2,  'd', axes = ax, c = 'g',  ms = 6.)
   pyg.vplot(prk.HNO3, 's', axes = ax, c = 'b',  ms = 6.)
   pyg.vplot(prk.N2O5, 'p', axes = ax, c = 'C1', ms = 6.)

   ax.setp(xscale = 'log', xlim = (1e-4, 3e1))
   ax.setp_xaxis(major_formatter = plt.LogFormatter(), \
                 major_locator = plt.LogLocator(subs = [1.]))

   ax.legend(loc = 'lower center', frameon = False, ncols = 2)

   ax = pyg.plot.grid([[axn, ax]])

   plt.ion()

   ax.render(fig)

# }}}

def test_n2o_column(days = 100, Nz=200, kappa = 0.):
# {{{
   c = modpac.Configuration('n2o_rad')
   c.radiation['active'] = True
   c.chemistry['active'] = True
   c.photolysis['active'] = True
   #c.convection['active'] = True
   #c.humidity['active'] = True
   c.grid['Nz'] = Nz

   c.dynamics['kappa_zz'] = kappa

   ref = pyg.open('data/QOSM/basic_state_from_rce.nc')
   prk = pyg.open('data/QOSM/park_noy_plot.nc')

   col = modpac.ModPAC(c, output_path_template = 'data/QOSM/{name}')

   pfull = 1000. * np.exp(-col.zfull / col.cfg.H)
   phalf = 1000. * np.exp(-col.zhalf / col.cfg.H)
   pf = pyg.Pres(pfull)
   ph = pyg.Pres(phalf)

   def to_col(var, pr):
      vi = var.interpolate('pres', pr, inx = pyg.log(var.pres), outx = pyg.log(pr), d_above = 0., d_below = 0.)
      return vi[:]

   # mixing ratio
   col.M[:] = 1.
   col.O2[:] = 0.21
   col.N2[:] = 0.78
   col.O1D[:] = 0
   col.CO2[:] = 400e-6

   col.O3[:] = to_col(ref.O3, pf)

   col.H2O[:] = 0.02 / 0.622 * np.exp(-col.zfull[:]/2000) # approximate profile with 2 km scale height
   col.H2O[col.H2O < 4e-6] = 4e-6

   # Better idea is to solve analytically for N2O given w, a lower boundary value, and a calculation of jn2o
   col.w[:] = to_col(ref.w, ph) * 1e-3
   col.w[col.w < 0] = 1e-5
   col.w[0] = 0.
   col.w[-1] = 0.
   col.wp[:] =  0.

   #col.N2O[:] = 3.26e-7
   #col.N2O[:] = to_col(prk.N2O*1e-9, pf)
   col.N2O[:] = solve_steady_n2o(col)

   col.T[:] = to_col(ref.T, pf)
   col.Tsfc = 300.

   for sp in col.species:
      col.columns[sp].fixed: print(sp)

   #o0 = col.solve(8 * 100, 3*3600, write_output = False, restart_file = '/data/QOSM/n2o_rad.json_2026-05-31/n2o_rad.json_2026-05-31_restart_000000799.nc')
   o0 = col.solve(8 * days, 3*3600, write_output = False)

   ds = modpac.to_pyg(col, o0)

   return col, ds
# }}}

def solve_steady_n2o(col, n2o_0 = 3.26e-7, jn2o = None):
# {{{
   if jn2o is None:
      s0 = col.create_internal_state(n = 1)
      o0 = col.create_output_state(np.arange(1))
      jn2o = o0.jn2o[0, :]

   #p_org = col.cfg.p0 * np.exp(-col.z_full / col.cfg.H)
   #nafull = p_org * 100 / (col.cfg.R * state.T[0, :])

   col.compute_photolysis(s0, o0, col.zfull, 0, 0)

   wfull = np.interp(col.zfull[::-1], col.zhalf[::-1], col.w[::-1])[::-1]
   #wfull = np.interp(col.zfull, col.zhalf, col.w)

   dt = jn2o / np.sqrt(wfull**2 + 1e-12)

   #tau = np.cumsum(dt, col.zfull[:])
   #print(dt.shape, col.zfull[:].shape, tau)

   #dz = np.diff(col.zhalf)
   dz = np.diff(col.zfull)

   tau = np.cumulative_sum((dt[1:] * dz), include_initial = True)
   tau = -tau[-1] + tau

   #tau = np.cumsum(dt * dz)
   #tau -= tau[-1]

   #return tau
   return n2o_0 * np.exp(-tau)
# }}}

def plot_n2o(col, ds, fig = 5):
# {{{
   plt.ioff()

   n2i = ds.N2O(si_time = 0)
   n2f = ds.N2O(s_time = 200)

   n2s = solve_steady_n2o(col)

   n2s = pyg.Var((ds.zfull,), values = n2s, name = 'n2o')

   ns = [ds.N2O(s_time = t) for t in [0, 50, 100, 150, 200]] + [n2s]
   ax = pyg.showlines(ns)
   #ax = pyg.showlines([n2i, n2f, n2s], labels = ['init', 'final', 'steady'])
   #ax = pyg.showlines([n2s], labels = ['init', 'final', 'steady'])

   plt.ion()

   ax.render(fig)
# }}}

def plot_basicstate(ds, times = None, fig=1):
# {{{
   ref = pyg.open('/data/QOSM/basic_state_from_rce.nc')
   prk = pyg.open('/data/QOSM/park_noy_plot.nc')

   pres = pyg.Pres(1000 * np.exp(-ds.zfull[:] / 7e3))

   if times is None:
      # Default to last 200 days
      times = (ds.time[-1] - 200, ds.time[-1])

   dst = ds(m_time = times).replace_axes(zfull = pres)

   plt.ioff()

   psize = (3.5, 3)

   axt = pyg.showlines([dst.T, ref.T], fmts = ['C0', 'k'], labels = ['run', 'reference'], size = psize)
   axt.setp(title = 'T', xlim = (180, 320))

   axo3 = pyg.showlines([1e6*dst.O3, 1e6*ref.O3], fmts = ['C0', 'k'], labels = ['run', 'reference'], size = psize)
   axo3.setp(title = 'O3', xlim = (0, 11.))

   axh2o = pyg.showlines([1e6*dst.H2O], fmts = ['C0'], labels = ['run'], size = psize)
   axh2o.setp(title = 'H2O', xlim = (2, 2e2), xscale = 'log')

   axn2o = pyg.showlines([1e9 * dst.N2O, prk.N2O], fmts = ['C0', 'k'], labels = ['run', 'Park et al. 2017'], size = psize)
   axn2o.setp(title = 'N2O', xlim = (1, 450), xscale = 'log')

   plt.ion()

   ax = pyg.plot.grid([[axt, axo3], [axh2o, axn2o]])

   ax.render(fig)
# }}}

def infer_jn2o(col):
# {{{
   dz = np.diff(col.zhalf)

   ip = np.arange(col.Nz) + 1
   im = np.arange(col.Nz) - 1

   ip[-1] = col.Nz - 1
   im[0] = 0

   dz = col.zfull[ip] - col.zfull[im]
   dn2o = col.N2O[ip] - col.N2O[im]

   dn2o = (np.log(col.N2O[1:]) - np.log(col.N2O[:-1])) / dz

   return -col.w[1:-1] * dn2o
# }}}

def test_h2o_column(ndays = 200, Nz=200, kappa = 0.):
# {{{
   c = modpac.Configuration('h2o_rad')
   c.radiation['active'] = True
   c.grid['Nz'] = Nz

   c.dynamics['kappa_zz'] = kappa#1e-2

   print(f"kappa: {c.dynamics['kappa_zz']}")

   ref = pyg.open('data/QOSM/basic_state_from_rce.nc')
   prk = pyg.open('data/QOSM/park_noy_plot.nc')

   col = modpac.ModPAC(c)
   col = modpac.ModPAC(c, output_path_template = 'data/QOSM/{name}_eq')

   pfull = 1000. * np.exp(-col.zfull / col.cfg.H)
   phalf = 1000. * np.exp(-col.zhalf / col.cfg.H)
   pf = pyg.Pres(pfull)
   ph = pyg.Pres(phalf)

   def to_col(var, pr):
      vi = var.interpolate('pres', pr, inx = pyg.log(var.pres), outx = pyg.log(pr), d_above = 0., d_below = 0.)
      return vi[:]

   # mixing ratio
   col.M[:] = 1.
   col.O2[:] = 0.21
   col.N2[:] = 0.78
   col.CO2[:] = 400e-6

   col.O3[:] = to_col(ref.O3, pf)

   col.H2O[:] = 0.02 / 0.622 * np.exp(-col.zfull[:]/2000) # approximate profile with 2 km scale height
   col.H2O[col.H2O < 4e-6] = 4e-6

   # Better idea is to solve analytically for N2O given w, a lower boundary value, and a calculation of jn2o
   col.w[:] = to_col(ref.w, ph) * 1e-3
   col.w[col.w < 0] = 1e-5
   col.w[0] = 0.
   col.w[-1] = 0.
   col.wp[:] =  0.

   col.T[:] = to_col(ref.T, pf)
   col.Tsfc = 300.

   for name, var in col.columns.items():
      if hasattr(var, 'fixed') and var.fixed: print(name)

   o0 = col.solve(8 * ndays, 3*3600, write_output = True, restart = 'latest')

   ds = modpac.to_pyg(col, o0)

   return col, ds
# }}}

def test_restart(Nz=200):
# {{{
   c = modpac.Configuration('h2o_rad')
   c.radiation['active'] = True
   c.grid['Nz'] = Nz

   c.dynamics['kappa_zz'] = 1e-2

   print(f"kappa: {c.dynamics['kappa_zz']}")

   ref = pyg.open('data/QOSM/basic_state_from_rce.nc')
   prk = pyg.open('data/QOSM/park_noy_plot.nc')

   col = modpac.ModPAC(c)
   col = modpac.ModPAC(c, output_path_template = 'data/QOSM/{name}')

   pfull = 1000. * np.exp(-col.zfull / col.cfg.H)
   phalf = 1000. * np.exp(-col.zhalf / col.cfg.H)
   pf = pyg.Pres(pfull)
   ph = pyg.Pres(phalf)

   def to_col(var, pr):
      vi = var.interpolate('pres', pr, inx = pyg.log(var.pres), outx = pyg.log(pr), d_above = 0., d_below = 0.)
      return vi[:]

   # mixing ratio
   col.M[:] = 1.
   col.O2[:] = 0.21
   col.N2[:] = 0.78
   col.CO2[:] = 400e-6

   col.O3[:] = to_col(ref.O3, pf)

   col.H2O[:] = 0.02 / 0.622 * np.exp(-col.zfull[:]/2000) # approximate profile with 2 km scale height
   col.H2O[col.H2O < 4e-6] = 4e-6

   # Better idea is to solve analytically for N2O given w, a lower boundary value, and a calculation of jn2o
   col.w[:] = to_col(ref.w, ph) * 1e-3
   col.w[col.w < 0] = 1e-5
   col.w[0] = 0.
   col.w[-1] = 0.
   col.wp[:] =  0.

   col.T[:] = to_col(ref.T, pf)
   col.Tsfc = 300.

   ndays = 100

   o0 = col.solve(    8 * ndays, 3*3600, write_output = True,  restart = None)
   o1 = col.solve(    8 * ndays, 3*3600, write_output = True,  restart = 'latest')
   o2 = col.solve(2 * 8 * ndays, 3*3600, write_output = False, restart = None)

   return o0, o1, o2
# }}}

def profile_run():
   import cProfile
   cProfile.run('test_strat_mechanism()', 'strat_timings')


