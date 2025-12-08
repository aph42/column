import numpy as np
from matplotlib import pyplot as plt

import pygeode as pyg

import column

import adiabat
from rrtm import astr

def test_rad(lat = 20., date = '2022-10-15'):
# {{{
   c = Configuration('chapman_rad.json', 'configs/')
   col = Column(c)

   col.CO2[:]  = 417e-6
   col.O3[:]   = 10e-6
   col.H2O[:]  = 0e-6

   col.vmask[:] = 0.5 * (1 - np.tanh((col.pfull - 150) / 20.))

   return col
# }}}

def solve_col(col, tf, ts = 0.5):
# {{{
   dsmrg = make_profiles(col)

   col.T[:] = dsmrg.t[:]
   col.CO2[:]  = 417e-6
   col.O3[:]   = dsmrg.O3_cl[:]
   col.H2O[:]  = dsmrg.H2O_cl[:]
   #col.H2Or[:]  = dsmrg.H2O_cl[:]

   s0 = col.get_internal_state()
   o0 = col.create_output_state()
   col.compute_radiation(s0, o0, 0, 0)

   col.dyn_hr = -(o0.lw_hr + o0.sw_hr)

   #col.CO2[:]  = 420e-6
   col.H2O[:]  = dsmrg.H2O_sat[:]
   col.O3[:]   = dsmrg.O3_tc[:]
   #col.H2Or[:]  = dsmrg.H2O_sat[:]

   ts, output = col.solve(tf, ts)
   #return ts, output, o0

   dsr = to_pyg(col, [0.], o0)
   dsc = to_pyg(col, ts, output)

   return dsr, dsc 
# }}}

def to_pyg(col, ts, out, init = None):
# {{{
   time = pyg.Yearless(ts, units = 'days', startdate = dict(year = 1, day = 0))
   pfull = pyg.Pres(col.pfull, name = 'pfull')
   phalf = pyg.Pres(col.phalf, name = 'phalf')

   def add_var(name, values, unit):
      if values.shape[1] == col.Nz:
         axs = (time, pfull,)
      elif values.shape[1] == col.Nz + 1:
         axs = (time, phalf,)
      else:
         raise ValueError(f'Variable {name} has unrecognized length.')

      v = pyg.Var(axs, name = name, values = values[:].copy())
      v.units = unit
      return v

   vs = []
   for name, vals in out.columns.items():
      if init is None:
         v = vals
      else:
         v = vals - init.columns[name][:]

      vs.append(add_var(name, v, ''))#col.variables[name].unit))

   #for name, var in col.output_variables.items():
      #vs.append(add_var(name, var))

   return pyg.asdataset(vs)
# }}}

def run_test():
# {{{
   col = column.ColumnBase('chapman.yaml', 0, 55e3, 166)

   dsr = pyg.open('/data/RO/sample_rrtm_profile.nc')

   col.pfull = dsr.pf[:]
   col.phalf = dsr.ph[:]

   col.T[:] = dsr.tf[:]
   col.CO2[:] = dsr.co2[:]
   col.O3[:] = dsr.o3[:]
   col.H2O[:] = dsr.h2o[:]

   col.Tsfc = dsr.tsfc[()]
   col.Emissivity = dsr.emis[()]
   col.cosz = dsr.cosz[()]
   col.Albedo = dsr.alb[()]

   s0 = col.get_internal_state()
   o0 = col.create_output_state()
   col.compute_radiation(s0, o0, 0, 0)

   swhr = pyg.Var((dsr.pfull,), name = 'swhr_col', values = o0.sw_hr[0, :])
   lwhr = pyg.Var((dsr.pfull,), name = 'lwhr_col', values = o0.lw_hr[0, :])

   pyg.showlines([swhr, lwhr, dsr.swhr, dsr.lwhr], fmts = ['C0', 'C1', 'k--', 'k--'], fig=4)
# }}}
