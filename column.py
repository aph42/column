import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from rrtm import rrtmg

import musica
import musica.mechanism_configuration as mc

class Configuration():
   def __init__(self, config_file, config_path):
# {{{
      self.config_file = config_file
      self.config_path = config_path

      # Read global configuration file
      with open(config_path + config_file, 'r') as f:
         d = json.load(f)

      self.name = d['name']
      self.version = d['version']

      # Initialize constants
      for k, v in d['constants'].items():
         self.__dict__[k] = v
         
      self.grid = d['grid']
      self.dynamics = d['dynamics']
      self.radiation = d['radiation']
      self.chemistry = config_path + d['chemistry'] + '.json'
# }}}

class ScalarVariable():
   def __init__(self, name, unit, initial_value):
   # {{{
      self.name = name
      self.value =  initial_value
      self.unit = unit
   # }}}

class ColumnVariable():
   def __init__(self, name, unit, Nz, initial_value):
   # {{{
      self.name = name
      self.Nz = Nz
      self.values = np.ones(Nz, 'd')
      self.values[:] = initial_value
      self.unit = unit
   # }}}

class State():
   def __init__(self, columns, scalars, steps = 1):
   # {{{
      self.__dict__['columns'] = {}
      self.__dict__['scalars'] = {}

      for name, c in columns.items():
         self.columns[name] = np.zeros((steps, c.Nz))
         self.columns[name][0, :] = c.values

      for name, s in scalars.items():
         self.scalars[name] = np.zeros(steps)
         self.scalars[name][0] = s.value
   # }}}

   def __setattr__(self, name, value):
   # {{{
      if name in self.columns:
         self.columns[name][:] = value

      elif name in self.scalars:
         self.scalars[name][:] = value

      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def __getattr__(self, name):
   # {{{
      if name in self.columns:
         return self.columns[name][:]
      elif name in self.scalars:
         return self.scalars[name][:]
      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

class Column():
   '''  A column model of the atmosphere, including radiative transfer, photochemistry using
        the MUSICA and TUVx components from NCAR, and vertical advection.'''
   def __init__(self, configuration):
   # {{{
      self.__dict__['grid'] = {}

      self.__dict__['variables'] = {}
      self.__dict__['scalars'] = {}

      self.__dict__['output_variables'] = {}

      self.__dict__['cfg'] = configuration

      # Initialize grid
      self.initialize_grid(**self.cfg.grid)

      # Initialize chemistry
      self.initialize_chemistry(self.cfg.chemistry)

      # Initialize dynamical quantities
      self.initialize_dynamics(**self.cfg.dynamics)

      # Initialize radiation
      self.initialize_radiation(**self.cfg.radiation)
   # }}} 

   def __setattr__(self, name, value):
   # {{{
      if name in self.grid:
         self.grid[name].values[:] = value

      elif name in self.variables:
         self.variables[name].values[:] = value

      elif name in self.scalars:
         self.scalars[name].value = value

      elif name in self.output_variables:
         self.output_variables[name].values[:] = value

      elif name in self.__dict__.keys():
         raise ValueError(f'{name} is read-only.')

      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def __getattr__(self, name):
   # {{{
      if name in self.grid:
         return self.grid[name].values[:]
      elif name in self.variables:
         return self.variables[name].values[:]
      elif name in self.scalars:
         return self.scalars[name].value
      elif name in self.output_variables:
         return self.output_variables[name].values[:]
      elif name in self.__dict__.keys():
         return self.__dict__[name]
      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def initialize_var(self, name, unit, Nz, initial_value, grid = False):
   # {{{
      if grid:
         if name in self.grid:
            raise ValueError(f'{name} has already been initialized.')
         
         self.grid[name] = ColumnVariable(name, unit, Nz, initial_value)
      else:
         if name in self.variables:
            raise ValueError(f'{name} has already been initialized.')
         
         self.variables[name] = ColumnVariable(name, unit, Nz, initial_value)
   # }}}

   def add_output(self, name, unit, Nz):
   # {{{
      if name in self.output_variables:
         raise ValueError(f'{name} has already been defined as an output variable.')
      
      self.output_variables[name] = ColumnVariable(name, unit, Nz, 0.)
   # }}}

   def initialize_scalar(self, name, unit, initial_value):
   # {{{
      if name in self.scalars:
         raise ValueError(f'{name} has already been initialized.')
      
      self.scalars[name] = ScalarVariable(name, unit, initial_value)
   # }}}

   def initialize_grid(self, *, Nz = 200, spacing = 'log_pres_equal', p_top = 0.1):
   # {{{
      # Grid parameters
      self.__dict__['Nz'] = Nz

      self.__dict__['p_top'] = p_top
      p_bot = self.cfg.p0

      z_top = self.cfg.H * np.log(p_top / self.cfg.p0) 
      z_bot = 0.
      Lz = z_top - z_bot

      # Grid must run from the top of the atmosphere down for RRTMG to work
      if spacing == 'log_pres_equal':
         zhalf = np.linspace(z_top, z_bot, self.Nz + 1)
         phalf = self.cfg.p0 * np.exp(-zhalf / self.cfg.H)
         pfull = np.sqrt(phalf[:-1]*phalf[1:])
         zfull = -self.cfg.H * np.log(pfull / self.cfg.p0)

         self.__dict__['z_top'] = z_top
         self.__dict__['dz'] = z_top / (self.Nz + 1)
      else:
         raise ValueError(f"Unrecognized grid spacing option: {spacing}")

      self.initialize_var('zhalf', 'm', self.Nz + 1, zhalf, grid = True)
      self.initialize_var('zfull', 'm', self.Nz, zfull, grid = True)
      self.initialize_var('phalf', 'Pa', self.Nz + 1, phalf, grid = True)
      self.initialize_var('pfull', 'Pa', self.Nz, pfull, grid = True)

      self.initialize_var('vmask', '1', Nz, 1., grid = True)
   # }}}

   def initialize_dynamics(self, *, advected_tracers = ['T']):
   # {{{
      self.initialize_var('T', 'K', self.Nz, 300.)

      self.initialize_var('w', 'm s-1', self.Nz, 0.1)
   # }}}

   def initialize_radiation(self, *, scon = 1368.22, **kwargs):
   # {{{
      self.__dict__['scon'] = scon

      self.initialize_scalar('Tsfc', 'K', 300.)
      self.initialize_scalar('Emissivity', '1', 0.99)
      #self.initialize_scalar('Latitude', 'deg', 0.)
      #self.initialize_scalar('Declination', 'deg', 0.)
      self.initialize_scalar('cosz', '1', 0.9)
      self.initialize_scalar('Albedo', '1', 0.3)

      self.add_output('lw_uflx', 'W m-2', self.Nz + 1) 
      self.add_output('lw_dflx', 'W m-2', self.Nz + 1) 
      self.add_output('sw_uflx', 'W m-2', self.Nz + 1) 
      self.add_output('sw_dflx', 'W m-2', self.Nz + 1) 

      self.add_output('lw_hr', 'K d-1', self.Nz) 
      self.add_output('sw_hr', 'K d-1', self.Nz) 

      rrtmg.init(self.cfg.cp)

      self.initialize_var('dyn_hr', 'K d-1', self.Nz, 0.)
   # }}}

   def initialize_chemistry(self, mechanism_file):
   # {{{
      parser = mc.Parser()
      self.__dict__['mechanism'] = parser.parse(mechanism_file)

      for sp in self.mechanism.species:
         self.initialize_var(sp.name, 'vmr', self.Nz, 0.)
   # }}}

   def get_internal_state(self, n = 1):
   # {{{
      return State(self.variables, self.scalars, n)
   # }}}

   def create_output_state(self, n = 1):
   # {{{
      return State(self.variables | self.output_variables, self.scalars, n)
   # }}}

   def save_state(self, state, output, i_state, i_out):
   # {{{
      for c in state.columns: 
         output.columns[c][i_out, :] = state.columns[c][i_state, :]

      for s in state.scalars: 
         output.scalars[s][i_out] = state.scalars[s][i_state]
   # }}}

   def compute_radiation(self, state, output, i_state, i_out):
   # {{{
      # Helper function to reshape grid arrays
      def _g(v): return np.asfortranarray(v.reshape(1, -1).copy(), 'd')

      # Helper function to reshape column arrays
      def _c(v): return np.asfortranarray(v[i_state, :].reshape(1, -1).copy(), 'd')

      # Helper function to reshape scalar quantities
      def _s(v): return np.asfortranarray(np.array(v[i_state:i_state + 1]), 'd')

      pfull = _g(self.pfull)
      phalf = _g(self.phalf)

      T   = _c(state.T)
      CO2 = _c(state.CO2)
      O3  = _c(state.O3 )
      H2O = _c(state.H2O)
      #H2Or = _c(state.H2Or)

      TSfc = _s(state.Tsfc)
      Emis = _s(state.Emissivity)
      alb  = _s(state.Albedo)
      #lat  = _s(state.Latitude)
      #dec  = _s(state.Declination)
      cosz  = _s(state.cosz)

      lw = rrtmg.rrtmg_lw(pfull, phalf, \
                          T, TSfc, Emis, \
                          CO2,  H2O,  O3)

      output.lw_uflx[i_out, :] = lw['uflxlw'][0, :]
      output.lw_dflx[i_out, :] = lw['dflxlw'][0, :]
      output.lw_hr[i_out, :]   = lw['lwhr'][0, :]

      #sw = rrtmg.rrtmg_sw_dm(pfull, phalf, \
            #T, TSfc,  c.scon, 4, \
            #alb, lat, dec, \
            #CO2, H2O, O3)

      sw = rrtmg.rrtmg_sw(pfull, phalf, \
            T, TSfc,  self.scon, \
            alb, cosz, \
            CO2, H2O, O3)

      output.sw_uflx[i_out, :] = sw['uflxsw'][0, :]
      output.sw_dflx[i_out, :] = sw['dflxsw'][0, :]
      output.sw_hr[i_out, :]   = sw['swhr'][0, :]

      #cl = 20.
      #output.lw_hr           = np.clip(output.lw_hr, -cl, cl)
      #output.sw_hr           = np.clip(output.ls_hr, -cl, cl)
   # }}}

   def solve(self, nsteps, dt, output_freq = 1):
   # {{{
      # Output grid
      nout   = int(nsteps / output_freq) + 1
      times  = np.arange(nout) * dt * output_freq

      s0 = self.get_internal_state()
      o0 = self.create_output_state(nout)

      i = 0
      i_step = 0
      i_out = 0

      self.compute_radiation(s0, o0, 0, i_out)
      self.save_state(s0, o0, 0, i_out)

      i_out += 1

      while i < nsteps:
         self.compute_radiation(s0, o0, 0, i_out)
         dT = o0.lw_hr[i_out, :] + o0.sw_hr[i_out, :] + self.dyn_hr[:]
         s0.T[0, :] += dT * dt * self.vmask[:]

         i += 1
         i_step += 1

         if i_step == output_freq:
            self.save_state(s0, o0, 0, i_out)
            i_out += 1
            i_step = 0

      return times, o0
   # }}}

