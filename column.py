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
      self.chemistry = d['chemistry']
# }}}

class ScalarVariable():
   def __init__(self, name, unit, initial_value):
   # {{{
      self.name = name
      self.value =  initial_value
      self.unit = unit
   # }}}

class ColumnVariable():
   def __init__(self, name, unit, Nz, initial_value, prognostic = False, output = False):
   # {{{
      self.name = name
      self.Nz = Nz

      if type(initial_value) == np.ndarray:
         dtype = initial_value.dtype
      else:
         dtype = type(initial_value)
      self.values = np.ones(Nz, dtype)
      self.values[:] = initial_value
      self.unit = unit
      self.prognostic = prognostic
      self.output = output
   # }}}

class SpeciesVariable(ColumnVariable):
   def __init__(self, name, unit, Nz, initial_value, advect = False, prognostic = False, output = False, **properties):
   # {{{
      ColumnVariable.__init__(self, name, unit, Nz, initial_value, prognostic, output)

      self.advect = advect

      if advect:
         self.surface_flux = 0.
         self.TOA_flux = 0.

      self.properties = properties
   # }}}

class State():
   def __init__(self, columns, scalars, steps = 1):
   # {{{
      self.__dict__['columns'] = {}
      self.__dict__['scalars'] = {}

      for name, c in columns.items():
         self.columns[name] = np.zeros((steps, c.Nz))
         self.columns[name][:, :] = c.values.reshape(1, -1)

      for name, s in scalars.items():
         self.scalars[name] = np.zeros(steps)
         self.scalars[name][:] = s.value
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

def interpolate_matrix(x_new, x_old, method = 'linear'):
   # {{{
      ''' Constructs a CSR sparse matrix that, when applied to a vector
      of quantities defined at locations x_old, yields interpolated values
      at x_new. x_old and x_new must be sorted. '''

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

         L = sparse.csr_array((entries, cols, indptr), shape = (N, M))
      elif method == 'cubic':
         Dl = np.zeros(M - 1)
         Dc = np.zeros(M)
         Dr = np.zeros(M - 1)

         Dr[0] = 1 / (x_old[-1] - x_old[-2])
         Dc[-1] = 1 / (x_old[1] - x_old[0])
         Dr[1:] = 1 / (x_old[2:] - x_old[:-2])
         Dl[:-1] = -Dr[1:]
         Dc[0] = -Dr[0]
         Dl[-1] = -Dc[-1]

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

         L = C + D @ Del

         N = L.sum(0).reshape(-1, 1)
         L = L / N
      else:
         raise ValueError(f'Unrecognized method {method}.')

      return L
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
      self.initialize_chemistry(**self.cfg.chemistry)

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
      
      self.output_variables[name] = ColumnVariable(name, unit, Nz, 0., output = True)
   # }}}

   def initialize_scalar(self, name, unit, initial_value):
   # {{{
      if name in self.scalars:
         raise ValueError(f'{name} has already been initialized.')
      
      self.scalars[name] = ScalarVariable(name, unit, initial_value)
   # }}}

   def initialize_grid(self, *, spacing = 'log_pressure_equal', Nz = 200, p_top = 0.1, **kwargs):
   # {{{
      ''' Initialize the column grid. The levels must run from the top of the atmosphere down
      (increasing pressure). The grid spacing can be set in various ways based on the choice
      of the argument `spacing`. Possible values include
       - 'log_pressure_equal' (default). Specify p_top, the pressure at the top of the domain in hPa, and
            the number of levels.
       - 'specified_pressure'. Provide arrays for phalf and pfull. '''

      # Grid must run from the top of the atmosphere down for RRTMG to work
      if spacing == 'log_pressure_equal':
         p_bot = self.cfg.p0

         z_top = -self.cfg.H * np.log(p_top / self.cfg.p0) 
         z_bot = 0.
         Lz = z_top - z_bot

         zhalf = np.linspace(z_top, z_bot, Nz + 1)
         phalf = self.cfg.p0 * np.exp(-zhalf / self.cfg.H)
         pfull = np.sqrt(phalf[:-1]*phalf[1:])
         zfull = -self.cfg.H * np.log(pfull / self.cfg.p0)

         #self.__dict__['dz'] = z_top / (self.Nz + 1)

      elif spacing == 'specified_pressure':
         # Full and half-levels must be specified
         pfull = kwargs.pop('pfull')
         phalf = kwargs.pop('phalf')
         zfull = -self.cfg.H * np.log(pfull / self.cfg.p0)
         zhalf = -self.cfg.H * np.log(phalf / self.cfg.p0)

         if len(phalf) != len(pfull) + 1:
            raise ValueError('There must be one more half level (phalf) than full levels (pfull)')

         Nz = len(pfull)
         p_top = phalf[0]
         p_bot = phalf[-1]
         z_top = zhalf[0]
         z_bot = zhalf[-1]
      else:
         raise ValueError(f"Unrecognized grid spacing option: {spacing}")

      if p_top > p_bot:
         raise ValueError("Pressure levels must be increasing (from top of the atmosphere down).")

      # Grid parameters
      self.__dict__['Nz'] = Nz
      self.__dict__['p_top'] = p_top
      self.__dict__['p_bot'] = p_bot
      self.__dict__['z_top'] = z_top
      self.__dict__['z_bot'] = z_bot

      self.initialize_var('zhalf', 'm', self.Nz + 1, zhalf, grid = True)
      self.initialize_var('zfull', 'm', self.Nz, zfull, grid = True)
      self.initialize_var('phalf', 'Pa', self.Nz + 1, phalf, grid = True)
      self.initialize_var('pfull', 'Pa', self.Nz, pfull, grid = True)

      self.initialize_var('vmask', '1', Nz, 1., grid = True)
   # }}}

### Methods related to dynamics/advection
   def initialize_dynamics(self, *, advected_tracers = ['T']):
   # {{{
      self.initialize_var('T', 'K', self.Nz, 300.) # Prognostic, advected

      self.initialize_var('w', 'm s-1', self.Nz + 1, 0.02) # not prognostic
      self.w[0] = 0.
      self.w[self.Nz] = 0.

      self.initialize_var('wp', 'm s-1', self.Nz + 1, 0j) # not prognostic
      self.wp[0] = 0.
      self.wp[self.Nz] = 0.

      # Conversion factor from temperature to potential temperature
      exner = (self.pfull / self.cfg.p0)**(-self.cfg.Rd / self.cfg.cp)

      self.initialize_var('Exner', '1', self.Nz, exner, grid = True)

      self.initialize_scalar('omega', 'd-1', 2 * np.pi / (86400. * 840.))

      # Reference grid for semi-lagrangian advection interpolation
      #self.__dict__['zadv'] = np.concatenate([[z_bot], zfull[::-1], [z_top]])
      self.__dict__['zadv'] = self.zfull[::-1]
   # }}}

   def get_courant(self, dt):
   # {{{ 
      dz = np.min(np.absolute(np.diff(self.zhalf)))
      wmax = np.max(np.absolute(self.w) + np.absolute(self.wp))

      return wmax * dt / dz
   # }}}

   def get_origins(self, state, j_now, j_new, dt, I = 2):
   # {{{
      dth = dt / 2.

      dz = self.zhalf[1:] - self.zhalf[:-1]

      # Destinations
      r_dest = self.zfull

      # Interpolate velocity at future time
      wF = 0.5 * (state.w[j_new, 1:] + state.w[j_new, :-1])
      aF = wF * (state.w[j_new, 1:] - state.w[j_new, :-1]) / dz

      # Future half of trajectory
      c2 = dth * wF - 0.5 * dth**2 * aF

      # First guess at origin points
      r_half = r_dest - c2
      zorg = r_half

      # Iterate estimate of past half of trajectory
      for i in range(I):
         # Interpolate velocity and acceleration to origin points
         aS = 0.5 * (state.w[j_now, 1:] + state.w[j_now, :-1]) * (state.w[j_now, 1:] - state.w[j_now, :-1]) / dz
         wS = np.interp(zorg[::-1], self.zhalf[::-1], state.w[j_now, ::-1])[::-1]
         aS = np.interp(zorg[::-1], self.zfull[::-1], aS[::-1])[::-1]

         # Past half of trajectory
         c1 = dth * wS + 0.5 * dth**2 * aS
         zorg = r_half - c1
      
      return zorg
   # }}}

   def build_advection_matrix(self, z_org):
   # {{{
      ''' Constructs matrices to build weights for advection interpolation. 
       Returns a tuple of three matrices, C, D, and Del; the weights for shape-preserving
       interpolation require a further diagonal matrix T to eliminate overshoot and can
       be calculated as C + D @ T @ Del.'''

      ip = self.zadv.searchsorted(z_org)
      iL = np.where(ip == 0.)[0]
      iR = np.where(ip == len(self.zadv))[0]

      ip[iL] = 1
      ip[iR] = len(self.zadv) - 1

      i0 = ip - 1

      dx = self.zadv[ip] - self.zadv[i0]
      tn = (z_org - self.zadv[i0]) / dx
      tn[iL] = 0.
      tn[iR] = 1.

      otn = 1 - tn

      N = len(z_org)
      M = len(self.zadv)

      Dl = np.zeros(M - 1)
      Dc = np.zeros(M)
      Dr = np.zeros(M - 1)

      Dx = np.diff(self.zadv)

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

      return C, D, Del
   # }}}

   def advect_quantity(self, C, D, Del, X):
   # {{{
      ''' Carry out advection of a given field X, given the components
      of the weights matrix C, D, and Del from build_advection_matrix().
      Calculates shape-preserving modifications for this field, applies
      normalization and boundary conditions. The array X is oriented in increasing
      height.'''

      # Compute shape-preserving modifications
      T = np.ones(self.Nz)

      def fmt(a): return ' '.join([f'{d:5.1f}' for d in a])

      # One-sided estimates at every grid point
      d0 = (X[1:] - X[:-1]) / (self.zadv[1:] - self.zadv[:-1])

      # Initial derivatives for the interpolation splines
      m = np.zeros(self.Nz)
      m[1:-1] = 0.5 * (d0[1:] + d0[:-1])
      m[0] = d0[0]
      m[-1] = d0[-1]

      #print('d: ' + fmt(d0*1e3))
      #print('m: ' + fmt(m*1e3))

      # Find indices of local extrema, and indices of complement.
      # Adjacent indices are tested for overshoot, so last element is omitted in latter.
      ext = (d0[1:] * d0[:-1] <= 0.)
      exti = np.where(ext)[0] + 1
      extn = np.concatenate([[0], np.where(~ext)[0] + 1])

      # Set slope at any extrema to zero
      m[exti] = 0
      T[exti] = 0.

      # Where the quantity tau < 1, we need to rescale the slopes
      #anz = m[extn] / d0[extn]
      #bnz = m[extn + 1] / d0[extn]
      #tau = 3 / np.sqrt(anz**2 + bnz**2)
      #tau = 3 * np.abs(d0[extn]) / np.sqrt(m[extn]**2 + m[extn + 1]**2 + 1e-16)
      tau = 3 * np.abs(d0) / np.sqrt(m[:-1]**2 + m[1:]**2 + 1e-32)
      #print('tau: ' + fmt(tau))

      nmt = np.where(tau < 1)[0]
      #m[extn[nmt]]     *= tau[nmt]
      #m[extn[nmt] + 1] *= tau[nmt]
      #T[extn[nmt]]     *= tau[nmt]
      #T[extn[nmt] + 1] *= tau[nmt]
      T[nmt]     *= tau[nmt]
      T[nmt + 1] *= tau[nmt]

      #print('z  : ' + fmt(self.zadv*1e-4))
      #print('X  : ' + fmt(X))
      #print('m  : ' + fmt(m*1e4))
      #print('T  : ' + fmt(T))

      # Calculate Weights
      T = sparse.diags_array([T], offsets = [0], shape = (self.Nz,self.Nz), format = 'csr')

      #print(D @ Del)
      #print(D @ T @ Del)
      L = (C + D @ T @ Del)
      #L = (C + D @ Del)

      #print('L: ')
      #print(L.tocsc())

      # Normalize weights
      #N = L.sum(0)
      #iN = np.where(N > 0.3)[0]
      #print('N: ' + fmt(N[1:-1]))
      #L[1:-1, :] = L[1:-1, :] / N[1:-1].reshape(-1, 1)

      # Apply interpolation
      return L @ X
   # }}}

   def step_advection(self, state, z_org, j_now, j_new, dt):
   # {{{
      C, D, Del = self.build_advection_matrix(z_org[::-1])

      L = C + D @ Del

      # Convert temperature to potential temperature
      Theta = state.T[j_now, :] * self.Exner

      # Advect potential temperature then convert back to temperature
      state.T[j_new, :] = (L @ Theta[::-1])[::-1] / self.Exner

      for s in self.advected:
         v = state.columns[s]
         v[j_new, :] = self.advect_quantity(C, D, Del, v[j_now, ::-1])[::-1]
   # }}}

### Methods related to radiative transfer
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
            #T, TSfc,  self.scon, 4, \
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

### Methods related to chemistry
   def initialize_chemistry(self, *, mechanism):
   # {{{  
      parser = mc.Parser()

      mechanism_file = self.cfg.config_path + mechanism + '.json'

      self.__dict__['mechanism'] = parser.parse(mechanism_file)
      self.__dict__['MICMsolver'] = musica.MICM(mechanism = self.mechanism, solver_type = musica.SolverType.rosenbrock_standard_order)
      self.__dict__['MICMstate'] = self.MICMsolver.create_state(200)

      species_list = []
      advected_list = []

      for sp in self.mechanism.species:
         properties = {'molecular_weight': sp.molecular_weight_kg_mol}
         properties.update(sp.other_properties)

         name = sp.name
         species_list.append(name)

         advect = properties.pop('__do advect', False)
         if advect == 'true': 
            advect = True
            advected_list.append(name)
         else: 
            advect = False

         if name in self.variables:
            raise ValueError(f'{name} has already been initialized.')
         
         self.variables[name] = SpeciesVariable(name, 'vmr', self.Nz, 0., advect, **properties)

      self.__dict__['species'] = species_list
      self.__dict__['advected'] = advected_list
   # }}}

   def step_chemistry(self, state, z_org, j_new, dt):
   # {{{
      # Update MICM state object with temperatures and pressures
      p_org = self.cfg.p0 * np.exp(-z_org / self.cfg.H)
      self.MICMstate.set_conditions(state.T[j_new, :], p_org)

      # For now update the concentrations manually
      # This will be more efficient if we structure the column
      # state vector to have a compatible memory structure
      st = self.MICMstate._State__states[0].concentration_strides()[0]
      sp = self.MICMstate.get_species_ordering()
      for s, i in sp.items():
         v = musica._musica.VectorDouble(state.columns[s][j_new, :])
         self.MICMstate._State__states[0].concentrations[i::st] = v

      self.MICMsolver.solve(self.MICMstate, dt)

      # Read out resulting concentrations
      for s, i in sp.items():
         state.columns[s][j_new, :] = self.MICMstate._State__states[0].concentrations[i::st]
   # }}}

### Methods related to solver
   def get_internal_state(self, n = 1):
   # {{{
      return State(self.variables, self.scalars, n)
   # }}}

   def create_output_state(self, n = 1):
   # {{{
      return State(self.variables | self.output_variables, self.scalars, n)
   # }}}

   def save_state(self, state, output, j_state, i_out):
   # {{{
      for c in state.columns: 
         output.columns[c][i_out, :] = state.columns[c][j_state, :]

      for s in state.scalars: 
         output.scalars[s][i_out] = state.scalars[s][j_state]
   # }}}

   def update_externals(self, state, j, t):
   # {{{
      state.w[j, :] = self.w + np.real(self.wp * np.exp(1j * self.omega * t))
   # }}}

   def solve(self, nsteps, dt, output_freq = 1):
   # {{{
      # Output grid
      nout   = int(nsteps / output_freq) + 1
      times  = np.arange(nout) * dt * output_freq

      s0 = self.get_internal_state(n = 3)
      o0 = self.create_output_state(nout)

      i = 0
      i_step = 0
      i_out = 0

      i_out += 1

      j_old, j_now = 0, 1
      #j_old, j_now, j_new = 0, 1, 2

      self.update_externals(s0, j_old, 0.)

      #self.compute_radiation(s0, o0, 0, i_out)
      self.save_state(s0, o0, j_old, i_out)

      for i in range(nsteps):
         # Update externally varying parameters
         self.update_externals(s0, j_now, (i + 1) * dt)

         # Compute Lagragian origin points
         z_org = self.get_origins(s0, j_old, j_now, dt)

         # Advect species
         self.step_advection(s0, z_org, j_old, j_now, dt)

         # Run chemistry for the time step
         self.step_chemistry(s0, z_org, j_now, dt)

         # Diabatic tendencies
         self.compute_radiation(s0, o0, j_now, i_out)
         dQ = o0.lw_hr[i_out, :] + o0.sw_hr[i_out, :] + self.dyn_hr[:]
         s0.T[j_now] += dt * dQ / 86400.

         i_step += 1

         if i_step >= output_freq:
            self.save_state(s0, o0, j_now, i_out)
            i_out += 1
            i_step = 0

         #j_old, j_now, j_new = j_now, j_new, j_old
         j_old, j_now = j_now, j_old

      return times, o0
   # }}}

import pygeode as pyg
def to_pyg(col, ts, out, init = None):
# {{{
   time = pyg.Yearless(ts / 86400., units = 'days', startdate = dict(year = 1, day = 0))
   pfull = pyg.Pres(col.pfull, name = 'pfull')
   phalf = pyg.Pres(col.phalf, name = 'phalf')
   zfull = pyg.Height(col.zfull, name = 'zfull')
   zhalf = pyg.Height(col.zhalf, name = 'zhalf')

   def add_var(name, values, unit):
      if values.shape[1] == col.Nz:
         axs = (time, zfull,)
      elif values.shape[1] == col.Nz + 1:
         axs = (time, zhalf,)
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

