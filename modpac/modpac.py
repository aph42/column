import os
import json
import datetime

import numpy as np

from rrtm import rrtmg

import musica
import musica.mechanism_configuration as mc
import musica.tuvx.vTS1

class ModPAC():
   '''  Modular Photochemistry in an Atmospheric Column (ModPAC)
        A column model of the atmosphere, including radiative transfer, photochemistry using
        the MUSICA and TUVx components from NCAR, and vertical advection.'''
   def __init__(self, configuration, name_template = "{config_file}", output_path_template = "{name}/{rundate}"):
   # {{{
      self.__dict__['grid'] = {}

      self.__dict__['columns'] = {}
      self.__dict__['scalars'] = {}

      self.__dict__['output_columns'] = {}

      self.__dict__['cfg'] = configuration

      # Used to generate filenames for output/restart files
      self.__dict__['name_template'] = name_template
      self.__dict__['output_path_template'] = output_path_template
      self.__dict__['output_file_template'] = '{outpath}/{runname}_{i0:05d}.nc'
      self.__dict__['restart_file_template'] =  '{outpath}/{runname}_restart_{i1:010d}.nc'

      # Initialize grid
      self.initialize_grid(**self.cfg.grid)

      # Initialize chemistry
      self.initialize_chemistry(**self.cfg.chemistry)

      # Initialize dynamical quantities
      self.initialize_dynamics(**self.cfg.dynamics)

      # Initialize radiation
      self.initialize_radiation(**self.cfg.radiation)

      # Initialize photolysis 
      self.initialize_photolysis(**self.cfg.photolysis)

      # Initialize convection
      self.initialize_convection(**self.cfg.convection)

      # Initialize humidity
      self.initialize_humidity(**self.cfg.humidity)

    # }}} 

   def __setattr__(self, name, value):
   # {{{
      if name in self.grid:
         self.grid[name].values[:] = value

      elif name in self.columns:
         self.columns[name].values[:] = value

      elif name in self.scalars:
         self.scalars[name].value = value

      elif name in self.output_columns:
         self.output_columns[name].values[:] = value

      elif name in self.__dict__.keys():
         raise ValueError(f'{name} is read-only.')

      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def __getattr__(self, name):
   # {{{
      if name in self.grid:
         return self.grid[name]
      elif name in self.columns:
         return self.columns[name]
      elif name in self.scalars:
         return self.scalars[name].value
      elif name in self.output_columns:
         return self.output_columns[name]
      elif name in self.__dict__.keys():
         return self.__dict__[name]
      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def __getitem__(self, name):
   # {{{
      if name in self.grid:
         return self.grid[name]
      elif name in self.columns:
         return self.columns[name]
      elif name in self.scalars:
         return self.scalars[name].value
      else:
         raise ValueError(f'{self} has no variable {name}. You may be trying to unpack a ModPAC instance as a tuple.')
   # }}}

   def initialize_var(self, name, unit, Nz, initial_value, grid = False, attributes = {}):
   # {{{
      from . import state
      if grid:
         if name in self.grid:
            raise ValueError(f'{name} has already been initialized.')
         
         self.grid[name] = state.ColumnVariable(name, unit, Nz, initial_value, attributes = attributes)
      else:
         if name in self.columns:
            raise ValueError(f'{name} has already been initialized.')
         
         self.columns[name] = state.ColumnVariable(name, unit, Nz, initial_value, attributes = attributes)
   # }}}

   def add_output(self, name, unit, Nz):
   # {{{
      from . import state
      if name in self.output_columns:
         raise ValueError(f'{name} has already been defined as an output variable.')
      
      self.output_columns[name] = state.ColumnVariable(name, unit, Nz, 0., output = True)
   # }}}

   def initialize_scalar(self, name, unit, initial_value, attributes = {}):
   # {{{
      from . import state

      if name in self.scalars:
         raise ValueError(f'{name} has already been initialized.')
      
      self.scalars[name] = state.ScalarVariable(name, unit, initial_value, attributes = attributes)
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
      
      attr_dict = dict(H = self.cfg.H, p0 = self.cfg.p0)

      self.initialize_var('zhalf', 'm',  self.Nz + 1,  zhalf, grid = True, attributes = dict(long_name = 'Log-pressure height of level interfaces', **attr_dict))
      self.initialize_var('zfull', 'm',  self.Nz,      zfull, grid = True, attributes = dict(long_name = 'Log-pressure height of level midpoints',  **attr_dict))

      # DEBUG: pfull inherits units of cfg.p0 from json file, default hPa
      self.initialize_var('phalf', 'hPa', self.Nz + 1,  phalf, grid = True, attributes = dict(long_name = 'Pressure of level midpoints')) 
      self.initialize_var('pfull', 'hPa', self.Nz,      pfull, grid = True, attributes = dict(long_name = 'Pressure of level midpoints'))

      #self.initialize_var('vmask', '1', Nz, 1., grid = True, attributes = dict(long_name = 'Mask for fixing a component of the column (1 = fully prognostic, 0 = fixed).')
   # }}}

### Methods related to dynamics/advection
   def initialize_dynamics(self, *, active = True, kappa_zz = 0.):
   # {{{
      from . import dynamics
      self.__dict__['do_dynamics'] = active

      self.initialize_var('T',  'K',     self.Nz,     300., attributes = dict(long_name = 'Temperature', standard_name = 'air_temperature'))

      self.initialize_var('w',  'm s-1', self.Nz + 1, 0.02, attributes = dict(long_name = 'Log-pressure vertical velocity', standard_name = 'upward_air_velocity'))
      self.w[0] = 0.
      self.w[self.Nz] = 0.

      self.initialize_var('wp', 'm s-1', self.Nz + 1, 0j,   attributes = dict(long_name = 'Anomalous log-pressure vertical velocity')) # not prognostic
      self.wp[0] = 0.
      self.wp[self.Nz] = 0.

      # Conversion factor from temperature to potential temperature
      exner = (self.pfull / self.cfg.p0)**(-self.cfg.Rd / self.cfg.cp)

      self.initialize_var('Exner', '1', self.Nz, exner, grid = True)

      self.initialize_scalar('omega', 'd-1', 2 * np.pi / (86400. * 840.))

      self.__dict__['kappa_zz'] = kappa_zz
      self.__dict__['L_diffusion'] = dynamics.make_diffusion_operator(self.zfull)

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

   def step_advection(self, state, z_org, j_old, j_now, dt):
   # {{{
      from . import dynamics

      C, D, Del = dynamics.build_advection_matrix(self.zadv, z_org[::-1])

      # Construct interpolation matrix for potential temperature 
      # (no shape-preserving adjustments are made, though we could think about that)
      L = C + D @ Del

      # Convert temperature to potential temperature
      Theta = state.T[j_old, :] * self.Exner

      # Advect potential temperature then convert back to temperature
      state.T[j_now, :] = (L @ Theta[::-1])[::-1] / self.Exner

      for s in self.species:
         sp = self.columns[s]
         if sp.advect and not sp.fixed:
            v = state.column_values[s]
            v[j_now, :] = dynamics.advect_quantity(self.zadv, C, D, Del, v[j_old, ::-1])[::-1]
   # }}}

   def step_diffusion(self, state, j_now, dt):
   # {{{
      Ld = self.kappa_zz * dt * self.L_diffusion

      state.T[j_now, :] += Ld @ state.T[j_now, :]

      for s in self.species:
         sp = self.columns[s]
         if sp.advect and not sp.fixed:
            v = state.column_values[s]
            v[j_now, :] += Ld @ v[j_now, :]
   # }}}

### Methods related to radiative transfer
   def initialize_radiation(self, *, active = True, scon = 1368.22, zenith = 'fixed_specified', **kwargs):
   # {{{
      from . import astr
      self.__dict__['do_radiation'] = active

      # Astronomical settings
      self.__dict__['scon'] = scon
      self.__dict__['zenith'] = zenith

      self.initialize_scalar('Tsfc',               'K',   300., attributes = dict(long_name = 'Surface temperture', standard_name = 'surface_temperature'))
      self.initialize_scalar('emissivity',         '1',   0.99, attributes = dict(long_name = 'Surface emissivity'))
      self.initialize_scalar('albedo',             '1',   0.3 , attributes = dict(long_name = 'Surface albedo', standard_name = 'surface_albedo'))
      self.initialize_scalar('solar_zenith_angle', 'deg', 0.  , attributes = dict(long_name = 'Solar zenith angle', standard_name = 'solar_zenith_angle'))

      # Initialize zenith angle
      if zenith == 'fixed_specified':
         # Zenith angle fixed and explicitly set
         self.solar_zenith_angle = kwargs.get('solar_zenith_angle', 0.)

         self.__dict__['initial_date'] = kwargs.get('initial_date', '2000-01-01')

      elif zenith == 'fixed_computed':
         # Zenith angle fixed, computed from latitude and initial date and (local) time

         self.__dict__['initial_date'] = kwargs.get('initial_date', '2000-01-01')
         self.__dict__['local_hour'] = kwargs.get('local_hour', 12.)
         self.__dict__['latitude'] = kwargs.get('latitude', 0.)

         n = astr.date_to_n(self.initial_date)
         declination = astr.declination(n)

         self.solar_zenith_angle = astr.zenith_from_declination(self.latitude, declination, local_hour)

      elif zenith == 'daily_mean_computed':
         # Zenith angle fixed, computed from latitude and initial date and (local) time

         self.__dict__['initial_date'] = kwargs.get('initial_date', '2000-01-01')
         self.__dict__['num_radiation_calls'] = kwargs.get('num_radiation_calls', 3)
         self.__dict__['latitude'] = kwargs.get('latitude', 0.)

         n = astr.date_to_n(self.initial_date)
         declination = astr.declination(n)

         angles, weights = astr.get_quadrature_angles_and_weights_gauss(self.latitude, declination, self.num_radiation_calls)
         self.__dict__['quadrature_angles'] = angles
         self.__dict__['quadrature_weights'] = weights

         self.solar_zenith_angle = astr.zenith_dailymean_equivalent(self.latitude, n)

      elif zenith == 'diurnal_cycle':
         # Zenith angle goes through fixed diurnal cycle, appropriate to given latitude and initial date
         self.__dict__['initial_date'] = kwargs.get('initial_date', '2000-01-01')
         self.__dict__['latitude'] = kwargs.get('latitude', 0.)

         n = astr.date_to_n(self.initial_date)
         declination = astr.declination(n)
         self.__dict__['declination'] = declination
      else:
         raise ValueError(f"Zenith option '{zenith}' unrecognized.")

      # Set up output options from radiation
      self.add_output('lw_uflx', 'W m-2', self.Nz + 1) 
      self.add_output('lw_dflx', 'W m-2', self.Nz + 1) 
      self.add_output('sw_uflx', 'W m-2', self.Nz + 1) 
      self.add_output('sw_dflx', 'W m-2', self.Nz + 1) 

      self.add_output('lw_hr', 'K d-1', self.Nz) 
      self.add_output('sw_hr', 'K d-1', self.Nz) 

      # Initialize rrtmg
      rrtmg.init(self.cfg.cp)

      self.initialize_var('dyn_hr', 'K d-1', self.Nz, 0.)
   # }}}

   def compute_radiation(self, state, output, j_now, i_out):
   # {{{
      # Helper function to reshape grid arrays
      def _g(v): return np.asfortranarray(v[:].reshape(1, -1).copy(), 'd')

      # Helper function to reshape column arrays
      def _c(v): return np.asfortranarray(v[j_now, :].reshape(1, -1).copy(), 'd')

      # Helper function to reshape scalar quantities
      def _s(v): return np.asfortranarray(np.array(v[j_now:j_now + 1]), 'd')

      pfull = _g(self.pfull)
      phalf = _g(self.phalf)

      T   = _c(state.T)
      CO2 = _c(state.CO2)
      O3  = _c(state.O3 )
      H2O = _c(state.H2O)

      TSfc = _s(state.Tsfc)
      Emis = _s(state.emissivity)
      alb  = _s(state.albedo)

      lw = rrtmg.rrtmg_lw(pfull, phalf, \
                          T, TSfc, Emis, \
                          CO2,  H2O,  O3)

      output.lw_uflx[i_out, :] = lw['uflxlw'][0, :]
      output.lw_dflx[i_out, :] = lw['dflxlw'][0, :]
      output.lw_hr[i_out, :]   = lw['lwhr'][0, :]

      if self.zenith in ['daily_mean_computed']:
         # Do multiple calls to SW to calculate daily mean value
         for n in range(self.num_radiation_calls):
            cosz = np.cos(np.deg2rad(self.quadrature_angles[n]))
            cosz = np.asfortranarray(cosz)

            sw = rrtmg.rrtmg_sw(pfull, phalf, \
                  T, TSfc,  self.scon, \
                  cosz, alb, \
                  CO2, H2O, O3)

            w = self.quadrature_weights[n]

            output.sw_uflx[i_out, :] += w * sw['uflxsw'][0, :]
            output.sw_dflx[i_out, :] += w * sw['dflxsw'][0, :]
            output.sw_hr[i_out, :]   += w * sw['swhr'][0, :]

      else:
         #cosz = np.cos(np.deg2rad(np.min([90., state.solar_zenith_angle[j_now]])))
         cosz = np.cos(np.deg2rad(state.solar_zenith_angle[j_now]))
         cosz = np.asfortranarray(cosz)

         sw = rrtmg.rrtmg_sw(pfull, phalf, \
               T, TSfc,  self.scon, \
               cosz, alb, \
               CO2, H2O, O3)

         output.sw_uflx[i_out, :] = sw['uflxsw'][0, :]
         output.sw_dflx[i_out, :] = sw['dflxsw'][0, :]
         output.sw_hr[i_out, :]   = sw['swhr'][0, :]
   # }}}

   def set_zenith_angle(self, state, j_now, t):
   # {{{
      from . import astr
      if self.zenith == 'diurnal_cycle':
         local_hour = np.mod(t / 3600., 24.)
         state.solar_zenith_angle[j_now] = astr.zenith_from_declination(self.latitude, self.declination, local_hour)
   # }}}

### Methods related to chemistry
   def initialize_chemistry(self, *, mechanism = '', active = True, **kwargs):
   # {{{  
      from . import state

      self.__dict__['do_chemistry'] = active

      # Regardless of whether chemistry is active, read in the mechanism
      # to initialize species
      parser = mc.Parser()
      mechanism_file = self.cfg.config_root + f'/mechanisms/{mechanism}.json'
      self.__dict__['mechanism'] = parser.parse(mechanism_file)

      species_list = []

      for sp in self.mechanism.species:
         if sp.molecular_weight_kg_mol is None:
            molecular_weight = 0
         else:
            molecular_weight = sp.molecular_weight_kg_mol

         attributes = {'molecular_weight': molecular_weight}
         attributes.update(sp.other_properties)

         name = sp.name
         species_list.append(name)

         advect = attributes.pop('__do advect', False)
         if advect == 'true': 
            advect = True
         else: 
            advect = False

         convect = attributes.pop('__do convect', False)
         if convect == 'true': 
            convect = True
         else: 
            convect = False

         fixed = attributes.pop('__tracer type', False)
         if fixed == 'CONSTANT': fixed = True

         if name in self.columns:
            raise ValueError(f'{name} has already been initialized.')
         
         self.columns[name] = state.SpeciesVariable(name, 'vmr', self.Nz, 0., advect, fixed, convect, attributes = attributes)

      self.__dict__['species'] = species_list

      if active:
         # We only need the solver if chemistry is active
         self.__dict__['MICMsolver'] = musica.MICM(mechanism = self.mechanism, solver_type = musica.SolverType.rosenbrock_standard_order)
         self.__dict__['MICMstate'] = self.MICMsolver.create_state(self.Nz)
   # }}}

   def step_chemistry(self, state, z_org, j_now, dt):
   # {{{
      # Update MICM state object with temperatures and pressures
      p_org = self.cfg.p0 * np.exp(-z_org / self.cfg.H)
      self.MICMstate.set_conditions(state.T[j_now, :], 100.*p_org)

      nafull = p_org * 100 / (self.cfg.R * state.T[j_now, :])

      # For now update the concentrations manually

      # This will be more efficient if we structure the column
      # state vector to have a compatible memory structure

      mstate = self.MICMstate.get_internal_state()
      stride = mstate.concentration_strides()[0]
      sp = self.MICMstate.get_species_ordering()
      for s, i in sp.items():
         # convert from vmr to mol m-3
         v = musica._musica.VectorDouble(state.column_values[s][j_now, :] * nafull)
         mstate.concentrations[i::stride] = v
         
      self.MICMsolver.solve(self.MICMstate, dt)

      # Read out resulting concentrations
      for s, i in sp.items():
         # convert back from mol m-3 to vmr
         if not self.columns[s].fixed:
            state.column_values[s][j_now, :] = mstate.concentrations[i::stride] / nafull
   # }}}

   def initialize_photolysis(self, *, mechanism = '', mapping = {}, active = True, parameterize_jNO = False):
# {{{
      # tuv-x height coordinates are bottom up
      self.__dict__['do_photolysis'] = active
      self.__dict__['parameterize_jNO'] = parameterize_jNO

      if not active: 
         # Nothing to initialize
         return

      self.__dict__['micm_to_tuvx'] = mapping.copy()
      #{'jO2':'jo2_b','jO3->O':'jo3_b','jO3->O1D':'jo3_a'}

      for key in self.micm_to_tuvx:
          self.add_output(self.micm_to_tuvx[key], 's-1', self.Nz) 

      if self.parameterize_jNO:
          self.add_output('jno', 's-1', self.Nz)
       
      # initialize photolysis 
      tuvx_config_path = os.path.dirname(musica.utils.find_config_path())
      tuvx_config_path = f'{tuvx_config_path}/configs/tuvx'
      self.__dict__['tuvx_mechanism_file'] = f'{tuvx_config_path}/{mechanism}.json'


      # Read in mechanism file
      with open(self.tuvx_mechanism_file) as f:
         config = json.load(f)

      # Remove photolysis reactions that are not specified by mechanism
      #rct = [r for r in config['photolysis']['reactions'] if r['name'] in self.micm_to_tuvx.values()]

      #config['photolysis']['reactions'] = rct

      # Set up grids
      grids = musica.tuvx.GridMap()
        
      heights = musica.tuvx.grid.Grid(name="height", units="km", num_sections=self.Nz)
      heights.edges = self.zhalf[::-1]/1000. 
      heights.midpoints = self.zfull[::-1]/1000.
      
      grids["height", "km"] = heights
      grids["wavelength", "nm"] = musica.tuvx.vTS1.wavelength_grid()
    
      # Set up profiles
      profiles = musica.tuvx.ProfileMap()
      profiles["air", "molecule cm-3"] = musica.tuvx.vTS1.profile("air", grids["height", "km"])
      profiles["O3", "molecule cm-3"] = musica.tuvx.vTS1.profile("O3", grids["height", "km"])
      profiles["O2", "molecule cm-3"] = musica.tuvx.vTS1.profile("O2", grids["height", "km"])
      profiles["temperature", "K"] = musica.tuvx.vTS1.profile("temperature", grids["height", "km"])
      profiles["surface albedo", "none"] = musica.tuvx.vTS1.profile("surface albedo", grids["wavelength", "nm"])
      profiles["extraterrestrial flux", "photon cm-2 s-1"] = musica.tuvx.vTS1.profile(
            "extraterrestrial flux", grids["wavelength", "nm"]
        )
        
      # Set up radiators
      radiators = musica.tuvx.RadiatorMap() # Note: radiators automatically includes air, O2, and O3 without being specified
      radiators["aerosol"] = musica.tuvx.vTS1.radiator("aerosol", grids["height", "km"], grids["wavelength", "nm"])
      
      # Change path to tuvx config so that relative paths in configuration file work (ugh)
      original_cwd = os.getcwd()
      os.chdir(tuvx_config_path)

      # Create TUV-x instance with v5.4 configuration file
      try:
         self.__dict__['tuvx'] = musica.tuvx.TUVX(
               grid_map     = grids,
               profile_map  = profiles,
               radiator_map = radiators,
               config_string  = json.dumps(config)
           )
      finally:
          os.chdir(original_cwd)
# }}}
        
   def compute_photolysis(self, state, output, z_org, j_new, i_out):
# {{{
      from . import photochem
      # update ozone and temperature, then calculate photolysis rates using TUV-x
      # TUV-x height coordinates are bottom-up  

      def full_to_half(v):    return np.interp(self.zhalf, self.zfull, v)
      def tuvx_to_full(jval): return jval.interp(vertical_edge = z_org/1000.).values
    
      # get the vertical profiles
      grids = self.tuvx.get_grid_map()
      profiles = self.tuvx.get_profile_map()
      
      # update the temperature profile
      T_profile = profiles["temperature", "K"]
      T_profile.midpoint_values =  state.T[j_new,::-1] 
      T_profile.edge_values = full_to_half(state.T[j_new,:])[::-1] 
      
      # convert from vmr to molecules cm-3
      n_air = self.pfull * 100 / (self.cfg.R * state.T[j_new,:])
      o3_mid = state.O3[j_new,:] * n_air * self.cfg.Av * 1e-6

      # update the ozone profile
      o3_profile = profiles["O3", "molecule cm-3"]
      o3_profile.midpoint_values = o3_mid[::-1]
      o3_profile.edge_values = full_to_half(o3_mid)[::-1]
      #o3_profile.edge_values = full_to_half(state.O3[j_new,:])[::-1] * self.nahalf[::-1] * self.cfg.Av * 1e-6 # molec cm-3
      o3_profile.calculate_layer_densities(grids["height", "km"]) # provide the height grid for layer thicknesses

      # calculate photolysis rates
      if self.zenith in ['daily_mean_computed']:
         # Do multiple calls to photolysis to calculate daily mean value
         for n in range(self.num_radiation_calls):
            sza = np.deg2rad(self.quadrature_angles[n])

            w = self.quadrature_weights[n]

            tuvx_output = self.tuvx.run(sza = sza, \
                                        earth_sun_distance = 1.0)
            tuvx_rates = tuvx_output['photolysis_rate_constants'] 

            # Save photolysis rates for output
            for micm_reaction in self.micm_to_tuvx.keys():
               tuvx_key = self.micm_to_tuvx[micm_reaction]
               jval = tuvx_to_full(tuvx_rates.sel(reaction = tuvx_key))
               output.column_values[tuvx_key][i_out, :] += w * jval

            if self.parameterize_jNO:
               jval = photochem.calc_jNO(self, state, sza, j_new)
               output.column_values['jno'][i_out, :] += w * jval
      else:
         #sza = np.deg2rad(np.min([90, state.solar_zenith_angle[j_new]]))
         sza = np.deg2rad(state.solar_zenith_angle[j_new])

         tuvx_output = self.tuvx.run(sza = sza, \
                                     earth_sun_distance = 1.0)
         tuvx_rates = tuvx_output['photolysis_rate_constants'] 
          
         # Save photolysis rates for output
         for micm_reaction in self.micm_to_tuvx.keys():
            tuvx_key = self.micm_to_tuvx[micm_reaction]
            jval = tuvx_to_full(tuvx_rates.sel(reaction = tuvx_key))
            output.column_values[tuvx_key][i_out, :] = jval

         if self.parameterize_jNO:
            jval = photochem.calc_jNO(self, state, sza, j_new)
            output.column_values['jno'][i_out, :] = jval

      # Update rates in MICM
      jvals = {}

      for micm_reaction in self.micm_to_tuvx.keys():
         tuvx_key = self.micm_to_tuvx[micm_reaction]
         micm_key = f'PHOTO.{micm_reaction}'
         jvals[micm_key] = output.column_values[tuvx_key][i_out, :]

      if self.parameterize_jNO:
         micm_key = f'PHOTO.jNO->N'
         jvals[micm_key] = output.column_values['jno'][i_out, :]

      self.MICMstate.set_user_defined_rate_parameters(jvals)
# }}}
    
### Methods related to convection/convective adjustment
   def initialize_convection(self, *, active = True):
   # {{{
      self.__dict__['do_convection'] = active

      self.initialize_var('T_conv', 'K', self.Nz, 0.) # Moist adiabatic temperature profile from Tsfc
      self.initialize_scalar('z_conv', 'm', 0.) # top of convective adjustment
   # }}}

   def convective_adjustment(self, state, j_now):
   # {{{
       # convective adjustment
       # after Thuburn and Craig (2002) in which T_conv sets the minimum temperature
       # T_conv is calculated as a moist adiabat

       T_deficit = state.T[j_now,:] - self.T_conv
       # T_deficit = np.mean(self.column_values['T'][i_out-nday:i_out+1,:],axis=0) - self.T_conv
       
       idx = np.where(T_deficit < 0)[0]
       state.z_conv[j_now] = self.zfull[idx[0]] if idx.size > 0 else np.nan
               
       state.T[j_now,:] = np.maximum(state.T[j_now,:], self.T_conv)

       # Convectively adjust flagged species
       for s in self.species:
         sp = self.columns[s]
         if sp.convect and not sp.fixed:
            v = state.column_values[s]
            v[j_now, :] = np.where(self.zfull <= state.z_conv[j_now], sp[:], v[j_now,:])
   # }}}

### Methods related to humidity (remove supersaturation, tropospheric RH)    
   def initialize_humidity(self, *, active = True):
   # {{{
      self.__dict__['do_humidity'] = active
 
      self.initialize_var('H2O_conv', 'vmr', self.Nz, 0.) # convective water vapor profile
    # }}}

   def remove_supersat(self,state,j_now):
   # {{{
       # This function does 2 things (both of which depend on saturation_vmr):
       # 1) remove water vapor in excess of supersaturation
       # 2) enforce the specified relative humidity profile from self.RH_troposphere

       saturation_vmr = self.calc_saturation_vmr(state.T[j_now,:],self.pfull)
 
       # remove water vapor in excess of supersaturation
       state.H2O[j_now, :] = np.minimum(state.H2O[j_now, :], saturation_vmr)

   #     # relax tropospheric humidity
   #     state.H2O[j_now,:] = np.where(self.zfull<=state.z_conv[j_now],self.H2O_conv,state.H2O[j_now,:])
   # # }}}

   def calc_saturation_vmr(self,T,p):
   # {{{
       Tc = T - self.cfg.T0Cel

       # calculate the saturation volume mixing ratio of water vapor over l
       #e_s = self.cfg.es_0 * np.exp(17.625 * Tc/(Tc + 243.04)) # hPa
       e_sv = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))

       e_si = 0.01 * np.exp(43.494 - (6545.8 / (Tc + 278.))) / (Tc + 868)**2

       e_s = np.minimum(e_sv, e_si)

       saturation_vmr = e_s / p # vmr (units must align between e_s and pfull [e.g., hPa])
       
       return saturation_vmr
   # }}}

### Methods related to solver
   def create_internal_state(self, n = 1):
   # {{{
      from . import state
      return state.State(self.columns, self.scalars, n)
   # }}}

   def create_output_state(self, times, initial_date = None):
   # {{{
      from . import state
      if initial_date is None:
         initial_date = self.initial_date

      return state.OutputState(self.columns | self.output_columns, self.scalars, times, initial_date = initial_date)
   # }}}

   def save_state(self, state, output, j_state, i_out):
   # {{{
      for c in state.columns: 
         output.column_values[c][i_out, :] = state.column_values[c][j_state, :]

      for s in state.scalars: 
         output.scalar_values[s][i_out] = state.scalar_values[s][j_state]
   # }}}

   def build_output_path(self):
   # {{{
      dtnow = datetime.datetime.now()
      params = dict(config_file = self.cfg.config_basename, 
                    rundate     = dtnow.strftime("%Y-%m-%d"),
                    runtime     = dtnow.strftime("%H:%M:%S"))

      name = self.name_template.format(**params)

      params['name'] = name

      path = self.output_path_template.format(**params)

      return path, name
   # }}}

   def find_latest_restart(self, outpath, runname):
   # {{{
      import glob

      restart_template = f'{outpath}/{runname}_restart*.nc'

      restarts = glob.glob(restart_template)
      
      if restarts is None or len(restarts) == 0:
         print("No restarts found.")
         return None

      restarts.sort()

      return restarts[-1]
   # }}}

   def update_externals(self, state, j_now, t):
   # {{{
      # Update periodic component of upwelling
      state.w[j_now, :] = self.w + np.real(self.wp * np.exp(1j * self.omega * t))

      # Update zenith angle
      self.set_zenith_angle(state, j_now, t)
   # }}}

   def solve(self, nsteps, dt, output_freq = 1, write_output = False, restart = None):
   # {{{
      from . import dynamics

      outpath, runname = self.build_output_path()

      print(f"Running integration '{runname}' for {nsteps} timesteps.", flush = True)

      dt_start = datetime.datetime.now()

      # Create internal state vector
      s0 = self.create_internal_state(n = 2)

      # Output grid
      if nsteps % output_freq != 0:
         raise ValueError(f"The number of steps ({nsteps}) must be a multiple of the output frequency ({output_freq}).")

      # Initialization
      if restart == 'latest':
         restart = self.find_latest_restart(outpath, runname)

      if restart is None:
         print(f"Starting new integration.")

         # Starting a new integration
         j_old, j_now = 0, 1

         i0 = 0
         t0 = 0

         # For an initial run, include the initial timestep in the output
         nout   = nsteps // output_freq + 1
         times  = t0 + np.arange(nout) * dt * output_freq
      else:
         # Read restart file
         attrs = s0.from_netcdf(restart)

         j_old = attrs.get('j_old', 0)
         j_now = attrs.get('j_now', 1)

         t0 = attrs.get('t0', 0)
         i0 = attrs.get('i1', 0) 

         print(f"Starting from restart file {restart}, timestep {i0}.")

         # For a restart, do not include the initial timestep in the output
         nout   = nsteps // output_freq
         times  = t0 + (1 + np.arange(nout)) * dt * output_freq

      o0 = self.create_output_state(times)

      i_step = 0
      i_out = 0

      if restart is None:
         # Calculate relevant rates for initial conditions
         # (only used for outputting initial state in new runs)
         self.update_externals(s0, j_old, i0 * dt)

         if self.do_photolysis:
            self.compute_photolysis(s0, o0, self.zfull, j_old, i_out)

         if self.do_radiation:
            self.compute_radiation(s0, o0, j_old, i_out)

         self.save_state(s0, o0, j_old, i_out)

         i_out += 1

      for i in range(i0, i0 + nsteps):
         if i % 500 == 0: print(f"Step {i:>5d}, day {(i * dt) / 86400:>8.1f}.", flush = True)

         # Update externally varying parameters
         self.update_externals(s0, j_now, (i + 1) * dt)

         # Compute Lagragian origin points
         z_org = self.get_origins(s0, j_old, j_now, dt)

         # Advect species
         self.step_advection(s0, z_org, j_old, j_now, dt)

         if self.do_photolysis:
            # Diagnose photolysis rates
            self.compute_photolysis(s0, o0, z_org, j_now, i_out)

         # Run chemistry for the time step
         if self.do_chemistry:
            self.step_chemistry(s0, z_org, j_now, dt)
         
         # Diabatic tendencies
         if self.do_radiation:
            self.compute_radiation(s0, o0, j_now, i_out)
            dQ = o0.lw_hr[i_out, :] + o0.sw_hr[i_out, :] + self.dyn_hr[:]
            s0.T[j_now] += dt * dQ / 86400.

         # Add diffusion 
         if self.kappa_zz > 0.:
            self.step_diffusion(s0, j_now, dt)

         # Convectively adjust temperature and flagged species
         if self.do_convection:
             self.convective_adjustment(s0, j_now)

         if self.do_humidity:
             self.remove_supersat(s0, j_now)
          
         i_step += 1

         if i_step >= output_freq:
            self.save_state(s0, o0, j_now, i_out)
            i_out += 1
            i_step = 0

         # Test for instabilities
         if np.max(s0.T[j_now]) > 1000.:
            raise ValueError(f'Temperatures exceeding 1000K produced (step {i}, day {((i + 1) * dt)/86400.:.2f}); instability developing?')

         j_old, j_now = j_now, j_old

      dt_end = datetime.datetime.now()

      if write_output:
         if not os.path.exists(outpath):
            print(f'Output path {outpath} does not exist. Creating directories.')
            os.makedirs(outpath)

         from modpac import __version__

         run_attrs = dict(description = f'Output produced by ModPac version {__version__}',
                          start_wallclock = dt_start.isoformat(),
                          end_wallclock = dt_end.isoformat())

         # Write restart
         i = i + 1
         rs_attrs = dict(t0 = i*dt, 
                         i0 = i0, 
                         i1 = i, 
                         j_old = j_old, 
                         j_now = j_now)

         rs_attrs.update(run_attrs)

         rfn = self.restart_file_template.format(outpath = outpath, runname = runname, **rs_attrs)
         print(f'Writing restart file {rfn}.')
         s0.to_netcdf(self, rfn, attributes = rs_attrs, precision = 'exact')

         # Write output file
         ofn = self.output_file_template.format(outpath = outpath, runname = runname, **rs_attrs)
         print(f'Writing output to {ofn}.')
         o0.to_netcdf(self, ofn, attributes = run_attrs)

      return o0
   # }}}

