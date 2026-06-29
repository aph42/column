import numpy as np
import netCDF4

# Take a function that operates on numpy arrays, create a
# function that operates on ColumnVariables
def wrap_npfunc (npfunc, doc=''):
# {{{
   #import functools
   #@functools.wraps(npfunc)
   def f(self, *args, **kwargs):
      return npfunc(self.values[:], *args, **kwargs)

   f.__name__ = npfunc.__name__
   f.__doc__ = npfunc.__doc__
   return f
# }}}

def guess_ntype(dtype):
# {{{
   if dtype in [float, np.double, np.float16, np.float32, np.float64, np.float128]:
      return 'float' 
   elif dtype in [complex, np.complex64, np.complex128, np.complex256]:
      return 'complex'
   elif dtype in [int]:
      return 'int'
   elif dtype in [bool, np.bool]:
      return 'bool'
   else:
      return 'float'
# }}}

def ntype_to_NCtype(ntype, precision):
# {{{
   if ntype in ['float', 'complex']:
      if precision == 'exact': return 'f8'
      else: return 'f4'
   elif ntype == 'int':
      return 'i4'
   elif ntype == 'bool':
      return 'i2'
# }}}

class Variable():
# {{{ Variable class definition
   def __init__(self, name, units, dtype, attributes = {}):
   # {{{
      self.name = name
      self.units = units
      self.dtype = dtype
      self.ntype = guess_ntype(dtype)

      self.attributes = attributes.copy()
   # }}}

   def create_NC_var(self, ds, dims, values, precision = 'output'):
   # {{{
      # Create NetCDF variable
      nctype = ntype_to_NCtype(self.ntype, precision)

      if self.ntype == 'complex':
         var   = ds.createVariable(f'{self.name}_r', nctype, dims)
         var_i = ds.createVariable(f'{self.name}_i', nctype, dims)

         var[...]   = np.real(values)
         var_i[...] = np.imag(values)

         var.  setncattr('_note', f'Real component of {self.name}')
         var_i.setncattr('_note', f'Imaginary component of {self.name}')
      else:
         var = ds.createVariable(self.name, nctype, dims)
         var[...] = values

      # Add attributes to NetCDF variable
      var.units = self.units
      for name, value in self.attributes.items():
         var.setncattr(name, value)

      return var
   # }}}
# }}}

class ScalarVariable(Variable):
# {{{ ScalarVariable class definition
   def __init__(self, name, units, initial_value, attributes = {}):
   # {{{
      self.value = initial_value

      dtype = type(initial_value)

      Variable.__init__(self, name, units, dtype, attributes)
   # }}}
# }}}

class ColumnVariable(Variable):
# {{{ ColumnVariable class definition
   def __init__(self, name, units, Nz, initial_value, output = False, attributes = {}):
   # {{{
      if type(initial_value) == np.ndarray:
         dtype = initial_value.dtype
      else:
         dtype = type(initial_value)

      Variable.__init__(self, name, units, dtype, attributes)

      self.Nz = Nz

      self.values = np.ones(Nz, self.dtype)
      self.values[:] = initial_value
      self.output = output
   # }}}

   def __getitem__(self, slice):  return self.values[slice]
   def __setitem__(self, key, value):  self.values[key] = value
   def __len__(self): return self.Nz

ColumnVariable.__add__      = wrap_npfunc(np.ndarray.__add__    )
ColumnVariable.__truediv__  = wrap_npfunc(np.ndarray.__truediv__)
ColumnVariable.__add__      = wrap_npfunc(np.ndarray.__add__)
ColumnVariable.__radd__     = wrap_npfunc(np.ndarray.__radd__)
ColumnVariable.__sub__      = wrap_npfunc(np.ndarray.__sub__)
ColumnVariable.__rsub__     = wrap_npfunc(np.ndarray.__rsub__)
ColumnVariable.__mul__      = wrap_npfunc(np.ndarray.__mul__)
ColumnVariable.__rmul__     = wrap_npfunc(np.ndarray.__rmul__)
ColumnVariable.__truediv__  = wrap_npfunc(np.ndarray.__truediv__)
ColumnVariable.__rtruediv__ = wrap_npfunc(np.ndarray.__rtruediv__)
ColumnVariable.__pow__      = wrap_npfunc(np.ndarray.__pow__)
ColumnVariable.__rpow__     = wrap_npfunc(np.ndarray.__rpow__)
ColumnVariable.__mod__      = wrap_npfunc(np.ndarray.__mod__)
ColumnVariable.__rmod__     = wrap_npfunc(np.ndarray.__rmod__)
ColumnVariable.__lt__       = wrap_npfunc(np.ndarray.__lt__)
ColumnVariable.__le__       = wrap_npfunc(np.ndarray.__le__)
ColumnVariable.__gt__       = wrap_npfunc(np.ndarray.__gt__)
ColumnVariable.__ge__       = wrap_npfunc(np.ndarray.__ge__)
ColumnVariable.__eq__       = wrap_npfunc(np.ndarray.__eq__)
ColumnVariable.__ne__       = wrap_npfunc(np.ndarray.__ne__)
ColumnVariable.__abs__      = wrap_npfunc(np.ndarray.__abs__, "Absolute value")
ColumnVariable.__neg__      = wrap_npfunc(np.ndarray.__neg__, "Negative value")
ColumnVariable.__pos__      = wrap_npfunc(np.ndarray.__pos__, "The value unchanged")
# }}}

class SpeciesVariable(ColumnVariable):
# {{{ SpeciesVariable class definition
   def __init__(self, name, unit, Nz, initial_value, advect = False, fixed = False, convect = False, output = False, attributes = {}):
   # {{{
      ColumnVariable.__init__(self, name, unit, Nz, initial_value, output, attributes)

      self.advect  = advect
      self.diffuse = advect
      self.fixed  = fixed
      self.convect = convect

      if advect:
         self.surface_flux = 0.
         self.TOA_flux = 0.
   # }}}
# }}}

class State():
# {{{ State class definition
   def __init__(self, columns, scalars, steps = 1):
   # {{{
      self.__dict__['columns'] = {}
      self.__dict__['column_values'] = {}
      self.__dict__['scalars'] = {}
      self.__dict__['scalar_values'] = {}
      self.__dict__['steps'] = steps

      for name, c in columns.items():
         self.columns[name] = c
         self.column_values[name] = np.zeros((steps, c.Nz), c.dtype)
         self.column_values[name][:, :] = c.values.reshape(1, -1)

      for name, s in scalars.items():
         self.scalars[name] = s
         self.scalar_values[name] = np.zeros(steps, c.dtype)
         self.scalar_values[name][:] = s.value
   # }}}

   def __setattr__(self, name, value):
   # {{{
      if name in self.columns:
         self.column_values[name][:] = value

      elif name in self.scalars:
         self.scalar_values[name][:] = value

      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def __getattr__(self, name):
   # {{{
      if name in self.columns:
         return self.column_values[name][:]
      elif name in self.scalars:
         return self.scalar_values[name][:]
      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def create_NC_record_dimension(self, ds : netCDF4.Dataset):
   # {{{
      rec_dim = ds.createDimension("steps", self.steps)

      rec_var = ds.createVariable("steps", 'i8', ("steps",))
      rec_var.setncattr("long_name", "Record dimension")
      rec_var[:] = np.arange(self.steps)

      return rec_dim
   # }}}

   def to_netcdf(self, mc, filename, attributes = {}, full = 'zfull', half = 'zhalf', precision = 'output'):
   # {{{
      with netCDF4.Dataset(filename, 'w', format = 'NETCDF4') as ds:
         # Create record dimension and variable
         rec_dim = self.create_NC_record_dimension(ds)

         # Create column dimensions and coordinate variables
         full_dim = ds.createDimension(full, mc.Nz)
         full_var = mc.grid[full].create_NC_var(ds, (full_dim.name, ), mc.grid[full][:])

         half_dim = ds.createDimension(half, mc.Nz + 1)
         half_var = mc.grid[half].create_NC_var(ds, (half_dim.name, ), mc.grid[half][:])

         # Create column variables
         for name, c in self.columns.items():
            if c.Nz == mc.Nz:
               c.create_NC_var(ds, (rec_dim.name, full_dim.name), self.column_values[name], precision = precision)
            else:
               c.create_NC_var(ds, (rec_dim.name, half_dim.name), self.column_values[name], precision = precision)

         # Create scalar variables
         for name, s in self.scalars.items():
            var = s.create_NC_var(ds, (rec_dim.name, ), self.scalar_values[name], precision = precision)

         # Add global attributes
         for name, value in attributes.items():
            ds.setncattr(name, value)
   # }}}

   def from_netcdf(self, filename):
   # {{{
      with netCDF4.Dataset(filename, 'r', format = 'NETCDF4') as ds:
         # Create column variables
         for name, c in self.columns.items():
            if self.column_values[name].dtype in [complex, np.complex64, np.complex128]:
               self.column_values[name][:] = ds.variables[f'{name}_r'][:] + 1j * ds.variables[f'{name}_i'][:]
            else:
               self.column_values[name][:] = ds.variables[name][:]

         # Create scalar variables
         for name, s in self.scalars.items():
            self.scalar_values[name][:] = ds.variables[name][:]

         return ds.__dict__
   # }}}
# }}}

class OutputState(State):
# {{{ OutputState class definition
   def __init__(self, columns, scalars, times, initial_date = '2000-01-01'):
   # {{{
      State.__init__(self, columns, scalars, len(times))

      self.__dict__['initial_date'] = initial_date
      self.__dict__['times'] = times
   # }}}

   def create_NC_record_dimension(self, ds : netCDF4.Dataset):
   # {{{
      rec_dim = ds.createDimension("time", self.steps)

      rec_var = ds.createVariable("time", 'f8', ("time",))
      rec_var.setncattr("long_name", "time")
      rec_var.setncattr("standard_name", "time")
      rec_var.setncattr("units", f"days since {self.initial_date}")
      rec_var.setncattr("calendar", f"none")

      rec_var[:] = self.times / 86400.

      return rec_dim
   # }}}
# }}}

import pygeode as pyg
def to_pyg(col, out, init = None):
# {{{
   time = pyg.Yearless(out.times / 86400., units = 'days', startdate = dict(year = 1, day = 0))
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

   def add_scalar(name, values, unit):
      axs = (time, )
      v = pyg.Var(axs, name = name, values = values[:].copy())
      v.units = unit
      return v

   vs = []
   for name, cl in out.columns.items():
      if init is None:
         v = out.column_values[name][:]
      else:
         v = out.column_values[name][:] - init.column_values[name][:]

      vs.append(add_var(name, v, cl.units))

   for name, sc in out.scalars.items():
      if init is None:
         v = out.scalar_values[name][:]
      else:
         v = out.scalar_values[name][:] - init.scalar_values[name][:]

      vs.append(add_scalar(name, v, cl.units))

   #for name, var in col.output_variables.items():
      #vs.append(add_var(name, var))

   return pyg.asdataset(vs)
# }}}
