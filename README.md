# column

Radiative/photochemical column model built around MUSICA and TUVx

## Dependencies

    - Musica
    - RRTMg
    - TUV-x (not yet)

## Numerical Methods

### Grid

The column is defined in log-pressure height coordinates from the ground to some upper boundary; it
is designed to be compatible with the RRTMg grid, so there are full levels (i = 1,...,N) and half levels
at the interfaces of the full levels.

The vertical velocity and radiative fluxes are defined on the half-levels,
while the temperatures and chemical species are defined on full-levels.

### Advection

The advection algorithm is still being developed. First attempt was to use a
centered difference in space designed to conserve the variance of the quantity
being advected, along with a leap-frog time step. In this case, The advective
tendency for a quantity $X$ at full level $j$ is approximated as

$$ (\omega \frac{dX}{dp})_j = \frac{1}{2} \omega_{i + \frac{1}{2}} \frac{X_{j+1} - X_j}{\Delta p_j}
+ \frac{1}{2} \omega_{i - \frac{1}{2}} \frac{X_{j} - X_{j-1}}{\Delta p_j} $$

This results in oscillations around regions of sharp curvature which might
introduce things like negative values of tracer constituents.

The next step was to implement a conservative, semi-Lagrangian scheme following
Kaas (2008; doi:10.1111/j.1600-0870.2007.00293.x). This involves computing
origin points for Lagrangian trajectories that end at each grid point, and
spatially interpolating the tracer field to these points. Computing the back
trajectories has been implemented following Kass (2008) (though see McGregor
1993, doi:10.1175/1520-0493(1993)121<0221:EDODPF>2.0.CO;2 as well) and I think
should be reasonbly efficient in this Python implementation.

This involves computing a set of interpolation weights such that the value of a tracer at the
origin points $z*_j$ for the trajectories ending at $z_k$ can be expressed
as a matrix multiplication: 

$$ X*_j = w_{jk} X_k. $$

The weights matrix will have non-zero entries for 4 levels around the origin
location. This form is then re-normalized to ensure global conservation of the
tracer mass. The re-normalization provides an estimate of the local divergence
which could potentially be used as a way to parameterize meridional transport.

At the moment the spatial interpolation required to compute tracer values at
the origin points is a cubic Hermite polynomial. This interpolation has been a
source of significant biases around the tropical tropopause (see Hardiman et
al. 2015; doi: 10.1175/JCLI-D-15-0075.1), although the difficulties in their
model apparently arose from mostly-reversable wave motions that generate
spurious irreversable transport around sharp changes in gradients.

The interpolation is not 'shape-preserving', meaning that it can still
introduce oscillations around regions of large curvature. This can be mitigated
by implementing a monotone interpolation scheme so that the interpolated curve
does not exhibit maxima that are not in un-interpolated gridded values (see
Rasch and Williamson 1990; doi: 10.1137/0911039). This will require that the
interpolation weights will be (potentially) different for each advected
quantity. It might ultimately be worth implementing this all in a Cython module
for efficiency.

There is a test for the advection scheme in col_tests.py that initializes a
local patch of tracer concentration then advects it up and down periodically.
There are boundary effects that can arise which I think have to do with the
normalization of the weights around the upper and lower boundaries of the
domain. I think this will be mitigated by working out how to deal with fluxes
at the boundaries in this framework.

#### To Do:
    -Work out fluxes at boundaries, and a better normalization scheme
    -Implement shape-preserving interpolation
    -Implement arbitrary list of advected species
    -Implement potential temperature advection
    -Add advection test with divergent flow

### Chemistry

#### To Do:
    -Try to write a mechanism that implements a simple decay rate?
    -Work out a good operator splitting scheme to call MUSICA solver

### Photolysis Rates

#### To Do:
    -Work out how to call TUVx and have it integrate with MUSICA

### Radiative Heating

#### To Do:
    -Work out a good operator splitting scheme to call RRTM
    -Revisit unit conversion
