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

The vertical velocity and radiative fluxes are defined on the half-levels, while the temperatures
and chemical species are defined on full-levels.

### Advection

The advection algorithm is still being developed. First attempt was to use a centered difference in space 
designed to conserve the variance of the quantity being advected, along with a leap-frog time step. In this case, 
The advective tendency for a quantity $X$ at full level $j$ is approximated as

$$ (\omega \frac{dX}{dp})_j = \frac{1}{2} \omega_{i + \frac{1}{2}} \frac{X_{j+1} - X_j}{\Delta p_j}
+ \frac{1}{2} \omega_{i - \frac{1}{2}} \frac{X_{j} - X_{j-1}}{\Delta p_j} $$

This results in oscillations around regions of sharp curvature which might introduce things like negative values of tracer constituents.

I'm currently exploring a conservative, semi-Lagrangian scheme following Kaas
(2008; doi:10.1111/j.1600-0870.2007.00293.x). This involves computing origin
points for Lagrangian trajectories that end at each grid point, and spatially
interpolating the tracer field to these points. Computing the back trajectories
has been implemented following Kass (2008) (though see McGregor 1993,
doi:10.1175/1520-0493(1993)121<0221:EDODPF>2.0.CO;2 as well) and I think should be reasonbaly
efficient in this Python implementation.

The spatial interpolation has been a source of significant biases around the
tropical tropopause (see Hardiman et al. 2015; doi: 10.1175/JCLI-D-15-0075.1),
although the difficulties arise from mostly-reversable wave motions that
generate spurious irreversable transport around sharp changes in gradients.

To implement this I need to be able to write a piecewise cubic interpolation
scheme in terms of a matrix multiplication, i.e., the value of a tracer at the
origin points $z*_j$ for the trajectories ending at $z_k$ should be expressible
as 

$$ X*_j = L_{jk} X_k, $$

where the matrix will have non-zero entries for 4 levels around the origin
location. This form is then re-normalized to ensure global conservation of the
tracer mass. The re-normalization provides an estimate of the local divergence
which could potentially be used as a way to parameterize meridional transport.

This should be possible for a Hermite cubic polynomial interpolation, although
the entries get complicated depending on what expression one wants to use for
the derivatives at the end point (see appendix A of Williamson 1990;
doi:10.3402/TELLUSA.V42I4.11887). 

I still need to work out how to deal with fluxes at the boundaries in this framework, 
as well as how to deal with the background density effects.

I would also like to come up with some numerical tests to explore the
properties of the scheme once it has been implemented, including things like
global conservation.  The tape recorder might be a good one to explore.

### Chemistry

### Photolysis Rates

### Radiative Heating
