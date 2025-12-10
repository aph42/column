# column

Radiative/photochemical column model built around MUSICA and TUVx

## Dependencies

    - Musica
    - RRTMg
    - TUV-x (not yet)

## Grid

The column is defined in log-pressure height coordinates from the ground to some upper boundary; it
is designed to be compatible with the RRTMg grid, so there are full levels (i = 1,...,N) and half levels
at the interfaces of the full levels.

The vertical velocity and radiative fluxes are defined on the half-levels, while the temperatures
and chemical species are defined on full-levels.

## Advection

The advective tendency for a quantity $X$ at full level $j$ is approximated as

$$ (\omega \frac{dX}{dp})_j = \frac{1}{2} \omega_{i + \frac{1}{2}} \frac{X_{j+1} - X_j}{\Delta p_j}
+ \frac{1}{2} \omega_{i - \frac{1}{2}} \frac{X_{j} - X_{j-1}}{\Delta p_j} $$

