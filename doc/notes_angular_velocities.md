# Advection on the sphere using angular velocities

Given a point on the sphere in cartesian coordinates at time $t: x_t$ and a rigid rotation with angular velocity $\omega \in \mathbb{R}^3$, the position of the point will follow the following ODE: 
$\dot{x} = \omega \times x$ with initial contidition $x(t) = x_t$.

We consider that the angular velocity is constant over the time step. 
The solution is given by the [Rodrigues rotation formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).

$x(t_{n+1}) = x_t \cos(\theta) + (\hat{\omega} \times x_t) \sin(\theta) + \hat{\omega} (\hat{\omega} \cdot x_t) (1 - \cos(\theta))  $ where $\theta = ||\omega|| t$ and $\hat{\omega} = \frac{\omega}{||\omega||}$

## Implementation

### 1. Coordinate conversion
Convert latitude/longitude to Cartesian coordinates:
```
x0 = [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]
```

### 2. Rotation parameters
- Compute rotation angle: $\theta = -||\omega|| \cdot dt$ (negative for backward advection)
- Normalize angular velocity: $\hat{\omega} = \frac{\omega}{||\omega||}$

### 3. Rodrigues formula application
Apply the rotation formula:
```
x_new = x0 * cos(θ) + (û × x0) * sin(θ) + û * (û · x0) * (1 - cos(θ))
```

### 4. Normalization and conversion back
- Normalize the result to ensure it lies on the unit sphere
- Convert back to spherical coordinates:
  - `lat_dep = arcsin(z_component)`
  - `lon_dep = atan2(y_component, x_component)`

This method computes departure points for semi-Lagrangian advection by rotating each grid point backward in time using the learned angular velocity field.
