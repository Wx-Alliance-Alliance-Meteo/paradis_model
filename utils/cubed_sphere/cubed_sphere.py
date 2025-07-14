import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

class CubedSphere:
    def __init__(self, num_elem: int, lambda0: float = 0.0, phi0: float = 0.0, alpha0: float = 0.0, padding_width: int = 0):
        """Initialize the cubed sphere geometry, for an earthlike sphere with no topography.

        This function initializes the basic CubedSphere geometry object, which provides the parameters necessary
        to define the values in numeric (x1, x2) coordinates, gnomonic (projected; X, Y) coordinates,
        spherical (lat, lon), Cartesian (Xc, Yc, Zc) coordinates.

        The cubed-sphere panelization is as follows:
        ```
              +---+
              | 4 |
          +---+---+---+---+
          | 3 | 0 | 1 | 2 |
          +---+---+---+---+
              | 5 |
              +---+
        ```
        where each panel has its own local (x1,x2) coordinate axis, representing the angular deviation from
        the panel center.  With typical parameters, panel 0 contains the intersection of the prime meridian and
        equator, the equator runs through panels 3-0-1-2 from west to east, panel 4 contains the north pole,
        and panel 5 contains the south pole.

        Parameters:
        -----------
        num_elem: int
           Number of elements in the (x1,x2) directions, per panel
        lambda0: float
           Grid rotation: physical longitude of the central point of the 0 panel
           Valid range ]-π/2,0]
        phi0: float
           Grid rotation: physical latitude of the central point of the 0 panel
           Valid range ]-π/4,π/4]
        alpha0: float
           Grid rotation: rotation of the central meridian of the 0 panel, relatve to true north
           Valid range ]-π/2,0]
        """
        self.num_panel = 6
        self.num_elem = num_elem
        self.padding_width = padding_width

        # Get base panel vectors and apply rotation
        panel_center, panel_up, panel_right = self.get_panel_faces()
        R = Rotation.from_euler('xyz', [-alpha0, -phi0, lambda0])
        self.panel_center = R.apply(panel_center)
        self.panel_up = R.apply(panel_up)
        self.panel_right = R.apply(panel_right)

        # Physical constants
        self.earth_radius = 6371220.0  # Mean radius of the earth (m)
        
        # Local coordinate 
        domain = (-np.pi / 4, np.pi / 4)
        delta_x = (domain[1] - domain[0]) / num_elem
        
        self.xi = domain[0] + delta_x * (np.arange(-padding_width, num_elem+padding_width) + 0.5)   # West to east
        self.eta = domain[0] + delta_x * (np.arange(-padding_width, num_elem+padding_width) + 0.5)  # South to north
        
        self.Xi, self.Eta = np.meshgrid(self.xi, self.eta, indexing='ij')
        # Xi and Eta are the same on all faces so we set first dim to size 1 so broadcast works
        self.Xi = self.Xi[None, ...] 
        self.Eta = self.Eta[None, ...] 
        
        self.coord_map = {
            'local': 0,
            'gnomonic': 1,
            'cartesian': 2,
            'spherical': 3,
            'physical': 4
        }
        
        vars = self.transform_coord('local', 'spherical', xi=self.Xi, eta=self.Eta)
        self.X = vars['X']
        self.Y = vars['Y']
        self.delta = np.sqrt(1.0 + self.X**2 + self.Y**2)

        self.Xc = vars['Xc']
        self.Yc = vars['Yc']
        self.Zc = vars['Zc']
        
        self.lat = vars['lat']
        self.lon = vars['lon']
        self.cos_lat = np.cos(self.lat)
        self.sin_lat = np.sin(self.lat)
        self.cos_lon = np.cos(self.lon)
        self.sin_lon = np.sin(self.lon)
        
        # Add latlon in degree 
        lon_deg = np.rad2deg(self.lon) % 360
        lat_deg = np.rad2deg(self.lat)

        self.lonlat = np.stack((lon_deg, lat_deg), axis=-1)
    
    def get_panel_faces(self):
        panel_center = np.array([
            [ 1, 0, 0],
            [ 0, 1, 0],
            [-1, 0, 0],
            [ 0,-1, 0],
            [ 0, 0, 1],
            [ 0, 0,-1],
        ])

        panel_up = np.array([
            [ 0,  0,  1],
            [ 0,  0,  1],
            [ 0,  0,  1],
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 1,  0,  0],
        ])

        panel_right = np.cross(panel_up, panel_center)

        return panel_center, panel_up, panel_right
    
    # Coordinate transform
    def _local_to_gnomonic(self, xi, eta):
        X = np.tan(xi)
        Y = np.tan(eta)
        return X, Y

    def _J_local_to_gnomonic(self, X, Y):
        J = np.zeros(X.shape + (2,2))
        J[..., 0, 0] = 1 + X**2
        J[..., 1, 1] = 1 + Y**2
        return J

    def _gnomonic_to_local(self, X, Y):
        xi = np.arctan(X)
        eta = np.arctan(Y)
        return xi, eta
    
    def _J_gnomonic_to_local(self, X, Y):
        J = np.zeros(X.shape + (2,2))
        J[..., 0, 0] = 1 / (1 + X**2)
        J[..., 1, 1] = 1 / (1 + Y**2)
        return J

    def _gnomonic_to_cartesian(self, X, Y, panel_id=None):
        if panel_id is not None:  # Compute for a specific panel. Dimension (1, num_elem, num_elem, 1)
            c = self.panel_center[panel_id, None, None, :] 
            r = self.panel_right[panel_id, None, None, :]
            u = self.panel_up[panel_id, None, None, :]
            X = X[..., None]
            Y = Y[..., None]
        else: # Compute for all panel. Dimension (num_panel, num_elem, num_elem, 1)
            c = self.panel_center[:, None, None, :]
            r = self.panel_right[:, None, None, :]
            u = self.panel_up[:, None, None, :]
            if X.ndim == 2:
                X = X[None, ..., None]
                Y = Y[None, ..., None]
            else: 
                X = X[..., None]
                Y = Y[..., None]

        delta = np.sqrt(1.0 + X**2 + Y**2)
        points = (c + r * X + u * Y) / delta
        Xc, Yc, Zc = points[..., 0], points[..., 1], points[..., 2]
        return Xc, Yc, Zc
    
    def _J_gnomonic_to_cartesian(self, X, Y, delta, c, r, u):
        J = np.zeros((6,) + X.shape[1:3] + (3,2))
        
        X, Y, delta = X[..., None], Y[..., None], delta[..., None]
        c, r, u = c[:,None,None,:], r[:,None,None,:], u[:,None,None,:]
        
        J[..., 0] = (r * (1+Y**2) - X * (c + u * Y)) / delta ** 3
        J[..., 1] = (u * (1+X**2) - Y * (c + r * X)) / delta ** 3
        return J

    def _cartesian_to_gnomonic(self, Xc, Yc, Zc):
        points = np.stack([Xc, Yc, Zc], axis=-1)
        
        # Compute dot products with panel centers to find closest panel
        p_dot_cs = np.sum(points[..., None, :] * self.panel_center, axis=-1) 
        panel_indices = np.argmax(p_dot_cs, axis=-1)
    
        # Select panel vectors for each point based on panel_indices 
        r = self.panel_right[panel_indices]   
        u = self.panel_up[panel_indices]   

        p_dot_c = p_dot_cs[panel_indices]   
        p_dot_r = np.sum(points * r, axis=-1)
        p_dot_u = np.sum(points * u, axis=-1)

        # Project vec onto panel_right and panel_up basis to get gnomonic coords X, Y
        X = p_dot_r / p_dot_c
        Y = p_dot_u / p_dot_c
        
        return X, Y, panel_indices # Return panel_indices as part of the transformation result

    def _J_cartesian_to_gnomonic(self, X, Y, delta, c, r, u):
        J = np.zeros((6,) + X.shape[1:3] + (2,3))
        
        X, Y, delta = X[..., None], Y[..., None], delta[..., None]
        c, r, u = c[:,None,None,:], r[:,None,None,:], u[:,None,None,:]
        
        J[..., 0, :] = delta * (r - X*c)
        J[..., 1, :] = delta * (u - Y*c)
        return J

    def _spherical_to_cartesian(self, lat, lon):
        Xc = np.cos(lat) * np.cos(lon)
        Yc = np.cos(lat) * np.sin(lon)
        Zc = np.sin(lat)
        return Xc, Yc, Zc

    def _J_spherical_to_cartesian(self, cos_lat, sin_lat, cos_lon, sin_lon):
        J = np.zeros(cos_lat.shape + (3,2))
        J[..., 0, 0] = -sin_lat*cos_lon
        J[..., 0, 1] = -cos_lat*sin_lon
        J[..., 1, 0] = -sin_lat*sin_lon
        J[..., 1, 1] = cos_lat*cos_lon
        J[..., 2, 0] = cos_lat
        return J

    def _cartesian_to_spherical(self, Xc, Yc, Zc):
        hypotxy = np.hypot(Xc, Yc)
        lat = np.arctan2(Zc, hypotxy)
        lon = np.arctan2(Yc, Xc)
        return lat, lon
    
    def _J_cartesian_to_spherical(self, cos_lat, cos_lon, sin_lon):
        J = np.zeros(cos_lat.shape + (2,3))
        J[..., 0, 2] = 1/cos_lat
        J[..., 1, 0] = -sin_lon/cos_lat
        J[..., 1, 1] = cos_lon/cos_lat
        return J

    def _J_spherical_to_physical(self, cos_lat, Re):
        J = np.zeros(cos_lat.shape + (2,2))
        J[..., 0, 0] = Re * cos_lat
        J[..., 1, 1] = Re
        return J
    
    def _J_physical_to_spherical(self, cos_lat, Re):
        J = np.zeros(cos_lat.shape + (2,2))
        J[..., 0, 0] = 1/(Re*cos_lat)
        J[..., 1, 1] = 1/Re
        return J
    
    def transform_coord(self, from_coord, to_coord, **vars):
        if from_coord not in self.coord_map and from_coord != 'physical':
            raise ValueError(f"Invalid from_coord: {from_coord}")
        if to_coord not in self.coord_map and to_coord != 'physical':
            raise ValueError(f"Invalid to_coord: {to_coord}")
        
        from_idx = self.coord_map[from_coord]
        to_idx = self.coord_map[to_coord]
        direction = np.sign(to_idx - from_idx)
        
        while from_idx != to_idx:
            if from_idx == 0 and direction == 1:       # Local to Gnomonic
                vars['X'], vars['Y'] = self._local_to_gnomonic(vars['xi'], vars['eta'])
            elif from_idx == 1 and direction == -1:    # Gnomonic to Local
                vars['xi'], vars['eta'] = self._gnomonic_to_local(vars['X'], vars['Y'])
            elif from_idx == 1 and direction == 1:     # Gnomonic to Cartesian
                vars['Xc'], vars['Yc'], vars['Zc'] = self._gnomonic_to_cartesian(vars['X'], vars['Y'])
            elif from_idx == 2 and direction == -1:    # Cartesian to Gnomonic
                vars['X'], vars['Y'], vars['panel_id'] = self._cartesian_to_gnomonic(vars['Xc'], vars['Yc'], vars['Zc'])
            elif from_idx == 2 and direction == 1:     # Cartesian to Spherical
                vars['lat'], vars['lon'] = self._cartesian_to_spherical(vars['Xc'], vars['Yc'], vars['Zc'])
            elif from_idx == 3 and direction == -1:    # Spherical to Cartesian
                vars['Xc'], vars['Yc'], vars['Zc'] = self._spherical_to_cartesian(vars['lat'], vars['lon'])
            
            from_idx += direction
        
        return vars
            
    
    def get_jacobian(self, from_coord, to_coord):
        if from_coord not in self.coord_map:
            raise ValueError(f"Invalid from_coord: {from_coord}")
        if to_coord not in self.coord_map:
            raise ValueError(f"Invalid to_coord: {to_coord}")
        
        from_idx = self.coord_map[from_coord]
        to_idx = self.coord_map[to_coord]
        direction = np.sign(to_idx - from_idx)
        
        cs_shape = self.X.shape
        X, Y, delta = self.X, self.Y, self.delta
        c, u, r = self.panel_center, self.panel_up, self.panel_right
        cos_lat, cos_lon = self.cos_lat, self.cos_lon
        sin_lat, sin_lon = self.sin_lat, self.sin_lon
        Re = self.earth_radius
        
        n_from = 3 if from_coord == 'cartesian' else 2
        
        # Initialize Jacobian to identity
        J = np.zeros(cs_shape + (n_from, n_from))
        for i in range(n_from):
                J[..., i, i] = 1
        
        while from_idx != to_idx:
            if from_idx == 0 and direction == 1:       # Local to Gnomonic
                J_cur = self._J_local_to_gnomonic(X, Y)
            elif from_idx == 1 and direction == -1:    # Gnomonic to Local
                J_cur = self._J_gnomonic_to_local(X, Y)
            elif from_idx == 1 and direction == 1:     # Gnomonic to Cartesian
                J_cur = self._J_gnomonic_to_cartesian(X, Y, delta, c, r, u)
            elif from_idx == 2 and direction == -1:    # Cartesian to Gnomonic
                J_cur = self._J_cartesian_to_gnomonic(X, Y, delta, c, r, u)
            elif from_idx == 2 and direction == 1:     # Cartesian to Spherical
                J_cur = self._J_cartesian_to_spherical(cos_lat, cos_lon, sin_lon)
            elif from_idx == 3 and direction == -1:    # Spherical to Cartesian
                J_cur = self._J_spherical_to_cartesian(cos_lat, sin_lat, cos_lon, sin_lon)
            elif from_idx == 3 and direction == 1:     # Spherical to Physical
                J_cur = self._J_spherical_to_physical(cos_lat, Re)
            elif from_idx == 4 and direction == -1:    # Physical to Spherical
                J_cur = self._J_physical_to_spherical(cos_lat, Re)
            
            # Update Jacobian 
            J = J_cur @ J
            from_idx += direction
        
        return J

    def contra2wind(self, u1u2):
        """
        Convert from contravariant winds (computational) to physical winds.

        Parameters:
        -----------
        u1, u2 : NDArray
           Contravariant winds along the local x1 and x2 directions.
           Shape: (num_panel, num_elem, num_elem)

        Returns:
        --------
        u, v : tuple[NDArray, NDArray]
           Zonal (u) and meridional (v) winds, in m/s.
        """

        # Get the Jacobian of the transformation from local to physical
        J = self.get_jacobian(from_coord='local', to_coord='physical')
        uv = J @ u1u2
        
        return uv

    def wind2contra(self, u, v, xi, eta):
        """
        Convert physical winds (zonal, meridional) to contravariant winds.

        Parameters:
        ----------
        u, v : NDArray
           Input zonal and meridional winds, in meters per second.
           Shape: (num_panel, num_elem, num_elem)
        xi, eta : NDArray
            Local angular coordinates for each grid point.

        Returns:
        -------
        u1u2 : NDArray
           Contravariant winds along the local x1 and x2 directions.
        """
        
        # Get the Jacobian of the transform from physical to pysical to local
        J = self.get_jacobian(from_coord='physical', to_coord='local')
        u1u2 = J @ np.stack([u, v], axis=-1)
        
        return u1u2
        

        
    def plot_data(self, data, remove_halo=False): 
        # Define the resolution for the faces
        res = data.shape[1]
        
        # Remove halo
        if remove_halo:
            data = data[:, 1:-1, 1:-1]

        # Create a canvas for the cross pattern
        width = 4 * res
        height = 3 * res
        cs_img = np.full((height, width), np.nan)

        # Define the positions for each face
        positions = {
            0: (res, res),     # Front
            1: (res, 2 * res), # Right
            2: (res, 3 * res), # Back
            3: (res, 0),              # Left
            4: (0, res),              # Top
            5: (2 * res, res)  # Bottom
        }

        # Place each face onto the canvas
        for i in range(6):
            y, x = positions[i]
            cs_img[y:y+res, x:x+res] = np.flipud(data[i].T)

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(cs_img)
        return fig
