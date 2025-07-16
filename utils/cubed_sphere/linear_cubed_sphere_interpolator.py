import numpy as np

class LinearCubedSphereInterpolator:
    def __init__(self, lat, lon, cubed_sphere):
        dlon = lon[1]-lon[0]
        dlat = lat[1]-lat[0]
        
        lon = np.concat((lon, [360]))

        lon_cs_deg = np.rad2deg(cubed_sphere.lon) % 360
        lat_cs_deg = np.rad2deg(cubed_sphere.lat)
        idx_lon = np.floor((lon_cs_deg - lon[0])/dlon).astype(int)
        idx_lat = np.floor((lat_cs_deg - lat[0])/dlat).astype(int)

        # Clip indices to ensure that idx and idx+1 are within valid bounds
        self.idx_lon = np.clip(idx_lon, 0, len(lon) - 2)
        self.idx_lat = np.clip(idx_lat, 0, len(lat) - 2)

        # Get the coordinates of the four corners of the cell
        x1 = lon[self.idx_lon]
        x2 = lon[self.idx_lon + 1]
        y1 = lat[self.idx_lat]
        y2 = lat[self.idx_lat + 1]

        # Calculate the normalized distances
        self.tx = (lon_cs_deg - x1) / (x2 - x1)
        self.ty = (lat_cs_deg - y1) / (y2 - y1)
        
    def interpolate(self, data):
        data = np.concat((data, data[..., 0:1]), axis=-1)
        
        # Get data at the corners
        q11 = data[..., self.idx_lat,     self.idx_lon]        
        q12 = data[..., self.idx_lat,     self.idx_lon + 1]    
        q21 = data[..., self.idx_lat + 1, self.idx_lon]    
        q22 = data[..., self.idx_lat + 1, self.idx_lon + 1]

        # Perform linear interpolation along x-direction
        R1 = q11 * (1 - self.tx) + q12 * self.tx
        R2 = q21 * (1 - self.tx) + q22 * self.tx

        # Perform linear interpolation along y-direction
        interpolated_values = R1 * (1 - self.ty) + R2 * self.ty
        
        return interpolated_values
