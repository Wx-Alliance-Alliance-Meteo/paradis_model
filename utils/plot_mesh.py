"""Visualization utilities for the cubed sphere graph."""

import numpy
import pyvista
from typing import Optional, Tuple
from graph.cubed_sphere import CubedSphereGraphBuilder


def visualize_cubed_sphere_graph(
    vertices: numpy.ndarray,
    edges: numpy.ndarray,
    grid_points: Optional[numpy.ndarray] = None,
    grid_to_mesh_edges: Optional[Tuple[numpy.ndarray, numpy.ndarray]] = None,
    show_vertices: bool = True,
    show_edges: bool = True,
    background_color: str = "black",
    window_size: Tuple[int, int] = (1024, 768),
    save_path: Optional[str] = None,
) -> None:
    """Visualize the cubed sphere graph structure.

    Args:
        vertices: [N, 3] array of vertex coordinates (cell centers)
        edges: [M, 2] array of edge indices connecting vertices
        grid_points: Optional [P, 3] array of grid point coordinates
        grid_to_mesh_edges: Optional tuple of (senders, receivers) for grid-to-mesh connections
        show_vertices: Whether to show the cell center vertices
        show_edges: Whether to show connections between vertices
        background_color: Color of the visualization background
        window_size: Size of the visualization window
        save_path: Optional path to save the visualization image
    """
    pl = pyvista.Plotter(window_size=window_size)
    pl.set_background(background_color)

    if show_vertices:
        vertices_cloud = pyvista.PolyData(vertices)
        pl.add_mesh(
            vertices_cloud, color="black", point_size=8, render_points_as_spheres=True
        )

    if show_edges:
        for edge in edges:
            start, end = edge
            line = pyvista.Line(vertices[start], vertices[end])
            pl.add_mesh(line, color="blue", line_width=2, opacity=0.5)

    # Add axes for reference
    pl.add_axes()

    # Save if path provided
    if save_path:
        pl.show(screenshot=save_path)
    else:
        pl.show()


if __name__ == "__main__":

    # Create sample lat/lon grid
    grid_lat = numpy.linspace(-90, 90, 73)
    grid_lon = numpy.linspace(0, 360, 144)

    # Initialize graph builder with lower resolution for visualization
    graphs = CubedSphereGraphBuilder(
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        resolution=4,  # Using small resolution for clear visualization
    )

    visualize_cubed_sphere_graph(
        vertices=graphs.vertices,
        edges=graphs.edges,
        show_vertices=True,
        show_edges=True,
    )
