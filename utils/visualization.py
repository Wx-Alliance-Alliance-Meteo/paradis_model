import numpy
import matplotlib.pyplot as plt


def plot_error_map(
    date_in,
    date_out,
    output_data,
    true_data,
    dataset,
    feature,
    cfg,
    level=None,
    temp_offset=0,
    ind=None,
):
    """Plot comparison maps for model output and ground truth."""

    # Determine the feature index and name
    if level is not None:
        # For atmospheric variables with pressure levels
        base_features = cfg.features.output.atmospheric
        level_index = cfg.features.pressure_levels.index(level)
        num_levels = len(cfg.features.pressure_levels)
        base_feature_index = base_features.index(feature)
        feature_index = base_feature_index * num_levels + level_index
    else:
        # For surface variables
        feature_index = dataset.dyn_output_features.index(feature)

    latitude = dataset.lat
    longitude = dataset.lon
    longitude, latitude = numpy.meshgrid(longitude, latitude)

    # Get the forecast data
    output_plot = output_data[feature_index]
    true_plot = true_data[feature_index]
    output_plot = numpy.abs(output_plot - true_plot)

    # Configure plot settings based on variable type
    if feature == "geopotential":
        g = 9.80665  # gravitational acceleration
        cmap = "viridis"
        output_plot = output_plot / g
        vmax = numpy.max(output_plot)
        vmin = numpy.min(output_plot)
        levels = numpy.linspace(vmin, vmax, 50)
        clabel = "Geopotential Height [m]"

    elif feature == "2m_temperature":
        cmap = "RdYlBu_r"
        vmax = numpy.max(output_plot)
        vmin = numpy.min(output_plot)
        levels = numpy.linspace(vmin, vmax, 100)
        clabel = "Temperature [°C]"

    elif feature == "total_precipitation_6hr":
        cmap = "Blues"
        max_precip = numpy.max(output_plot)

        # Create exponentially spaced levels for precipitation
        # This ensures levels are strictly increasing and capture the range of values
        if max_precip > 0:
            # Use exponential spacing to focus on smaller values
            levels = (
                numpy.exp(
                    numpy.linspace(numpy.log(0.1), numpy.log(max_precip + 0.1), 50)
                )
                - 0.1
            )
            # Remove any negative values that might occur due to floating point arithmetic
            levels = levels[levels >= 0]
        else:
            levels = numpy.linspace(0, 0.1, 10)  # Fallback for no precipitation
        clabel = "Precipitation [mm/6h]"

    else:
        cmap = "RdYlBu_r"
        vmax = numpy.max(output_plot)
        vmin = numpy.min(output_plot)
        levels = numpy.linspace(vmin, vmax, 100)
        clabel = feature.replace("_", " ").title()

    # Create figure and axes
    fig, ax = plt.subplots(ncols=1, figsize=(6, 5))

    # Plot contours
    if feature == "total_precipitation_6hr":
        contours = ax.contourf(
            longitude, latitude, output_plot, levels=levels, cmap=cmap, extend="max"
        )
    else:
        contours = ax.contourf(
            longitude, latitude, output_plot, levels=levels, cmap=cmap
        )

    if feature == "geopotential":
        # Add contour lines for geopotential
        contour_levels = levels[::5]  # Take every 5th level
        ax.contour(
            longitude,
            latitude,
            output_plot,
            levels=contour_levels,
            colors="k",
            linewidths=0.5,
        )

    # Set titles
    title = (
        f"{feature} error at {level} hPa"
        if level
        else feature.replace("_", " ").title()
    )
    plt.suptitle(f"{title}\nForecast date: {date_out}\nInput date: {date_in}")
    ax.set_title(f"PARADIS")

    # Adjust layout and add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contours, cax=cbar_ax)
    cbar.ax.set_ylabel(clabel, rotation=90)

    # Save figure
    filename = (
        f"results/{feature}_{level}hPa_prediction_error"
        if level
        else f"results/{feature}_prediction_error"
    )
    if ind is not None:
        filename += "_" + str(ind)
    filename += ".png"

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(plt.gcf())

    return numpy.max(output_plot)


def plot_forecast_map(
    date_in,
    date_out,
    output_data,
    true_data,
    dataset,
    feature,
    cfg,
    level=None,
    temp_offset=0,
    ind=None,
):
    """Plot comparison maps for model output and ground truth."""

    # Determine the feature index and name
    if level is not None:
        # For atmospheric variables with pressure levels
        base_features = cfg.features.output.atmospheric
        level_index = cfg.features.pressure_levels.index(level)
        num_levels = len(cfg.features.pressure_levels)
        base_feature_index = base_features.index(feature)
        feature_index = base_feature_index * num_levels + level_index
        feature_name = f"{feature}_h{level}"
    else:
        # For surface variables
        feature_index = dataset.dyn_output_features.index(feature)
        feature_name = feature

    latitude = dataset.lat
    longitude = dataset.lon
    longitude, latitude = numpy.meshgrid(longitude, latitude)

    # Get the forecast data
    output_plot = output_data[feature_index]
    true_plot = true_data[feature_index]

    # Configure plot settings based on variable type
    if feature == "geopotential":
        g = 9.80665  # gravitational acceleration
        output_plot = output_plot / g
        true_plot = true_plot / g
        cmap = "viridis"
        vmax = 5900 #numpy.max([numpy.max(output_plot), numpy.max(true_plot)])
        vmin = 4900 #numpy.min([numpy.min(output_plot), numpy.min(true_plot)])
        levels = numpy.linspace(vmin, vmax, 50)
        clabel = "Geopotential Height [m]"

    elif feature == "2m_temperature":
        cmap = "RdYlBu_r"
        output_plot = output_plot - temp_offset
        true_plot = true_plot - temp_offset
        vmax = numpy.max([numpy.max(output_plot), numpy.max(true_plot)])
        vmin = numpy.min([numpy.min(output_plot), numpy.min(true_plot)])
        levels = numpy.linspace(vmin, vmax, 100)
        clabel = "Temperature [°C]"

    elif feature == "total_precipitation_6hr":
        cmap = "Blues"
        max_precip = max(numpy.max(output_plot), numpy.max(true_plot))

        # Create exponentially spaced levels for precipitation
        # This ensures levels are strictly increasing and capture the range of values
        if max_precip > 0:
            # Use exponential spacing to focus on smaller values
            levels = (
                numpy.exp(
                    numpy.linspace(numpy.log(0.1), numpy.log(max_precip + 0.1), 50)
                )
                - 0.1
            )
            # Remove any negative values that might occur due to floating point arithmetic
            levels = levels[levels >= 0]
        else:
            levels = numpy.linspace(0, 0.1, 10)  # Fallback for no precipitation
        clabel = "Precipitation [mm/6h]"

    else:
        cmap = "RdYlBu_r"
        vmax = numpy.max([numpy.max(output_plot), numpy.max(true_plot)])
        vmin = numpy.min([numpy.min(output_plot), numpy.min(true_plot)])
        levels = numpy.linspace(vmin, vmax, 100)
        clabel = feature.replace("_", " ").title()

    # Create figure and axes
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

    # Plot contours
    for i, data in enumerate([output_plot, true_plot]):
        if feature == "total_precipitation_6hr":
            contours = ax[i].contourf(
                longitude, latitude, data, levels=levels, cmap=cmap, extend="max"
            )
        else:
            contours = ax[i].contourf(
                longitude, latitude, data, levels=levels, cmap=cmap
            )

        if feature == "geopotential":
            # Add contour lines for geopotential
            contour_levels = levels[::5]  # Take every 5th level
            ax[i].contour(
                longitude,
                latitude,
                data,
                levels=contour_levels,
                colors="k",
                linewidths=0.5,
            )

    # Set titles
    title = f"{feature} at {level} hPa" if level else feature.replace("_", " ").title()
    plt.suptitle(f"{title}\nForecast date: {date_out}\nInput date: {date_in}")
    ax[0].set_title(f"PARADIS")
    ax[1].set_title(f"ERA5")

    # Adjust layout and add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contours, cax=cbar_ax)
    cbar.ax.set_ylabel(clabel, rotation=90)

    # Save figure
    filename = (
        f"results/{feature}_{level}hPa_prediction"
        if level
        else f"results/{feature}_prediction"
    )
    if ind is not None:
        filename += "_" + str(ind)
    filename += ".png"

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(plt.gcf())
