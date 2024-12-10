import os

import matplotlib.pyplot as plt

def visualize_predictions(lat, lon, true_data, output_data, output_dir='output'):
    # Extract features for a specific variable (e.g., u-component of wind at 10m)
    feature_index = 20

    # Load latitude and longitude dimensions
    lat_size = len(lat)
    lon_size = len(lon)

    # Extract and reshape the feature data for visualization
    feature_data = true_data[:, feature_index]  # Extract data for the desired feature
    feature_data_reshaped = feature_data.detach().cpu().numpy().reshape(lat_size, lon_size)  # Detach, move to CPU and convert to numpy

    # Visualize the true data feature
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(feature_data_reshaped, cmap='coolwarm')
    plt.colorbar(label='Feature Value')
    plt.title(f"True Data - Feature {feature_index}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Save or display the plot for true data
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/true_feature_{feature_index}.png')
    else:
        plt.show()

    # Visualize the output data feature if available
    if output_data is not None:
        output_feature_data = output_data[:, feature_index]
        output_feature_data_reshaped = output_feature_data.detach().cpu().numpy().reshape(lat_size, lon_size)  # Detach, move to CPU and convert to numpy

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(output_feature_data_reshaped, cmap='coolwarm')
        plt.colorbar(label='Output Value')
        plt.title(f"Output Data - Feature {feature_index}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Save or display the plot for output data
        if output_dir:
            plt.savefig(f'{output_dir}/output_feature_{feature_index}.png')
        else:
            plt.show()

    plt.close()