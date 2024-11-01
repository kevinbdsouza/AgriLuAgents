import pandas as pd

from optim import *
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely.wkt import loads as wkt_loads

# Parameter sweep ranges
PARAMETER_RANGES = {
    'alpha': [0.5, 2.5, 5, 10],
    'delta': [0.5, 2.5, 5, 10],
    'beta': [0.005, 0.05, 0.5, 2],
    'epsilon': [0.005, 0.05, 0.5, 2],
    'gamma': [0.005, 0.05, 0.5, 2],
    'zeta': [0.005, 0.05, 0.5, 2],
    'r': [0.05, 0.1],
    't': [10, 20, 40],
    'margin_costs': [200, 500, 1000, 2000],
    'habitat_costs': [200, 500, 1000, 2000],
    'maintenance_costs': [200, 500, 1000, 2000],
    'p_c': [500, 1000, 2000]
}


def sweep_parameters(farm_gdf, params):
    """
    Sweep through parameter ranges, optimizing farm configuration for each set of parameters.
    """
    base_params = params.copy()
    for param_name, values in PARAMETER_RANGES.items():
        for value in values:

            print("Running param: {} - value {}".format(param_name, value))

            # Update the parameter value for sweeping
            current_params = base_params.copy()
            if param_name == 'margin_costs':
                current_params['costs']['margin']['implementation'] = value
            elif param_name == 'habitat_costs':
                current_params['costs']['habitat']['implementation'] = value
            elif param_name == 'maintenance_costs':
                current_params['costs']['margin']['maintenance'] = value
                current_params['costs']['habitat']['maintenance'] = value
            else:
                # Update value in all crop parameters
                for crop in current_params['crops'].values():
                    if param_name in crop['margin']:
                        crop['margin'][param_name] = value
                    elif param_name in crop['habitat']:
                        crop['habitat'][param_name] = value
                    else:
                        crop[param_name] = value

            # Run optimization
            result = optimize_farm(farm_gdf, current_params)

            # Store results
            result_file = os.path.join(RESULTS_DIR, f"{param_name}_{value}.geojson")
            farm_gdf_copy = farm_gdf.copy()

            farm_gdf_copy['margin_intervention'] = result.x[:len(farm_gdf)]
            farm_gdf_copy['habitat_conversion'] = result.x[len(farm_gdf):]

            farm_gdf_copy['margin_intervention'] = farm_gdf_copy.apply(
                lambda row: row['margin_intervention'] if row['type'] == 'ag_plot' and row[
                    'margin_intervention'] >= 0.2 else 0, axis=1)
            farm_gdf_copy['habitat_conversion'] = farm_gdf_copy.apply(
                lambda row: row['habitat_conversion'] if row['type'] == 'ag_plot' and row[
                    'habitat_conversion'] >= 0.2 else 0, axis=1)

            farm_gdf_copy.to_file(result_file, driver='GeoJSON')

            # Save parameter metadata
            config_file = os.path.join(RESULTS_DIR, f"{param_name}_{value}_config.txt")
            with open(config_file, 'w') as f:
                f.write(str(current_params))


def load_and_visualize_results():
    """
    Load all results from CSV files and visualize them.
    """
    results = []
    images = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".geojson"):
            param, value = filename.replace(".geojson", "").rsplit("_", 1)
            value = float(value)

            filepath = os.path.join(RESULTS_DIR, filename)

            # Load result
            farm_gdf = gpd.read_file(filepath)
            avg_margin_intervention = farm_gdf['margin_intervention'].mean()
            avg_habitat_conversion = farm_gdf['habitat_conversion'].mean()

            results.append({
                'param': param,
                'value': value,
                'avg_margin_intervention': avg_margin_intervention,
                'avg_habitat_conversion': avg_habitat_conversion
            })

            # Visualize optimized farm and save as image
            image_path = os.path.join(RESULTS_DIR, f"{param}_{value}.png")
            visualize_optimized_farm(filepath, image_path, exp=param + "_" + str(value))
            images.append(Image.open(image_path))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Visualize results summary
    for param in results_df['param'].unique():
        param_df = results_df[results_df['param'] == param]
        plt.figure(figsize=(10, 6))
        plt.plot(param_df['value'], param_df['avg_margin_intervention'], label='Average Margin Intervention',
                 marker='o')
        plt.plot(param_df['value'], param_df['avg_habitat_conversion'], label='Average Habitat Conversion', marker='o')
        plt.xlabel(f"{param} value")
        plt.ylabel("Average Intervention")
        plt.title(f"Impact of {param} on Interventions")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f"{param}_avg.png"))

    # Create GIF from visualizations
    gif_path = os.path.join(RESULTS_DIR, "optimization_visualizations.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, disposal=2, loop=0)


def csv_to_geojson():
    filepath = os.path.join(cfg.data_dir, "crop_inventory", "farms", "farm_5_red.geojson")
    farm_gdf = gpd.read_file(filepath)

    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".csv"):
            param, value = filename.replace(".csv", "").rsplit("_", 1)
            filepath = os.path.join(RESULTS_DIR, filename)
            temp_gdf = pd.read_csv(filepath)

            farm_gdf["margin_intervention"] = pd.Series([float(a) for a in list(temp_gdf['margin_intervention'])])
            farm_gdf["habitat_conversion"] = pd.Series([float(a) for a in list(temp_gdf['habitat_conversion'])])

            geojson_path = os.path.join(RESULTS_DIR, f"{param}_{value}.geojson")

            farm_gdf.to_file(geojson_path, driver='GeoJSON')


# Main function to run parameter sweep
if __name__ == '__main__':
    cfg = Config()

    # Directory to save optimization results
    RESULTS_DIR = os.path.join(cfg.data_dir, "crop_inventory", "farms", "optimization_results")
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    filepath = os.path.join(cfg.data_dir, "crop_inventory", "farms", "farm_5_red.geojson")

    # Load farm data
    farm_gdf = load_farm_geojson(filepath)

    # Base parameters to start sweeping from
    base_params = cfg.params

    # Sweep parameters and save results
    # sweep_parameters(farm_gdf, base_params)

    # Load and visualize the results

    load_and_visualize_results()

    #csv_to_geojson()