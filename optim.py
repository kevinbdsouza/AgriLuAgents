import geopandas as gpd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from config import Config
import os
from scipy.optimize import Bounds, LinearConstraint
from shapely.geometry import Point
import random
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd

# Load GeoJSON file
def load_farm_geojson(filepath):
    farm_gdf = gpd.read_file(filepath)
    return farm_gdf


# Calculate distance between plots
def calculate_distance(geometry1, geometry2):
    return geometry1.distance(geometry2)


# Generate random points within a polygon
def generate_random_points_within_polygon(polygon, n_points):
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    while len(points) < n_points:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            points.append(random_point)
    return points


# Calculate the cumulative effect from margin intervention at the boundary
def calculate_margin_effect_within_polygon(polygon, alpha, beta, gamma, T):
    n_points = 10  # Number of points proportional to margin fraction
    points = generate_random_points_within_polygon(polygon, n_points)
    boundary = polygon.boundary

    cumulative_effect = []
    for point in points:
        cum = []
        d = point.distance(boundary)
        for t in range(1, T + 1):
            cum.append(alpha * np.exp(-beta * d) * (1 - np.exp(-gamma * t)))
        cumulative_effect.append(cum)

    return np.sum(cumulative_effect, axis=0)  # Average effect over the points


# Pollination Service Function
def pollination_service(fraction, polygon, d, alpha, beta, gamma, T, mode):
    if mode == 'margin':
        return fraction * calculate_margin_effect_within_polygon(polygon, alpha, beta, gamma, T)
    elif mode == 'habitat':
        poll = []
        for t in range(1, T + 1):
            poll.append(fraction * alpha * np.exp(-beta * d) * (1 - np.exp(-gamma * t)))
        return poll
    else:
        raise ValueError("Invalid mode. Use 'margin' or 'habitat'.")


# Pest Control Service Function
def pest_control_service(fraction, polygon, d, delta, epsilon, zeta, T, mode):
    if mode == 'margin':
        return fraction * calculate_margin_effect_within_polygon(polygon, delta, epsilon, zeta, T)
    elif mode == 'habitat':
        pest = []
        for t in range(1, T + 1):
            pest.append(fraction * delta * np.exp(-epsilon * d) * (1 - np.exp(-zeta * t)))
        return pest
    else:
        raise ValueError("Invalid mode. Use 'margin' or 'habitat'.")


# Combined Yield Impact
def combined_yield_impact(y_base, pollination_services, pest_control_services):
    return y_base * (1 + np.sum(pollination_services, axis=0) + np.sum(pest_control_services, axis=0))


# Define objective function for optimization
def objective_function(variables, farm_gdf, params):
    npv = 0
    margin_variables = variables[:len(farm_gdf)]
    habitat_variables = variables[len(farm_gdf):]

    for idx, row in farm_gdf.iterrows():
        # Area of the plot
        #plot_area = row.geometry.area
        plot_area = 1

        if row['type'] == 'ag_plot':
            # Yield Calculation
            pollination_services = []
            pest_control_services = []
            crop_params = params['crops'][row['label']]

            # Margin interventions affecting the yield of the plot
            pollination_services.append(pollination_service(
                margin_variables[idx], row.geometry, 0, crop_params['margin']['alpha'],
                crop_params['margin']['beta'], crop_params['margin']['gamma'], params['t'], mode='margin'))
            pest_control_services.append(pest_control_service(
                margin_variables[idx], row.geometry, 0, crop_params['margin']['delta'],
                crop_params['margin']['epsilon'], crop_params['margin']['zeta'], params['t'], mode='margin'))

            for other_idx, other_row in farm_gdf.iterrows():
                if idx != other_idx and other_row["type"] != "farm":
                    d = calculate_distance(row.geometry.centroid, other_row.geometry.centroid)
                    if d <= 1000:  # Consider only plots within 500m distance
                        if other_row['type'] == 'hab_plots':
                            habitat_value = 1  # Permanent habitat value for existing natural habitats
                        elif other_row['type'] == 'ag_plot':
                            habitat_value = habitat_variables[
                                other_idx]  # Habitat conversion value for agricultural plots
                        else:
                            continue

                        crop_habitat_params = crop_params['habitat']

                        pollination_services.append(pollination_service(
                            habitat_value, row.geometry, d, crop_habitat_params['alpha'],
                            crop_habitat_params['beta'], crop_habitat_params['gamma'], params['t'], mode='habitat'))
                        pest_control_services.append(pest_control_service(
                            habitat_value, row.geometry, d, crop_habitat_params['delta'],
                            crop_habitat_params['epsilon'], crop_habitat_params['zeta'], params['t'], mode='habitat'))

            y_impact = combined_yield_impact(row['yield'], pollination_services, pest_control_services)

            # Implementation Costs
            implementation_costs = plot_area*(margin_variables[idx] * params['costs']['margin']['implementation'] + \
                                   habitat_variables[idx] * params['costs']['habitat']['implementation'])
            npv -= implementation_costs

            # Maintenance Costs
            maintenance_costs = plot_area*(params['costs']['margin']['maintenance'] * margin_variables[idx] + \
                                params['costs']['habitat']['maintenance'] * habitat_variables[idx])

            for t in range(1, params['t'] + 1):
                npv += y_impact[t - 1] * crop_params['p_c'] * plot_area * (1 + params['r']) ** -t
                # opportunity_costs
                npv -= habitat_variables[idx] * crop_params['p_c'] * plot_area * row['yield'] * (1 + params['r']) ** -t
                npv -= maintenance_costs * (1 + params['r']) ** -t

        elif row['type'] == 'hab_plots':
            # Maintenance Costs for existing habitats (farmer pays only 10%)
            maintenance_costs = 0.1 * params['costs']['habitat']['maintenance'] * plot_area
            for t in range(1, params['t'] + 1):
                npv -= maintenance_costs * (1 + params['r']) ** -t

    return -npv  # Maximization turned into minimization for optimization


# Optimize the farm configuration
def optimize_farm(farm_gdf, params):
    initial_variables = np.zeros(
        2 * len(farm_gdf))  # Initial guess for intervention proportions (margin + habitat + habitat type)
    bounds = [(0, 1) for _ in range(2 * len(farm_gdf))]

    # Add constraints for habitat plots to ensure habitat_variables are fixed to 1
    constraints = []
    for idx, row in farm_gdf.iterrows():
        # Ensure that either margin_variable or habitat_variable is 0 when the other is non-zero
        constraints.append(
            {'type': 'eq', 'fun': lambda variables, idy=idx: variables[idy] * variables[len(farm_gdf) + idy]})

    result = minimize(objective_function, initial_variables, args=(farm_gdf, params),
                      bounds=bounds, constraints=constraints, method='trust-constr',
                      options={'maxiter': 20, 'disp': True})
    return result


# Apply threshold and save optimized farm_gdf
def apply_threshold_and_save(farm_gdf, threshold=0.2):
    farm_gdf['margin_intervention'] = farm_gdf.apply(
        lambda row: row['margin_intervention'] if row['type'] == 'ag_plot' and row[
            'margin_intervention'] >= threshold else 0, axis=1)
    farm_gdf['habitat_conversion'] = farm_gdf.apply(
        lambda row: row['habitat_conversion'] if row['type'] == 'ag_plot' and row[
            'habitat_conversion'] >= threshold else 0, axis=1)
    farm_gdf.to_file(os.path.join(cfg.data_dir, "crop_inventory", "farms", "farm_5_optim.geojson"),
                     driver="GeoJSON")
    return farm_gdf


# Visualization function
def visualize_optimized_farm(filepath, image_path, exp):
    farm_gdf = gpd.read_file(filepath)

    fig, ax = plt.subplots(figsize=(10, 10))

    farm_gdf.loc[farm_gdf['margin_intervention'] > 1, 'margin_intervention'] = 1
    farm_gdf.loc[farm_gdf['habitat_conversion'] > 1, 'habitat_conversion'] = 1

    # Plot margin interventions
    farm_gdf.boundary.plot(ax=ax, color='grey')
    margin_gdf = farm_gdf[farm_gdf['margin_intervention'] > 0]
    margin_gdf.plot(ax=ax, color='red', alpha=margin_gdf['margin_intervention'], aspect=1)

    # Plot habitat conversions
    habitat_gdf = farm_gdf[farm_gdf['habitat_conversion'] > 0]
    habitat_gdf.plot(ax=ax, color='green', alpha=habitat_gdf['habitat_conversion'], aspect=1)

    # Plot existing habitats
    hab_plots_gdf = farm_gdf[farm_gdf['type'] == 'hab_plots']
    hab_plots_gdf.plot(ax=ax, color='blue', alpha=0.5, aspect=1)

    # Create legend
    patches = [
        mpatches.Patch(color='red', label='Margin Interventions'),
        mpatches.Patch(color='green', label='Habitat Conversions'),
        mpatches.Patch(color='blue', label='Existing Habitats')
    ]
    plt.legend(handles=patches)
    plt.title(exp)
    #plt.show()
    plt.savefig(image_path)


# Main function to run the optimization framework
def main(filepath):
    # Load farm data
    farm_gdf = load_farm_geojson(filepath)

    params = cfg.params

    # Run optimization
    result = optimize_farm(farm_gdf, params)

    # Update farm_gdf with optimized results
    farm_gdf['margin_intervention'] = result.x[:len(farm_gdf)]
    farm_gdf['habitat_conversion'] = result.x[len(farm_gdf):]

    # Apply threshold and save
    farm_optim = apply_threshold_and_save(farm_gdf)

    return farm_optim


if __name__ == '__main__':
    cfg = Config()
    filepath = os.path.join(cfg.data_dir, "crop_inventory", "farms", "farm_5_red.geojson")

    optimized_farm_gdf = main(filepath)
    visualize_optimized_farm(os.path.join(cfg.data_dir, "crop_inventory", "farms", "farm_5_optim.geojson"),
                             os.path.join(cfg.data_dir, "crop_inventory", "farms", "farm_5_optim.png"), "farm_5_optim")

