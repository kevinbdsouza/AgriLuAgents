from rasterio.features import shapes
import geopandas as gp
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import TwoSlopeNorm
import os
from config import Config
import json
from netCDF4 import Dataset
import math
from json import JSONDecoder
from functools import partial
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
import pyproj
from functools import partial
from shapely.ops import transform, unary_union
import logging
import random

# Fix for PROJ database context error
# Set PROJ_LIB environment variable to help pyproj find the database
if 'PROJ_LIB' not in os.environ:
    # Try to find proj.db in common locations
    possible_paths = [
        '/usr/share/proj',  # Linux
        '/usr/local/share/proj',  # macOS with Homebrew
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'proj'),  # Project data directory
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'proj.db')):
            os.environ['PROJ_LIB'] = path
            logging.info(f"Set PROJ_LIB to {path}")
            break
    
    if 'PROJ_LIB' not in os.environ:
        logging.warning("Could not find proj.db. Some geospatial operations may fail.")
        # Create a minimal proj.db if needed
        try:
            from pyproj.datadir import get_data_dir
            os.environ['PROJ_LIB'] = get_data_dir()
            logging.info(f"Set PROJ_LIB to {get_data_dir()}")
        except Exception as e:
            logging.error(f"Failed to set PROJ_LIB: {e}")

class ncload():
    def __init__(self, fname):
        """
        Load netcdf file.

        This function will load a netcdf file using Netcdf4 standard (absolute address or relative address supported).

        Instance attributes:
        -------
        self.nc             : current netcdf file
        """
        if fname[-3:] == '.nc':
            ncfile = fname
        else:
            ncfile = fname + '.nc'
        self.nc = Dataset(ncfile, 'r')
        self.name = ncfile

    def _getvar(self, vname):
        """
        Choose a variable in the nc file.

        This function will read the variable named 'varname'.

        Returns
        -------
        var : netCDF.Variable object
        """
        return self.nc.variables[vname]

    def get(self, *vnames):
        """
        Choose a variable or a list of variales.

        Parameters
        ----------
        varnames : list or string
               Names of variables. Could be string (one variable) or list (several variables)

        Returns
        -------
        varlist: netCDF4.Variable objects
        """
        varlist = [self._getvar(vname) for vname in vnames]
        self._refvar = varlist[0]
        return varlist[0] if len(vnames) == 1 else varlist

    def close(self):
        self.nc.close()

# Fix for PROJ version mismatch
# This is a workaround for the error: "DATABASE.LAYOUT.VERSION.MINOR = 2 whereas a number >= 4 is expected"
def fix_proj_version_mismatch():
    try:
        # Try to create a CRS to see if we have the version mismatch error
        pyproj.CRS.from_epsg(4326)
        logging.info("PROJ database version is compatible")
        return True
    except Exception as e:
        if "DATABASE.LAYOUT.VERSION.MINOR" in str(e):
            logging.warning("PROJ database version mismatch detected. Applying workaround...")
            
            # Create a custom CRS for EPSG:3347 (NAD83 / Statistics Canada Lambert)
            # This is a workaround for the version mismatch
            custom_crs = pyproj.CRS.from_proj4(
                "+proj=lcc +lat_1=49 +lat_2=77 +lat_0=49 +lon_0=-95 +x_0=0 +y_0=0 "
                "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
            )
            
            # Register the custom CRS with pyproj
            pyproj.CRS.register_crs(custom_crs, "EPSG:3347")
            logging.info("Registered custom CRS for EPSG:3347")
            return True
        else:
            logging.error(f"Unexpected PROJ error: {e}")
            return False

# Apply the fix
fix_proj_version_mismatch()

def basemap_plot(cfg, var, plot_name, plot_title, lats, lons, cmap, cbar_ticks, mode):
    fig = plt.figure()

    if mode == "canada":
        m = Basemap(llcrnrlat=44, llcrnrlon=-148, urcrnrlat=75, urcrnrlon=-50, resolution='l', projection='merc',
                    lat_0=60, lon_0=-99)
    elif mode == "globe":
        m = Basemap(resolution='l', projection='cyl')
    lon, lat = np.meshgrid(lons, lats)
    xi, yi = m(lon, lat)

    norm = TwoSlopeNorm(vcenter=0.5)
    cs = m.pcolor(xi, yi, np.squeeze(var), cmap=cmap, norm=norm)
    m.drawcoastlines()
    m.drawcountries()

    cbar = m.colorbar(cs, location='bottom', pad="10%")
    if cbar_ticks:
        cbar.set_ticks(cbar_ticks)

    plt.title(plot_title)

    plt.savefig(os.path.join(cfg.plot_dir, plot_name + "_" + mode + ".png"))
    plt.close()


def get_max_id_for_prop(prop, name):
    max_id = None
    major = prop[name]
    if len(major) != 0:
        max_n = max(major, key=major.count)
        max_id = [i for i, v in enumerate(major) if v == max_n][0]
    return max_id


def normalize_eco_con(eco_con, lats, lons):
    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])
    max_dist = np.linalg.norm(red - blue)

    n_lats = len(lats)
    n_lons = len(lons)
    norm = np.zeros((n_lats, n_lons))
    for lat_i in range(n_lats):
        for lon_j in range(n_lons):
            samp = np.array(eco_con[:-1, lat_i, lon_j])
            dist = np.linalg.norm(blue - samp)
            norm[lat_i, lon_j] = np.round(dist / max_dist, 6)
    return norm


def convert_array_to_geojson(array, lats, lons):
    data_dict = {}
    data_dict["type"] = "FeatureCollection"
    data_dict["features"] = []
    ct = 1
    n_lats = len(lats)
    n_lons = len(lons)
    for lat_i in range(n_lats):
        for lon_j in range(n_lons):
            if lat_i == n_lats - 1 or lon_j == n_lons - 1:
                break

            if lons[lon_j] > -92 or lons[lon_j] < -122 or lats[lat_i] < 49 or lats[lat_i] > 62:
                continue

            coords = [[float(lons[lon_j].data), float(lats[lat_i].data)],
                      [float(lons[lon_j + 1].data), float(lats[lat_i].data)],
                      [float(lons[lon_j + 1].data), float(lats[lat_i + 1].data)],
                      [float(lons[lon_j].data), float(lats[lat_i + 1].data)],
                      [float(lons[lon_j].data), float(lats[lat_i].data)]]
            v = np.round(((array[lat_i, lon_j] + array[lat_i, lon_j + 1] + array[lat_i + 1, lon_j] +
                           array[lat_i + 1, lon_j + 1]) / 4), 4)

            feature = {}
            feature["type"] = "Feature"
            feature["id"] = ct
            ct += 1
            feature["geometry"] = {}
            feature["geometry"]["type"] = "Polygon"
            feature["geometry"]["coordinates"] = [coords]
            feature["properties"] = {}
            feature["properties"]["eco_con"] = v

            data_dict["features"].append(feature)
    return data_dict


def tiff_to_polygons(tiff_file):
    meta_data = rasterio.open(tiff_file).meta
    crs = str(meta_data['crs'])

    mask = None
    with rasterio.open(tiff_file) as src:
        image = src.read(1)
        geoms = []
        for i, (s, v) in enumerate(shapes(image, mask=mask, transform=meta_data['transform'])):
            geoms.append({'properties': {'eco_con': v}, 'geometry': s})

    polygons = gp.GeoDataFrame.from_features(geoms, crs=crs)
    dst_crs = 'epsg:4326'
    polygons['geometry'] = polygons['geometry'].to_crs({'init': dst_crs})
    return polygons


def transform_to_gpd(compiled_data, cfg):
    t_data = {}
    t_data["type"] = "FeatureCollection"
    t_data["features"] = []
    for i, feature in enumerate(compiled_data["features"]):
        feat = {}
        feat["type"] = "Feature"
        feat["id"] = feature["id"]
        feat["geometry"] = feature["geometry"]
        feat["properties"] = {}
        for prop in feature["properties"]:
            if prop in cfg.data_keys:
                for k in cfg.data_keys[prop]:
                    if type(feature["properties"][prop][k]) == list:
                        feat["properties"][k] = None
                    else:
                        feat["properties"][k] = feature["properties"][prop][k]
            else:
                feat["properties"][prop] = feature["properties"][prop]

        t_data["features"].append(feat)

    with open(cfg.compiled_gpd_json, 'w') as f:
        json.dump(t_data, f)


def remove_nones(compiled_data, cfg):
    for feature in compiled_data["features"]:
        for prop in feature["properties"]:
            prop_val = feature["properties"][prop]
            if type(prop_val) == str:
                continue
            if prop_val is None or math.isnan(prop_val):
                feature["properties"][prop] = 0

    with open(cfg.compiled_gpd_nn_json, 'w') as f:
        json.dump(compiled_data, f)


def visuzalize_geojson(compiled_data, cfg):
    canada_shp = gp.read_file(os.path.join(cfg.data_dir, "canada_shp", "lpr_000b21a_e.shp"))
    canada_shp = canada_shp.to_crs("epsg:4326")

    for k in cfg.all_props:
        if k not in ["WHAF_2015_CLASS_EN"]:
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        canada_shp.plot(ax=ax, facecolor="oldlace")
        compiled_data.plot(ax=ax, column=k, legend=True)
        ax.set_title(k)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1, 1.2))
        plt.savefig(os.path.join(cfg.plot_dir, k + ".png"), )


def json_parse(fileobj, decoder=JSONDecoder(), buffersize=2048):
    buffer = ''
    for chunk in iter(partial(fileobj.read, buffersize), ''):
        buffer += chunk
        while buffer:
            try:
                result, index = decoder.raw_decode(buffer)
                breakpoint()
                yield result
                buffer = buffer[index:].lstrip()
            except ValueError:
                breakpoint()
                # Not enough data to decode, read more
                break


def fix_polygons(compiled_data, cfg):
    for feature in compiled_data["features"]:
        coords = feature["geometry"]["coordinates"]
        tmp = coords[0][2]
        coords[0][2] = coords[0][3]
        coords[0][3] = tmp

    with open(os.path.join(cfg.data_dir, 'compiled_reduced.geojson'), 'w') as f:
        json.dump(compiled_data, f)

    transform_to_gpd(compiled_data, cfg)
    with open(cfg.compiled_gpd_json, 'r') as j:
        compiled_data = json.loads(j.read())
    remove_nones(compiled_data, cfg)
    pass


def clean_geo_data(compiled_data, cfg):
    drop_list = []
    for i, feature in enumerate(compiled_data["features"]):
        if feature["properties"]["WHAF_2015_CLASS_EN"] not in ["High", "Low", "Moderate", "Very High", "Very Low"]:
            drop_list.append(i)

    compiled_data["features"] = [feat for i, feat in enumerate(compiled_data["features"]) if i not in drop_list]

    with open(os.path.join(cfg.data_dir, 'compiled_gpd_no_nones2.geojson'), 'w') as f:
        json.dump(compiled_data, f)
    pass


def compute_nh_nbs(data, n=1):
    for id1, poly1 in data.iterrows():
        whaf1 = poly1["Wildlife Habitat Capacity"]

        data.at[id1, str(n) + "h_neighbs"] = []
        nbs = poly1[str(n - 1) + "h_neighbs"]
        for id2, poly2 in data.iterrows():
            ohnbs = poly2["0h_neighbs"]
            intersec = list(set(ohnbs).intersection(nbs))
            ln_in = len(intersec)
            if ln_in > 0:
                data.at[id1, str(n) + "h_neighbs"].append(id2)
                if whaf1 == "High" or whaf1 == "Very High":
                    data.at[id2, "n_" + str(n) + "h_neighbs"] = data.loc[id2, "n_" + str(n) + "h_neighbs"] + ln_in
    return data


def extract_radius(center_id, num_hops):
    json_path = os.path.join(cfg.data_dir, "sel_pos_4.geojson")

    # Load the GeoJSON data
    with open(json_path, "r") as f:
        data = json.loads(f.read())

    # Find the center feature
    for feature in data['features']:
        if feature['properties']['id'] == center_id:
            break

    ids_to_include = [center_id]
    hops = ["central"]
    for hop in range(num_hops + 1):
        if feature['properties'][f'{hop}h_neighbs'] is not None:
            for nb in feature['properties'][f'{hop}h_neighbs']:
                ids_to_include.append(nb)
                hops.append(hop)

    # Create the new feature collection
    new_features = []
    prop_names = ['Ecological Connectivity Pither et al.', 'Wildlife Habitat Capacity', 'id']
    for feature in data['features']:
        if feature['properties']['id'] in ids_to_include:
            hop = [hops[i] for i, idx in enumerate(ids_to_include) if feature['properties']['id'] == idx][0]

            props = {}
            for prop in feature['properties']:
                if "_YLD" in prop or prop in prop_names:
                    props[prop] = feature['properties'][prop]
            props['nb_hop'] = hop

            new_feature = {
                'type': 'Feature',
                'properties': props,
                'geometry': feature['geometry']
            }
            new_features.append(new_feature)

    # Create the new GeoJSON
    new_geojson = {
        'type': 'FeatureCollection',
        'name': data['name'],
        'crs': data['crs'],
        'features': new_features
    }

    json_out = os.path.join(cfg.data_dir, "center_temp_" + str(center_id) + ".geojson")

    with open(json_out, "w") as f:
        json.dump(new_geojson, f)


def plot_shp():
    #shp = gp.read_file(os.path.join(cfg.data_dir, "temp", "lcsd000b16a_e", "lcsd000b16a_e.shp"))
    # shp.plot(ax=ax, facecolor="oldlace", edgecolor="dimgray")

    gdf = gp.read_file(os.path.join(cfg.data_dir, "crop_inventory", "plots.geojson"))
    list = ["Barley", "Broadleaf", "Canola/rapeseed", "Corn",
            "Grassland", "Oats", "Soybeans", "Spring wheat", "Urban/developed", "Water"]
    colors_list = ["red", "green", "blue", "yellow", "cyan", "magenta", "lightcoral", "grey", "orange", "goldenrod",
                   "olive", "lawngreen", "darkviolet"]

    for idx, _ in gdf.iterrows():
        if gdf.loc[idx, "label"] not in list:
            gdf.loc[idx, "label"] = "others"

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, column="label", legend=True, aspect=1)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.1, 1.1))
    plt.savefig(os.path.join(cfg.plot_dir, "agri2.png"), )
    plt.close()
    pass

def test_func():
    # Load the raster data
    tif_path = os.path.join(cfg.data_dir, "crop_inventory", "cropped.tif")
    csv_path = os.path.join(cfg.data_dir, "crop_inventory", "aci_crop_classifications_iac_classifications_des_cultures.csv")

    with rasterio.open(tif_path) as src:
        raster_data_band1 = src.read(1)
        raster_data_band2 = src.read(2)
        raster_data_band3 = src.read(3)
        transform = src.transform
        crs = src.crs

    # Load the CSV data with labels
    csv_data = pd.read_csv(csv_path, encoding='ISO-8859-1')
    csv_data.rename(columns={"Code": "class_code", "Label": "label"}, inplace=True)\

    raster_data = np.stack([raster_data_band1, raster_data_band2, raster_data_band3], axis=-1)

    # Extract unique colors for segmentation
    unique_colors = np.unique(raster_data.reshape(-1, raster_data.shape[2]), axis=0)

    # Separate the farm boundaries and plot areas based on color codes
    farm_color = [0, 0, 0]  # Assuming farm boundaries are marked with white color (255, 255, 255)
    plot_colors = [color for color in unique_colors if not np.array_equal(color, farm_color)]

    # Initialize lists to hold farm and plot geometries
    farms = []
    plots = []

    # Create masks for farms and plots, and generate polygons using contours
    for color in plot_colors:
        mask = np.all(raster_data == color, axis=-1)
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            poly_coords = [(transform * (coord[1], coord[0])) for coord in contour]
            plot_polygon = Polygon(poly_coords)
            if plot_polygon.is_valid:
                # Find the label for this plot color based on RGB values
                label_row = csv_data[
                    (csv_data['Red'] == color[0]) & (csv_data['Green'] == color[1]) & (csv_data['Blue'] == color[2])
                    ]
                label = label_row['label'].values[0] if not label_row.empty else "Unknown"
                plots.append({"geometry": plot_polygon, "label": label})

    plot_gdf = gpd.GeoDataFrame(plots, crs=crs)

    # Combine farms and plots into a single GeoJSON-like structure
    result_geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Add plot features
    for idx, plot in plot_gdf.iterrows():
        plot_feature = {
            "type": "Feature",
            "properties": {
                "label": plot["label"]
            },
            "geometry": plot["geometry"].__geo_interface__
        }
        result_geojson["features"].append(plot_feature)

    # Save the result to a GeoJSON file
    output_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "plots.geojson")
    with open(output_geojson_path, "w") as geojson_file:
        json.dump(result_geojson, geojson_file)

    print("here")

# --- Land Quality Assignment based on Yield --- 

# Conversion factors needed for yield calculation 
ACRE_TO_HECTARE = 0.404686
BUSHELS_TO_TONNES = {
    'corn': 0.0254,
    'soy': 0.0272,     
    'wheat': 0.0272,
    'oat': 0.0145
}

def assign_land_quality_based_on_yields(land_manager_data, parcel_ids, config):
    """Calculates total potential yield per parcel, determines yield percentiles, 
       and assigns a land quality label and description based on these percentiles.
       Uses the exact logic previously in main.py lines 35-90.

    Args:
        land_manager_data (pd.DataFrame): DataFrame containing parcel data, including yield columns.
        parcel_ids (list): List of parcel IDs to process.
        config (Config): The configuration object.

    Returns:
        tuple: A tuple containing:
            - parcel_yield_details (dict): Dictionary mapping parcel_id to its specific base yields (t/ha).
            - parcel_assigned_qualities (dict): Dictionary mapping parcel_id to its assigned quality label and description.
    """
    # --- Start of code block moved from main.py (lines 35-90) --- 
    logging.info("Pre-calculating total potential yields and assigning land quality...")
    all_parcel_yields = {}
    parcel_yield_details = {} # To store intermediate yields for quality assignment

    for parcel_id in parcel_ids:
        parcel_data = land_manager_data.loc[parcel_id]
        parcel_specific_base_yields_t_ha = {}
        total_yield_t_ha = 0

        # Use function argument config
        available_crops = config.crops_data["crop_list"]

        for crop in available_crops:
            parcel_yield_col = f"{crop}_yield"
            yield_bu_ac = parcel_data.get(parcel_yield_col)
            if yield_bu_ac is not None:
                yield_t_ha = yield_bu_ac
            else:
                yield_t_ha = 0 

            parcel_specific_base_yields_t_ha[crop] = yield_t_ha
            total_yield_t_ha += yield_t_ha

        all_parcel_yields[parcel_id] = total_yield_t_ha
        parcel_yield_details[parcel_id] = parcel_specific_base_yields_t_ha 

    # --- Calculate Percentiles --- 
    yield_values = list(all_parcel_yields.values())
    p33 = np.nanpercentile(yield_values, 33.33)
    p66 = np.nanpercentile(yield_values, 66.67)
    logging.info(f"Yield Percentiles: 33rd={p33:.2f} t/ha, 66th={p66:.2f} t/ha")

    # --- Assign Quality Label and Description --- 
    parcel_assigned_qualities = {}
    config_land_quality = config.land_quality
    
    for parcel_id, total_yield in all_parcel_yields.items():
        quality_label = ""
        quality_description = ""
        if total_yield <= p33:
            quality_label = "poor"
        elif total_yield <= p66:
            quality_label = "satisfactory"
        else:
            quality_label = "good"

        # Get description
        descriptions = config_land_quality.get(quality_label) 
        if isinstance(descriptions, list) and descriptions:
            quality_description = random.choice(descriptions)
        elif isinstance(descriptions, str):
            quality_description = descriptions
        
        parcel_assigned_qualities[parcel_id] = {
            'label': quality_label,
            'description': quality_description
        }
        logging.debug(f"Parcel {parcel_id}: Total Yield={total_yield:.2f} -> Quality='{quality_label}', Desc='{quality_description}'")
    return parcel_yield_details, parcel_assigned_qualities


def save_results_to_geojson(land_manager_data: gpd.GeoDataFrame, agent_map: dict, output_dir: str, filename: str):
    """Saves simulation results to a GeoJSON file.

    Args:
        land_manager_data: Original GeoDataFrame containing parcel geometries.
        agent_map: Dictionary mapping parcel_id to LandManagerAgent instances.
        output_dir: Directory to save the output file.
        filename: Name of the output GeoJSON file (e.g., 'simulation_results.geojson').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, filename) 
    logging.info(f"Preparing results for saving to {output_file}...")

    # Start with the original GeoDataFrame containing geometries
    if not land_manager_data.empty:
        # Select only the geometry column and index to start
        results_gdf = land_manager_data[['geometry']].copy()

        # Prepare lists for agent details and decision history
        final_details_list = []
        decision_history_list = []
        
        for parcel_id in results_gdf.index: # Iterate using the GeoDataFrame index
            agent = agent_map.get(parcel_id)
            if agent:
                # Append the entire details dictionary
                final_details_list.append(agent.manager_details)
                decision_history_list.append(agent.decision_dict)
            else:
                # Append empty dict/None if no agent data exists for a parcel
                final_details_list.append({}) 
                decision_history_list.append(None)
                logging.warning(f"No agent found for parcel_id {parcel_id} when saving results.")

        # Add decision history as a JSON string column
        try:
            results_gdf['decision_history'] = [json.dumps(h) if h is not None else None for h in decision_history_list]
        except TypeError as e:
            logging.error(f"Could not serialize decision_history to JSON: {e}. Storing as string representation.")
            results_gdf['decision_history'] = [str(h) if h is not None else None for h in decision_history_list]

        # Convert the list of final details dictionaries into a DataFrame
        details_df = pd.DataFrame(final_details_list, index=results_gdf.index)
        
        # Drop geometry column from details_df if it accidentally exists
        if 'geometry' in details_df.columns:
             details_df = details_df.drop(columns=['geometry'])
             
        # Join the details columns into the GeoDataFrame
        results_gdf = results_gdf.join(details_df)

        # Save to GeoJSON
        results_gdf.to_file(output_file, driver='GeoJSON')
        logging.info(f"Saved final land manager states and decision history to {output_file} (GeoJSON format, properties as columns)")

    else:
        logging.info("No land manager data found, skipping results saving.")


if __name__ == "__main__":
    cfg = Config()
    #nc_file = ncload(cfg.ecological_connectivity_nc)
    #eco_con = nc_file.get("eco_con")
    #lats = nc_file.get("lat")
    #lons = nc_file.get("lon")

    # array = np.load(os.path.join(cfg.data_dir, "ecological_connectivity", "norm.npy"))
    # data_dict = convert_array_to_geojson(array, lats, lons)
    # with open(os.path.join(cfg.data_dir, "ecological_connectivity", 'eco_con.geojson'), 'w') as f:
    #    json.dump(data_dict, f)

    #eco_con = gp.read_file(os.path.join(cfg.data_dir, "ecological_connectivity", 'eco_con.geojson'))
    #fig, ax = plt.subplots(figsize=(8, 8))
    #eco_con.plot(ax=ax, legend=True)
    #plt.show()
    #print("here")

    # extract_radius(570, 1) # 0, 10, 570 (4, 14, 24)

    # plot_shp()

    # test_func()


