import os.path
import geopandas as gp
from utils import Config, fix_proj_version_mismatch
import logging


class Data:
    def __init__(self):
        self.cfg = Config()
        self.land_manager_data = self.load_land_manager_data()

    def load_land_manager_data(self):
        geojson_path = os.path.join('data', 'quadrants_landuse.geojson')
        if not os.path.exists(geojson_path):
             raise FileNotFoundError(f"GeoJSON file not found at: {geojson_path}")
        
        try:
            land_manager_data = gp.read_file(geojson_path)
            logging.info("Successfully loaded land manager data with default CRS")
            return land_manager_data
        except Exception as e:
            logging.error(f"Error loading land manager data: {e}")
            raise


if __name__ == "__main__":
    pass 
