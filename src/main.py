import numpy as np
import yaml
import logging
import os 
import pandas as pd
import random 
import json 
import geopandas as gpd
from config import Config 
from environment import Environment 
from super import SuperAgent
from utils import save_results_to_geojson

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


def main():
    """Main simulation function."""
    logging.info("=== Starting Simulation ===")

    # 1. Initialize Configuration, SuperAgent, and Environment
    config = Config() # Load configuration
    super_agent = SuperAgent(config) # Initializes Data and loads land_manager_dat
    parcel_ids = super_agent.parcel_ids

    environment = Environment(config, parcel_ids=parcel_ids) 
    logging.info(f"Configuration, SuperAgent, and Environment (for {len(parcel_ids)} parcels) initialized.")

    # Simulation parameters from config
    start_year = 1
    num_years = config.simulation_params["num_years"]
    base_year = config.simulation_params["base_year"]

    # --- Simulation Loop ---
    logging.info(f"Starting simulation from year {start_year} for {num_years} years.")
    for sim_year in range(start_year, start_year + num_years):
        current_real_year = base_year + sim_year - start_year
        logging.info(f"--- Starting Year {sim_year} (Real Year: {current_real_year}) ---")

        # 1. Update Environment State for the current year (Steps global market prices and all parcel climates)
        if sim_year > start_year: 
            environment.step()  

        # Iterate through each land parcel from the loaded data
        num_parcels = len(super_agent.land_manager_data)
        logging.info(f"Processing {num_parcels} land parcels for year {sim_year}...")
        processed_agents_count = 0
        
        # Use the parcel_ids list 
        for parcel_id in parcel_ids: 
            parcel_data = super_agent.land_manager_data.loc[parcel_id].copy() # Use copy to modify
            
            if parcel_id in super_agent.agent_map:
                current_agent = super_agent.agent_map[parcel_id]
                current_agent.prepare_for_next_year(current_real_year, sim_year)
                logging.debug(f"Retrieved and updated LandManagerAgent for parcel {parcel_id}")
            else:
                # Pass parcel_data (now including assigned quality) and specific yields
                agent = super_agent.create_land_manager_agent(
                    sim_year=sim_year, 
                    parcel_id=parcel_id, 
                    parcel_data=parcel_data, # Pass the modified data 
                    config=config
                )
                super_agent.agent_map[parcel_id] = agent
                current_agent = agent
                logging.debug(f"Created LandManagerAgent for parcel {parcel_id}") 
            
            try:
                current_environment_state = environment.get_current_state(parcel_id)
                current_agent.in_context_decision_making(current_environment_state)
                processed_agents_count += 1
            except Exception as e:
                    logging.error(f"Error during decision making for parcel {parcel_id} (Agent {current_agent.__class__.__name__}): {e}", exc_info=True)
                    # Decide if we should store the agent anyway or mark as failed for the year

        logging.info(f"--- Year {sim_year} Complete. Processed {processed_agents_count}/{num_parcels} agents ---")

    # --- Simulation End ---
    logging.info("=== Simulation Finished ===")
    
    # Save results using the utility function
    output_dir = config.paths["output_dir"]
    save_results_to_geojson(
        land_manager_data=super_agent.land_manager_data, 
        agent_map=super_agent.agent_map, 
        output_dir=output_dir, 
        filename='simulation_results.geojson'
    )


if __name__ == "__main__":
    main()
