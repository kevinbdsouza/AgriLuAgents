from data import Data
from land_manager import LandManagerAgent
import random
import numpy as np
from utils import *
from typing import Dict, Union, List
from scipy.optimize import linear_sum_assignment
import logging
import geopandas as gp
import json
import os
from utils import assign_land_quality_based_on_yields 


class SuperAgent:
    def __init__(self, config):
        """
        Initializes the SuperAgent with a configuration object.
        """
        self.config = config 
        self.agent_map = {} # Map to store LandManagerAgent instances by parcel_id
        
        # Initialize data loader
        self.data = Data() 
        self.land_manager_data = self.data.land_manager_data 
        self.parcel_ids = self.land_manager_data.index.tolist()
        # --- Pre-computation: Assign Land Quality based on Yield (using utility function) ---
        self.parcel_yield_details, self.parcel_assigned_qualities = assign_land_quality_based_on_yields(
            self.land_manager_data, 
            self.parcel_ids, 
            config
        )
        
        logging.info("SuperAgent initialized with Config and loaded land manager data.")
        logging.info(f"Loaded {len(self.land_manager_data)} land parcels.")

    def get_current_policies(self, sim_year):
        # TODO: Add more sophisticated policy generation
        policies = self.config.government_policies
        policies = f"Year {sim_year} Policies: {random.choice(policies)}"
        logging.debug(f"Policies for year {sim_year}: {policies}")
        return policies

    def get_current_news(self, sim_year):
        # TODO: Add more sophisticated news generation
        news_items = [
            "Market prices stable.",
            "Input costs rising slightly.",
            "Weather forecast predicts average rainfall."
        ]
        # Could add logic based on environment state if needed
        news = f"Year {sim_year} News Report: {random.choice(news_items)}"
        logging.debug(f"News for year {sim_year}: {news}")
        return news

    def create_land_manager_agent(self, sim_year, parcel_id, parcel_data, config): 
        """Creates a LandManagerAgent instance using the provided parcel data, config, and specific yields."""
        logging.debug(f"Creating land manager agent for parcel {parcel_id} in year {sim_year}")

        manager_details = parcel_data.to_dict() 
        manager_details['parcel_id'] = parcel_id 

        # Example: Generate initial age and capital
        manager_details["initial_age"] = int(random.choice(config.manager_age))
        manager_details["liquid_capital"] = int(random.choice(config.liquid_capital))

        # 2. Get context for agents (Policies, News) - These are assumed global for now
        manager_details["current_policies"] = self.get_current_policies(sim_year)
        manager_details["current_news"] = self.get_current_news(sim_year)
        logging.debug(f"Retrieved policies and news for year {sim_year}.")

        # Calculate current age based on simulation year
        start_year = 1
        base_year = config.simulation_params["base_year"]
        manager_details["current_age"] = manager_details["initial_age"] + (sim_year - start_year)
        current_real_year = base_year + sim_year - start_year

        # Traits - Load from the passed config object
        traits = {}
        cfg_traits = config.traits
        for k, v in cfg_traits.items():
            if isinstance(v, (list, tuple)) and v:
                choice = random.choice(v)
                if isinstance(choice, (list, tuple)) and choice: choice = random.choice(choice)
                traits[k] = choice
            else: logging.warning(f"Invalid format for trait '{k}' in config: {v}")
        manager_details['traits'] = traits

        # Goals - Load from the passed config object
        goals = []
        cfg_goals = config.goals
        if isinstance(cfg_goals, list) and cfg_goals:
            num_goals_to_select = min(2, len(cfg_goals))
            ids = np.random.choice(np.arange(len(cfg_goals)), size=num_goals_to_select, replace=False)
            goals = [random.choice(cfg_goals[idx]) for idx in ids]
        manager_details['goals'] = goals 

        manager_details['land_quality_label'] = self.parcel_assigned_qualities[parcel_id]['label']
        manager_details['land_quality_description'] = self.parcel_assigned_qualities[parcel_id]['description']
        
        manager_details['managerial_ability'] = random.choice(config.managerial_ability)
        manager_details['province'] = random.choice(config.province)
        manager_details['land_size'] = parcel_data['geometry'].area / 10000 # Calculate area in hectares

        land_manager_agent = LandManagerAgent(
            manager_details=manager_details, 
            traits=traits,
            goals=goals,
            year=current_real_year,
            sim_year=sim_year,
            config=config,
            parcel_specific_base_yields=self.parcel_yield_details[parcel_id]
        )
        logging.info(f"Created LandManagerAgent {parcel_id} for year {current_real_year} (sim_year {sim_year})")
        return land_manager_agent
       

if __name__ == "__main__":
    pass 
