import os
import yaml  
import numpy as np


class Config:
    def __init__(self, config_path='config.yaml'):
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, 'r') as stream:
                self.cfg_data = yaml.safe_load(stream)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found.")
            self.cfg_data = {}
        except yaml.YAMLError as exc:
            print(f"Error parsing configuration file '{config_path}': {exc}")
            self.cfg_data = {}

        # --- Accessing Config Values ---
        self.simulation_params = self.cfg_data.get('simulation', {})
        self.economic_params = self.cfg_data.get('economic_params', {})
        self.gemini_params = self.cfg_data.get('gemini', {})
        self.paths = self.cfg_data.get('paths', {})
        self.environment_params = self.cfg_data.get('environment', {})
        self.crops_data = self.cfg_data.get('crops_data', {})
        self.government_policies = self.cfg_data.get('government_policies', [])

        # --- Land Manager Properties ---
        lm_props = self.cfg_data.get('lm_props', {})
        self.land_quality = lm_props.get('land_quality', {})
        self.traits = lm_props.get('traits', {})
        self.goals = lm_props.get('goals', [])
        self.province = lm_props.get('province', [])
        self.managerial_ability = lm_props.get('managerial_ability', [])
        
        age_range = lm_props.get('manager_age_range', {})
        self.manager_age = np.arange(
            age_range.get('start', 23),
            age_range.get('end', 76)
        )
        
        capital_range = lm_props.get('liquid_capital_range', {})
        self.liquid_capital = np.arange(
            capital_range.get('start', 100000),
            capital_range.get('end', 1000000),
            capital_range.get('step', 100000)
        )


if __name__ == '__main__':
    pass 