import random
import numpy as np
from config import Config

class Environment:
    """
    Manages dynamic environmental factors like weather and market prices.
    Climate (precipitation, temperature) is handled on a per-parcel basis.
    Market prices are global.
    """
    def __init__(self, config: Config, parcel_ids: list):
        """
        Initializes the environment based on configuration and parcel identifiers.

        Args:
            config: The main Config object (loaded from config.yaml).
            parcel_ids: A list of unique identifiers for each land parcel.
        """
        self.time_step = 0
        self.config = config
        self.parcel_ids = parcel_ids
        self.base_year = config.simulation_params.get('base_year', 2025)

        # Load environment parameters from config
        env_params = config.environment_params
        self.base_market_prices = env_params.get('base_market_prices', {})
        self.market_price_volatility = env_params.get('market_price_volatility', 0.1)
        # Get climate ranges and variability
        precip_range = env_params.get('base_precipitation_range', [600, 600]) # Default to single value if range not provided
        temp_range = env_params.get('base_temperature_range', [15, 15]) # Default to single value
        self.precipitation_variability = env_params.get('precipitation_variability', 100)
        self.temperature_variability = env_params.get('temperature_variability', 2)
        # Get climate trends
        self.ssp3_precip_trend = env_params.get('ssp3_precipitation_trend', 0) # Trend factor per year
        self.ssp3_temp_trend = env_params.get('ssp3_temperature_trend', 0) # Absolute change per year
        # Carbon and timber prices
        self.base_carbon_price = env_params.get('base_carbon_price', 0)
        self.carbon_price_volatility = env_params.get('carbon_price_volatility', 0.1)
        self.base_timber_price = env_params.get('base_timber_price', 0)
        self.timber_price_volatility = env_params.get('timber_price_volatility', 0.1)

        # Load crop data
        crops_data = config.crops_data
        self.crops = crops_data.get('crop_list', [])
        self.crop_params = {crop: crops_data.get(crop, {}) for crop in self.crops}

        # Initialize market prices (global)
        self.current_prices = self.base_market_prices.copy()
        self.current_prices['carbon'] = self.base_carbon_price
        self.current_prices['timber'] = self.base_timber_price

        # Initialize parcel-specific climate states
        self.base_precipitation = {}
        self.base_temperature = {}
        self.current_precipitation = {}
        self.current_temperature = {}

        for pid in self.parcel_ids:
            # TODO: Replace random sampling with actual historical/projected climate data per parcel.
            # This might involve loading data from a file or an external source based on parcel_id or location.
            # Sample base climate values for each parcel from the specified range (CURRENTLY RANDOM PLACEHOLDER)
            base_precip = random.uniform(precip_range[0], precip_range[1])
            base_temp = random.uniform(temp_range[0], temp_range[1])
            self.base_precipitation[pid] = base_precip
            self.base_temperature[pid] = base_temp
            # Initialize current climate with base values
            self.current_precipitation[pid] = base_precip
            self.current_temperature[pid] = base_temp


    def step(self):
        """
        Advances the environment by one time step (e.g., a year or growing season), updating dynamic factors.
        """
        self.time_step += 1
        self._update_market_prices()
        self._update_climate() # Updates climate for all parcels

    def _update_market_prices(self):
        """Updates market prices based on volatility and trends (if any)."""
        for item, base_price in self.base_market_prices.items():
            change_factor = np.random.normal(1, self.market_price_volatility)
            # Prevent negative prices
            self.current_prices[item] = max(0.01, base_price * change_factor)

        # Update carbon price
        carbon_change = np.random.normal(1, self.carbon_price_volatility)
        self.current_prices['carbon'] = max(0.01, self.current_prices.get('carbon', self.base_carbon_price) * carbon_change)

        # Update timber price
        timber_change = np.random.normal(1, self.timber_price_volatility)
        self.current_prices['timber'] = max(0.01, self.current_prices.get('timber', self.base_timber_price) * timber_change)

    def _update_climate(self):
        """Updates climate variables for each parcel based on variability and trends."""
        for pid in self.parcel_ids:
            # Apply trend first (simple linear trend for SSP3 example)
            # Trend modifies the base for the current year before variability is applied
            current_year_base_precip = self.base_precipitation[pid] * (1 + self.ssp3_precip_trend * self.time_step)
            current_year_base_temp = self.base_temperature[pid] + self.ssp3_temp_trend * self.time_step

            # Apply variability around the current year's trended base
            self.current_precipitation[pid] = max(0, np.random.normal(current_year_base_precip, self.precipitation_variability))
            self.current_temperature[pid] = np.random.normal(current_year_base_temp, self.temperature_variability)

    def get_yield_adjustment_factor(self, crop_name, parcel_id):
        """
        Calculates the yield adjustment factor for a given crop on a specific parcel,
        based on the parcel's current weather conditions compared to the crop's optimal conditions.
        Returns a factor (e.g., 1.0 for optimal, <1.0 for suboptimal).

        Args:
            crop_name: The name of the crop.
            parcel_id: The identifier of the parcel.
        """
        params = self.crop_params.get(crop_name)
        if not params or parcel_id not in self.current_precipitation:
             # Return 1.0 if no crop data or parcel climate data exists
            return 1.0

        current_precip = self.current_precipitation[parcel_id]
        current_temp = self.current_temperature[parcel_id]

        precip_deviation = abs(current_precip - params.get('optimal_precipitation', current_precip))
        temp_deviation = abs(current_temp - params.get('optimal_temperature', current_temp))

        precip_sensitivity = params.get('yield_sensitivity_to_precipitation', 0)
        temp_sensitivity = params.get('yield_sensitivity_to_temperature', 0)

        # Calculate penalty based on deviation and sensitivity
        # Simple linear penalty model
        precip_penalty = precip_deviation * precip_sensitivity
        temp_penalty = temp_deviation * temp_sensitivity

        # Total adjustment factor (capped at 0 minimum)
        adjustment_factor = max(0, 1.0 - precip_penalty - temp_penalty)

        return adjustment_factor

    def get_current_state(self, parcel_id=None):
        """
        Returns the current state of the environment. If parcel_id is provided,
        includes parcel-specific climate data. Otherwise, returns only global data.

        Args:
            parcel_id: The identifier of the parcel for which to get climate data. (Optional)

        Returns:
            A dictionary containing the current environment state.
        """
        state = {
            "market_prices": self.current_prices.copy(),  # Includes crops, carbon, timber (global)
            "year": self.base_year + self.time_step,     # Global
        }
        state["climate"] = {
            "precipitation": self.current_precipitation.get(parcel_id, None), 
            "temperature": self.current_temperature.get(parcel_id, None)  
        }
        return state

# Example Usage (if run directly)
if __name__ == '__main__':
    pass 