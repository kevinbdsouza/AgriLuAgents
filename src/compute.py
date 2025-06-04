import random
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import logging
import os


class Compute:
    def __init__(self, config, parcel_specific_base_yields=None):
        self.cfg = config
        self.base_yields = parcel_specific_base_yields or {}
        logging.debug(f"Compute module initialized with specific base yields: {self.base_yields}")
        
        crop_lus = self.cfg.crops_data["crop_list"]
        other_lus = self.cfg.crops_data["other_lus"]
        self.lus = crop_lus + other_lus 

        self.interest_rate = self.cfg.economic_params["interest_rate"]
        self.inflation_rate = self.cfg.economic_params["inflation_rate"]
        self.income_dict = {}
        self.pl_dict = {}
        self.annual_cost_dict = {}
        self.init_cost_dict = {}
        self.other_scores = {}
        logging.info("Compute module initialized with Config.")
        self.solver_name = 'ipopt'
        self.solver_path = self.cfg.paths["solver_path"]

    def r_ch(self, a, b):
        return random.choice([a, b])

    def get_pl(self, num):
        return "profit" if num >= 0 else "loss"

    def get_sign(self, pl):
        return 1 if pl == "profit" else -1

    def _solve_model(self, model):
        if not self.solver_name:
            logging.error("Cannot solve model: Solver not available.")
            return None, SolverStatus.error

        solver_executable = self.solver_path if self.solver_path else None
        solver = pyo.SolverFactory(self.solver_name, executable=solver_executable)

        try:
            results = solver.solve(model, tee=False)
        except Exception as e:
            logging.error(f"Error during solver execution with executable '{solver_executable}': {e}")
            return None, SolverStatus.error

        if (results.solver.status == SolverStatus.ok) and \
           (results.solver.termination_condition == TerminationCondition.optimal or \
            results.solver.termination_condition == TerminationCondition.locallyOptimal or \
            results.solver.termination_condition == TerminationCondition.feasible):
            logging.debug(f"Solver found a solution. Status: {results.solver.status}, Condition: {results.solver.termination_condition}")
            return results, results.solver.status
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logging.warning("Solver determined the problem is infeasible.")
            return results, results.solver.status
        else:
            logging.error(f"Solver failed. Status: {results.solver.status}, Condition: {results.solver.termination_condition}")
            return results, results.solver.status

    def compute_agent_income(self, land_manager_agent, environment_state):
        manager_details = land_manager_agent.manager_details
        sim_year = land_manager_agent.sim_year
        parcel_id = manager_details.get('parcel_id', 'N/A') 

        current_prices = environment_state.get('market_prices', {})
        yield_adjustments = environment_state.get('yield_adjustments', {})

        # Get the assigned land quality label and look up the factor
        quality_label = manager_details.get('land_quality_label', 'good') 
        quality_description = manager_details.get('land_quality_description', 'N/A') 

        size = manager_details.get("land_size", 100)
        liquid_capital = manager_details.get("liquid_capital", 50000)
        interest = liquid_capital * (self.interest_rate / 100.0)
        compound_inflation = (1 + self.inflation_rate)**(sim_year - 1)

        self.income_dict = {}
        self.pl_dict = {}
        self.annual_cost_dict = {}
        self.init_cost_dict = {}

        logging.debug(f"Computing income for Land Manager {parcel_id}, Year {sim_year}")
        logging.debug(f" Quality Label: '{quality_label}', Desc: '{quality_description}'")
        logging.debug(f" Env State: Prices={ {k:f'{v:.2f}' for k,v in current_prices.items()} }, Yield Adjustments={ {k:f'{v:.2f}' for k,v in yield_adjustments.items()} }")

        available_crops = self.cfg.get('crops_data.crop_list', [])
        for crop in available_crops:
            logging.debug(f"-- Evaluating Crop: {crop} --")
            
            # --- Get Base Yield: Prioritize specific, then config default --- 
            base_yield_per_ha = self.base_yields.get(crop) 
            yield_source = "parcel-specific"
            base_cost_per_ha = self.cfg.crops_data[crop]["base_cost"]
            market_price = current_prices.get(crop, 0)
            yield_adj_factor = yield_adjustments.get(crop, 1.0) 

            logging.debug(f"  Parcel {parcel_id}, Crop {crop}: Base Yield ({yield_source}) = {base_yield_per_ha:.2f} t/ha")
            current_yield_per_ha = base_yield_per_ha * yield_adj_factor
            
            model = pyo.ConcreteModel(name=f"Crop_{crop}")
            model.size = pyo.Param(initialize=size)
            model.liquid_capital = pyo.Param(initialize=liquid_capital)
            model.interest_earned = pyo.Param(initialize=interest)
            model.compound_inflation = pyo.Param(initialize=compound_inflation)
            model.market_price = pyo.Param(initialize=market_price)
            model.yield_per_ha = pyo.Param(initialize=current_yield_per_ha)
            model.cost_per_ha = pyo.Param(initialize=base_cost_per_ha)
            model.total_yield = model.yield_per_ha * model.size
            model.total_revenue = model.total_yield * model.market_price
            model.total_variable_cost = model.cost_per_ha * model.size
            model.total_annual_cost = model.total_variable_cost * model.compound_inflation 
            model.net_profit = model.total_revenue - model.total_annual_cost + model.interest_earned
            model.objective = pyo.Objective(expr=model.net_profit, sense=pyo.maximize)
            try:
                net_profit_value = pyo.value(model.net_profit)
                total_cost_value = pyo.value(model.total_annual_cost)
                status = SolverStatus.ok
            except Exception as e:
                logging.error(f"Error evaluating Pyomo model for crop {crop} on parcel {parcel_id}: {e}")
                net_profit_value = -np.inf
                total_cost_value = np.inf
                status = SolverStatus.error
            if status == SolverStatus.ok:
                logging.debug(f"  Crop {crop}: Price=${market_price:.2f}, Final Yield/ha={current_yield_per_ha:.2f}, Cost/ha=${base_cost_per_ha:.2f}")
                logging.debug(f"  -> Total Revenue: ${pyo.value(model.total_revenue):.2f}, Total Cost: ${total_cost_value:.2f}, Net Profit: ${net_profit_value:.2f}")
                self.income_dict[crop] = net_profit_value
                self.pl_dict[crop] = self.get_pl(net_profit_value)
                self.annual_cost_dict[crop] = total_cost_value
                self.init_cost_dict[crop] = 0
            else:
                logging.warning(f"Could not calculate profit for crop {crop} on parcel {parcel_id}. Status: {status}")
                self.income_dict[crop] = 0 
                self.pl_dict[crop] = "loss"
                self.annual_cost_dict[crop] = 0
                self.init_cost_dict[crop] = 0

        if 'afforestation' in self.lus:
            logging.debug("-- Evaluating LU: afforestation --")
            base_carbon_seq = self.cfg.get('afforestation_data.base_carbon_sequestration', 0)
            base_timber_yield = self.cfg.get('afforestation_data.base_timber_yield', 0)
            maturity_years = self.cfg.get('afforestation_data.timber_maturity_years', 40)
            establishment_cost_ha = self.cfg.get('afforestation_data.establishment_cost', 0)
            maintenance_cost_ha = self.cfg.get('afforestation_data.maintenance_cost', 0)
            harvest_cost_m3 = self.cfg.get('afforestation_data.harvest_cost', 0)
            
            carbon_price = current_prices.get('carbon', 0)
            timber_price = current_prices.get('timber', 0)
            
            aff_years = land_manager_agent.aff_continuous
            is_first_year = (aff_years == 0)

            initial_cost_total = establishment_cost_ha * size if is_first_year else 0
            maintenance_cost_total = maintenance_cost_ha * size * compound_inflation
            annual_cost = maintenance_cost_total

            carbon_revenue = base_carbon_seq * size * carbon_price
            timber_revenue = 0
            if aff_years >= maturity_years:
                 timber_yield_total = base_timber_yield * size
                 harvest_cost_total = timber_yield_total * harvest_cost_m3 * compound_inflation
                 timber_revenue = (timber_yield_total * timber_price) - harvest_cost_total
                 annual_cost += harvest_cost_total
            
            total_revenue = carbon_revenue + timber_revenue
            net_profit = total_revenue - annual_cost + interest

            self.income_dict['afforestation'] = net_profit
            self.pl_dict['afforestation'] = self.get_pl(net_profit)
            self.annual_cost_dict['afforestation'] = annual_cost
            self.init_cost_dict['afforestation'] = initial_cost_total
            logging.debug(f"  Afforestation: CarbonRev=${carbon_revenue:.2f}, TimberRev=${timber_revenue:.2f}, AnnCost=${annual_cost:.2f}, InitCost=${initial_cost_total:.2f}, NetProfit=${net_profit:.2f}")

        if 'rewilding' in self.lus:
            logging.debug("-- Evaluating LU: rewilding --")
            base_carbon_seq = self.cfg.get('rewilding_data.base_carbon_sequestration', 0)
            establishment_cost_ha = self.cfg.get('rewilding_data.establishment_cost', 0)
            maintenance_cost_ha = self.cfg.get('rewilding_data.maintenance_cost', 0)
            
            carbon_price = current_prices.get('carbon', 0)

            rewild_years = land_manager_agent.rewild_continuous
            is_first_year = (rewild_years == 0)

            initial_cost_total = establishment_cost_ha * size if is_first_year else 0
            maintenance_cost_total = maintenance_cost_ha * size * compound_inflation
            annual_cost = maintenance_cost_total

            carbon_revenue = base_carbon_seq * size * carbon_price
            total_revenue = carbon_revenue
            
            net_profit = total_revenue - annual_cost + interest

            self.income_dict['rewilding'] = net_profit
            self.pl_dict['rewilding'] = self.get_pl(net_profit)
            self.annual_cost_dict['rewilding'] = annual_cost
            self.init_cost_dict['rewilding'] = initial_cost_total
            logging.debug(f"  Rewilding: CarbonRev=${carbon_revenue:.2f}, AnnCost=${annual_cost:.2f}, InitCost=${initial_cost_total:.2f}, NetProfit=${net_profit:.2f}")

        return self.income_dict

    def get_eco_score(self):
        pass

    def get_carbon_score(self):
        pass

    def get_food_security_score(self):
        pass

    def get_energy_security_score(self):
        pass
