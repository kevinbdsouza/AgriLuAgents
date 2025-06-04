import os
from compute import Compute
from in_context_learn import InContextLearner
from prompts import LandManagerPrompts
import logging


class LandManagerAgent:
    def __init__(self, manager_details, traits, goals, year, sim_year, config, parcel_specific_base_yields=None): 
        self.manager_details = manager_details 
        self.personality_traits = traits
        self.goals = goals
        self.year = year
        self.sim_year = sim_year
        self.config = config 
        self.base_yields = parcel_specific_base_yields or {}
        logging.debug(f"Agent {self.manager_details.get('parcel_id', 'N/A')} received specific yields: {self.base_yields}")

        # Instantiate Compute with config AND specific yields
        self.compute = Compute(config, parcel_specific_base_yields=self.base_yields)

        self.decision_dict = {}
        self.inventory_dict = {} 
        self.aff_continuous = self.manager_details.get('aff_continuous', 0) 
        self.rewild_continuous = self.manager_details.get('rewild_continuous', 0)

        self.prompts = LandManagerPrompts(manager_details, traits, goals, self.config)
        
        logging.info(f"LandManagerAgent initialized for manager/parcel {self.manager_details.get('manager_id', self.manager_details.get('parcel_id', 'N/A'))} for year {self.year} (sim_year {self.sim_year}).")

    def in_context_decision_making(self, environment_state):
        """
        Makes a land use decision based on the current environment state and land manager's characteristics.
        Uses the LLM to generate a decision with reasoning.
        """
        # Update prompts with current environment state
        self.prompts.update_prompts(
            year=self.year,
            env_state=environment_state,
            sim_year=self.sim_year,
            last_decision=self.decision_dict.get(self.sim_year - 1, {}),
            compute_results=self.compute
        )

        # Get the full prompt
        full_prompt = self.prompts.get_full_prompt()
        
        # Get decision from LLM
        # Assuming get_decision_from_llm is defined elsewhere or needs to be implemented
        # For now, let's assume it exists and returns the required tuple.
        # lu_decision, reasoning, lu_verbose = self.get_decision_from_llm(full_prompt)
        # Placeholder call for now:
        lu_decision, reasoning, lu_verbose = self.parse_decision_from_response("Placeholder LLM Response. Final Decision: rewilding") # Placeholder

        # Store the decision text
        self.decision_dict[self.sim_year] = {
            'lu_decision': lu_decision,
            'reasoning': reasoning,
            'lu_verbose': lu_verbose
        }

        # Update land manager state based on the *chosen* land use (lu_decision)
        if lu_decision != "no decision" and lu_decision != "unknown decision":
            net_cash = self.compute.income_dict.get(lu_decision, 0)
            profit_loss = self.compute.pl_dict.get(lu_decision, "loss" if net_cash < 0 else "profit")
            annual_cost = self.compute.annual_cost_dict.get(lu_decision, 0)
            initial_cost = self.compute.init_cost_dict.get(lu_decision, 0)

            logging.debug(f" Applying decision '{lu_decision}': NetCash=${net_cash:.2f}, PL={profit_loss}, AnnCost=${annual_cost:.2f}, InitCost=${initial_cost:.2f}")

            if lu_decision not in self.inventory_dict:
                self.inventory_dict[lu_decision] = {}
            self.inventory_dict[lu_decision]["year"] = self.sim_year

            current_capital = self.manager_details.get("liquid_capital", 0)
            capital_after_init_costs = current_capital - initial_cost
            
            manager_id = self.manager_details.get('manager_id', self.manager_details.get('parcel_id', 'N/A'))
            if capital_after_init_costs < 0:
                 logging.warning(f" Land Manager {manager_id} has insufficient capital ({current_capital:.2f}) for initial costs ({initial_cost:.2f}) of {lu_decision}. Decision may not be feasible.")
                 net_cash = capital_after_init_costs 
                 profit_loss = "loss"
                 self.manager_details["liquid_capital"] = 0
            else:
                 self.manager_details["liquid_capital"] = capital_after_init_costs + net_cash
            
            self.decision_dict[self.sim_year]['net_cash'] = net_cash 
            self.decision_dict[self.sim_year]['pl'] = profit_loss
            self.decision_dict[self.sim_year]['annual_cost'] = annual_cost
            self.decision_dict[self.sim_year]['initial_cost'] = initial_cost
            self.decision_dict[self.sim_year]['liquid_capital'] = self.manager_details["liquid_capital"]

            logging.debug(f" Land Manager {manager_id} updated liquid capital to: {self.manager_details['liquid_capital']:.2f}")

            self.update_continuous_counters(lu_decision)

        else: 
             manager_id = self.manager_details.get('manager_id', self.manager_details.get('parcel_id', 'N/A'))
             logging.info(f"Land Manager {manager_id} made no decision or decision was unknown. Capital unchanged.")
             self.decision_dict[self.sim_year]['net_cash'] = 0
             self.decision_dict[self.sim_year]['pl'] = "neutral"
             self.decision_dict[self.sim_year]['liquid_capital'] = self.manager_details.get("liquid_capital", 0)

    def update_continuous_counters(self, current_decision):
        """Updates counters for consecutive years of the same land use."""
        # Only track counters for multi-year processes
        if current_decision == "afforestation":
            self.aff_continuous += 1
            self.reset_other_counters(current_decision)
        elif current_decision == "rewilding":
            self.rewild_continuous += 1
            self.reset_other_counters(current_decision)
        # Removed CSA checks
        # elif current_decision == "csa food":
        #     self.csa_food_continuous += 1
        #     self.reset_other_counters(current_decision)
        # elif current_decision == "csa bio":
        #     self.csa_bio_continuous += 1
        #     self.reset_other_counters(current_decision)
        else: # Any other decision (crops) resets the multi-year counters
            self.reset_other_counters("")
        
        # Store updated counters back into manager_details - Important for persistence
        self.manager_details['aff_continuous'] = self.aff_continuous
        self.manager_details['rewild_continuous'] = self.rewild_continuous
        # Remove CSA counters from manager_details if they were stored there
        # self.manager_details.pop('csa_food_continuous', None)
        # self.manager_details.pop('csa_bio_continuous', None)

    def reset_other_counters(self, except_lu):
        """Resets all continuous counters except the one specified."""
        if except_lu != "afforestation": self.aff_continuous = 0
        if except_lu != "rewilding": self.rewild_continuous = 0
        # Removed CSA resets
        # if except_lu != "csa food": self.csa_food_continuous = 0
        # if except_lu != "csa bio": self.csa_bio_continuous = 0

    def parse_decision_from_response(self, response_text):
        logging.debug(f"Parsing response: {response_text[:200]}...")
        reasoning = response_text # Default reasoning is the full text
        lu_decision = "unknown decision" # Default
        lu_verbose = ""
        final_decision_marker = "final decision:"

        response_lower = response_text.lower()
        marker_index = response_lower.rfind(final_decision_marker)

        if marker_index != -1:
            # Extract text after the marker
            decision_text = response_text[marker_index + len(final_decision_marker):].strip()
            # Keep the reasoning part before the marker
            reasoning = response_text[:marker_index].strip()
            lu_verbose = decision_text # The raw text of the decision part
            
            logging.debug(f"Extracted text after marker: '{decision_text}'")

            # --- Strict Decision Matching --- 
            valid_decisions = self.compute.lus
            found_match = False
            for valid_option in valid_decisions:
                # Check for exact match (case-insensitive, trimmed)
                if decision_text.lower() == valid_option.lower():
                    lu_decision = valid_option # Use the correctly cased option name
                    found_match = True
                    logging.info(f"Strictly matched valid decision: {lu_decision}")
                    break
            
            if not found_match:
                 logging.warning(
                     f"Text after 'Final Decision:' ('{decision_text}') did not exactly match any valid option: {valid_decisions}. Decision unknown."
                 )
                 # lu_decision remains "unknown decision"
        else:
            # Marker not found - cannot reliably parse
            logging.warning(f"'{final_decision_marker}' marker not found in response. Cannot parse decision. Response: {response_text[:100]}...")
            lu_verbose = response_text # Verbose is the whole response if marker missing
            # lu_decision remains "unknown decision"
        
        # Store the extracted reasoning separately if marker was found
        # self.decision_dict[self.sim_year]['reasoning'] = reasoning # Already done when storing dict
        
        logging.debug(f"Final Parsed decision: {lu_decision}, Reasoning: {reasoning[:100]}..., Verbose: {lu_verbose[:100]}...")
        return lu_decision, reasoning, lu_verbose

    def get_decision_from_llm(self, prompt):
        # This method needs to be implemented to call the actual LLM
        logging.warning("get_decision_from_llm is not implemented. Returning placeholder response.")
        # Replace this with your actual LLM call logic
        # For example, using InContextLearner or another library
        # response_text = self.in_context_learner.generate(prompt)
        response_text = "Placeholder response based on prompt. Considering market trends and land manager goals. Soil is good. Final Decision: corn"
        return self.parse_decision_from_response(response_text)

    def get_neighbour_info(self):
        logging.debug("get_neighbour_info called - currently no implementation.")
        pass

    # Add a method to update state for the next step
    def prepare_for_next_year(self, next_year, next_sim_year):
        self.year = next_year
        self.sim_year = next_sim_year
        # Carry over relevant state: manager_details (incl updated capital), inventory, counters
        # Traits and goals are likely static
        # Policies, market_info, news will be updated by the main loop
        manager_id = self.manager_details.get('manager_id', self.manager_details.get('parcel_id', 'N/A'))
        logging.debug(f"Land Manager {manager_id} prepared for year {self.year}.")
