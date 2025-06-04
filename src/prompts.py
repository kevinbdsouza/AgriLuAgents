import logging
import textwrap 

class LandManagerPrompts:
    def __init__(self, manager_details, personality_traits, goals, config):
        self.manager_details = manager_details
        self.personality_traits = personality_traits
        self.goals = goals
        self.config = config
        self.potential_land_uses = config.crops_data["crop_list"] + config.crops_data["other_lus"]
        self.available_options = ", ".join(self.potential_land_uses)
        self.current_policies =  manager_details["current_policies"]
        self.current_news =  manager_details["current_news"]
        self.land_quality_label = manager_details.get('land_quality_label', 'unknown')
        self.land_quality_description = manager_details.get('land_quality_description', 'unknown')
        self.land_size = manager_details.get('land_size', 'unknown')
        self.province = manager_details.get('province', 'unknown')
        self.managerial_ability = manager_details.get('managerial_ability', 'average')
        
        trait_list = [v for k, v in personality_traits.items()]
        traits_str = ", ".join(trait_list)
        goals_str = ", ".join(f"{goal}" for goal in goals)
        if not goals_str: goals_str = "No specific goals defined."

        self.system_prompt = textwrap.dedent(f"""
            You are a land manager in Canada who has to make a land use decisions in your parcel.
            IMPORTANT: Conclude your response *strictly* with the line 
            'Final Decision: [Option]' 
            where '[Option]' is EXACTLY ONE of the following valid choices: {self.available_options}
            Your Personality Traits are:{traits_str}
            Your Goals are: {goals_str} 
            """
        ).strip()

        # Initialize other prompt parts as empty strings
        self.yearly_info_prompt = ""
        self.environment_context_prompt = ""
        self.previous_year_context_prompt = ""
        self.options_summary_prompt = ""
        self.land_prompt = textwrap.dedent("""Land Details:
            - Land Quality for Agriculture: {self.land_quality_label}
            - Limitations: {self.land_quality_description}
            - Size: {self.land_size} hectares
            - Location: {self.province}
            - Managerial Ability: {self.managerial_ability} 
        """)
        self.final_instruction_prompt = ""

    def update_prompts(self, year, env_state, sim_year, last_decision, compute_results):
        """Updates all prompt components for the current simulation year."""
        
        # --- Yearly Info Prompt --- 
        manager_age = self.manager_details.get('current_age')
        self.yearly_info_prompt = textwrap.dedent(f"""
            --- Current Year: {year} (Simulation Year: {sim_year}) ---
            Your Age: {manager_age}
            Government Policies Active: {self.current_policies}
            News Highlights: {self.current_news} 
            """
        ).strip()

        # --- Environment Context Prompt --- 
        market_info = env_state.get("market_prices")
        prices_str = ", ".join([f"{crop}: ${price:.2f}" for crop, price in market_info.items()])
        self.environment_context_prompt = textwrap.dedent(f"""
            Environmental and Market Conditions:
            - Precipitation: {env_state["climate"]["precipitation"]:.2f} mm
            - Temperature: {env_state["climate"]["temperature"]:.2f} degree C
            - Current Market Prices ($/unit): {prices_str}
            """
        ).strip()

        # --- Previous Year Context Prompt --- 
        current_capital = self.manager_details.get('liquid_capital', 0)
        prev_prompt_parts = [f"Your current liquid capital is ${current_capital:.2f}."]
        if sim_year > 1:
            prev_lu = last_decision.get('lu_decision', 'unknown')
            prev_pl = last_decision.get('pl', 'unknown')
            prev_net_cash = last_decision.get('net_cash', 0)
            
            prev_prompt_parts.append(
                f"Last year ({year-1}), you chose '{prev_lu}' which resulted in a {prev_pl} of ${prev_net_cash:.2f}."
            )
            # Add generic advice about considering change vs consistency
            prev_prompt_parts.append(
                 "Consider whether repeating this or changing strategy is better given the new conditions and potential outcomes."
                 # TODO: Add notes about sunk costs/inventory reuse if relevant, especially for non-crop LUs
            )
        else:
             prev_prompt_parts.append("This is the first decision round.")
        self.previous_year_context_prompt = "\n".join(prev_prompt_parts)
        
        # --- Options Summary Prompt (Generated from compute_results) --- 
        options_parts = ["\n--- Potential Land Use Outcomes for This Year ---"]
        
        # Get the evaluated options from the compute results
        evaluated_options = list(compute_results.income_dict.keys())
        if not evaluated_options:
            options_parts.append("Error: No outcome data calculated for land use options.")
        else:
            for option in evaluated_options:
                option_detail = f"\n* Option: {option.upper()} *"
                
                # Get financial results safely
                net_profit = compute_results.income_dict.get(option, 0)
                pl_status = compute_results.pl_dict.get(option, 'unknown')
                annual_cost = compute_results.annual_cost_dict.get(option, 0)
                init_cost = compute_results.init_cost_dict.get(option, 0)
                
                financial_summary = (
                    f"  - Expected Result: {pl_status.capitalize()} of ${net_profit:.2f}\n"
                    f"  - Estimated Annual Cost: ${annual_cost:.2f}\n"
                    f"  - Estimated Initial Cost: ${init_cost:.2f}"
                )
                
                # TODO: Add specific notes/wait times for afforestation/rewilding here
                # based on 'option' key once their compute logic is finalized.
                # Example placeholder:
                other_notes = ""
                if option == "afforestation":
                     other_notes = "  - Note: Timber income requires long-term commitment (e.g., 40 years). Carbon credits may apply earlier."
                elif option == "rewilding":
                     other_notes = "  - Note: Financial benefits mainly from potential carbon credits or land value changes."

                options_parts.append(option_detail)
                options_parts.append(financial_summary)
                if other_notes: options_parts.append(other_notes)

        self.options_summary_prompt = "\n".join(options_parts)

        # Update the list of available options in the final instruction
        available_options_str = ", ".join(evaluated_options) if evaluated_options else "None calculated"
        # Make instruction stricter
        self.final_instruction_prompt = textwrap.dedent(f"""
            Based on all the information provided (your land details, personality, goals, 
            current year conditions, previous year context, policies, news, and the calculated potential outcomes 
            for each land use option), please analyze the situation and make a clear decision 
            for your land use this year. Explain your reasoning step-by-step in your Chain-of-Thought.
            
            IMPORTANT: Conclude your response *strictly* with the line 
            'Final Decision: [Option]' 
            where '[Option]' is EXACTLY ONE of the following valid choices: {available_options_str}
            Do not add any other text or explanation after the 'Final Decision:' line.
            """
        ).strip()

    def get_full_prompt(self):
        """Assembles the complete prompt for the LLM."""
        full_prompt = "\n\n".join([
            self.system_prompt,
            self.yearly_info_prompt,
            self.environment_context_prompt,
            self.previous_year_context_prompt,
            self.options_summary_prompt,
            self.final_instruction_prompt
        ])
        logging.debug(f"Generated Full Prompt:\n-------\n{full_prompt[:500]}...\n-------")
        return full_prompt

# Example Usage (Illustrative - Requires Mock Objects)
if __name__ == '__main__':
    pass 
