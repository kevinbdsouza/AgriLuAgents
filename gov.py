import numpy as np
from config import Config
import json
from data import Data
from farmer import FarmerAgent
import random
from market import Market
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class GovAgent:
    def __init__(self, model_dir, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=4000,
                        temperature=0.7, device="cuda:0")
        self.hf = HuggingFacePipeline(pipeline=pipe)

        #self.hf = ""
        self.agent_map = {}
        self.cfg = Config()
        self.data = Data()
        self.market = Market()

    def get_all_farms_info(self):
        farms = {}
        for i in range(3):
            farms[i] = {}
        return farms

    def get_polygon_props(self, farm):
        agent_polygon = ""
        land_size = 300
        land_quality_crops = ["excellent", "adverse climate"]
        land_quality_trees = ["satisfactory"]
        current_climate = [10, 150]
        future_climate = [+1.5, +50]
        ecological_connectivity = ["does", 100]
        province = "Ontario"
        return (agent_polygon, land_size, land_quality_crops, land_quality_trees, current_climate, future_climate,
                ecological_connectivity, province)

    def get_farm_details(self, farm):
        farmer_age = random.choice(self.cfg.farmer_age)
        managerial_ability = random.choice(self.cfg.managerial_ability)

        liquid_capital = random.choice(self.cfg.liquid_capital)
        plot_details = {}
        return (farmer_age, managerial_ability, liquid_capital,
                plot_details)

    def generate_news(self):
        news = "no news about what neighbours and other farmers are doing"
        return news

    def create_farmer_agent(self, sim_year, farm_id, farm):
        # farm details
        (agent_polygon, land_size, land_quality_crops, land_quality_trees, current_climate, future_climate,
         ecological_connectivity, province) = self.get_polygon_props(farm)
        (farmer_age, managerial_ability, liquid_capital, plot_details) = self.get_farm_details(farm)

        farm_details = {"polygon": agent_polygon, "land_size": land_size, "land_quality_crops": land_quality_crops,
                        "land_quality_trees": land_quality_trees, "current_climate": current_climate,
                        "future_climate": future_climate, "ecological_connectivity": ecological_connectivity,
                        "province": province, "farm_id": farm_id, "farmer_age": farmer_age - sim_year - 1,
                        "managerial_ability": managerial_ability,
                        "liquid_capital": liquid_capital, "plot_details": plot_details}

        # traits
        traits = {}
        for k, v in self.cfg.traits.items():
            choice = random.choice(v)
            if isinstance(choice, list):
                choice = random.choice(choice)
            traits[k] = choice

        # goal
        ids = np.arange(0, len(self.cfg.goals))
        ids = np.random.choice(ids, size=2, replace=False)
        goals = [random.choice(self.cfg.goals[idx]) for idx in ids]

        # government_policies
        government_policies = np.random.choice(self.cfg.government_policies, 2, replace=False)

        # market_info
        market_info = self.market.market_lu_dict

        # news
        news = self.generate_news()

        farmer_agent = FarmerAgent(farm_details, traits, goals, government_policies, market_info, news,
                                   self.cfg.base_year + sim_year, sim_year, self.hf)
        return farmer_agent

    def update_farmer_agent(self, sim_year, farmer_agent):
        farmer_agent.sim_year = sim_year
        farmer_agent.year = self.cfg.base_year + sim_year

        lu_decision = farmer_agent.decision_dict[sim_year - 1]['lu_decision']
        for i, lu_i in enumerate(farmer_agent.compute.lus):
            if lu_i == lu_decision:
                farmer_agent.prompts.st_prompts[lu_i] = (
                    f"You performed {farmer_agent.compute.lu_details[i]} last year. "
                    f"If you decide to continue with ")
            else:
                farmer_agent.prompts.st_prompts[lu_i] = "If you decide to perform "

        if lu_decision in farmer_agent.inventory_dict:
            farmer_agent.prompts.inventory_reuse_prompts[
                lu_decision] = ("You may not need to repurchase technology units as you already have them "
                                "in your inventory. ")

        if lu_decision == "afforestation":
            farmer_agent.aff_continuous += 1
            farmer_agent.csa_food_continuous = 0
            farmer_agent.csa_bio_continuous = 0
            farmer_agent.rewild_continuous = 0

            if farmer_agent.aff_continuous >= 40:
                farmer_agent.prompts.wait_prompts["timber"] = (
                    "You have completed 40 or more continuous years of afforestation. "
                    "You can start making profit from timber this year. ")
            else:
                farmer_agent.prompts.wait_prompts["timber"] = (
                    f"You will need to wait {40 - farmer_agent.aff_continuous} years "
                    "before you start making profit from timber, but might be "
                    "worth it, depends on what you want to achieve. ")
        elif lu_decision == "csa food":
            farmer_agent.csa_food_continuous += 1
            farmer_agent.aff_continuous = 0
            farmer_agent.csa_bio_continuous = 0
            farmer_agent.rewild_continuous = 0

            if farmer_agent.csa_food_continuous >= 5:
                farmer_agent.prompts.wait_prompts["csa food"] = (
                    "You have completed 5 or more continuous years of climate "
                    "smart agriculture for food crops. "
                    "You can start seeing increase in yields this year. ")
            else:
                farmer_agent.prompts.wait_prompts["csa food"] = (
                    f"You will need to wait {5 - farmer_agent.csa_food_continuous} years "
                    "before you start seeing increase in yields, but might be "
                    "worth it, depends on what you want to achieve. ")
        elif lu_decision == "csa bio":
            farmer_agent.csa_bio_continuous += 1
            farmer_agent.csa_food_continuous = 0
            farmer_agent.aff_continuous = 0
            farmer_agent.rewild_continuous = 0

            if farmer_agent.csa_bio_continuous >= 5:
                farmer_agent.prompts.wait_prompts["csa bio"] = (
                    "You have completed 5 or more continuous years of climate "
                    "smart agriculture for bioenergy crops. "
                    "You can start seeing increase in yields this year. ")
            else:
                farmer_agent.prompts.wait_prompts["csa bio"] = (
                    f"You will need to wait {5 - farmer_agent.csa_bio_continuous} years "
                    "before you start seeing increase in yields, but "
                    "might be worth it, depends on what you want to achieve. ")
        elif lu_decision == "rewilding":
            farmer_agent.rewild_continuous += 1
            farmer_agent.csa_bio_continuous = 0
            farmer_agent.csa_food_continuous = 0
            farmer_agent.aff_continuous = 0
        else:
            farmer_agent.csa_bio_continuous = 0
            farmer_agent.csa_food_continuous = 0
            farmer_agent.aff_continuous = 0
            farmer_agent.rewild_continuous = 0

        farmer_agent.compute.compute_agent_income(farmer_agent)

        # government_policies
        farmer_agent.government_policies = np.random.choice(self.cfg.government_policies, 2, replace=False)

        # market_info
        farmer_agent.market_info = self.market.market_lu_dict

        # news
        farmer_agent.news = self.generate_news()

        farmer_agent.prompts.update_prompts(farmer_agent.year, farmer_agent.government_policies,
                                            farmer_agent.market_info, farmer_agent.news,
                                            farmer_agent.sim_year, farmer_agent.decision_dict,
                                            farmer_agent.compute)
        farmer_agent.in_context_decision_making(self.hf)
        return farmer_agent
