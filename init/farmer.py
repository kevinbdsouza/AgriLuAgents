import os
from compute import Compute
from in_context_learn import InContextLearner
from prompts import FarmerPrompts

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


class FarmerAgent:
    def __init__(self, farm_details, traits, goals, government_policies,
                 market_info, news, year, sim_year, hf_pipeline):
        self.farm_details = farm_details
        self.personality_traits = traits
        self.goals = goals
        self.government_policies = government_policies
        self.market_info = market_info
        self.news = news
        self.year = year
        self.sim_year = sim_year
        self.compute = Compute()

        self.decision_dict = {}
        self.inventory_dict = {}
        self.aff_continuous = 0
        self.csa_food_continuous = 0
        self.csa_bio_continuous = 0
        self.rewild_continuous = 0

        self.compute.compute_agent_income(self)

        self.prompts = FarmerPrompts(farm_details, traits, goals, self.compute.lus)
        self.prompts.update_prompts(year, government_policies, market_info, news, sim_year, self.decision_dict,
                                    self.compute)
        self.in_context_decision_making(hf_pipeline)

    def in_context_decision_making(self, hf_pipeline):
        icl_ob = InContextLearner(self)
        response = icl_ob.chain_of_thought(hf_pipeline)
        lu_decision, reasoning, lu_verbose = self.parse_decision_from_response(response)

        #lu_decision = "rewilding"
        #reasoning, lu_verbose = "", ""

        self.decision_dict[self.sim_year] = {}
        self.decision_dict[self.sim_year]['lu_decision'] = lu_decision
        self.decision_dict[self.sim_year]['reasoning'] = reasoning
        self.decision_dict[self.sim_year]['lu_verbose'] = lu_verbose
        if lu_decision not in self.inventory_dict:
            self.inventory_dict[lu_decision] = {}
            self.inventory_dict[lu_decision]["year"] = self.sim_year

        self.inventory_dict[lu_decision]["lab_tech"] = self.compute.labour_tech_capacity[lu_decision]

        if lu_decision == "afforestation":
            net_cash = self.compute.get_sign(self.compute.pl_dict[lu_decision]["carbon"]) * \
                       self.compute.income_dict[lu_decision]["carbon"] + self.compute.get_sign(
                self.compute.pl_dict[lu_decision]["timber"]) * self.compute.income_dict[lu_decision]["timber"]
        else:
            net_cash = self.compute.income_dict[lu_decision]

        self.decision_dict[self.sim_year]['net_cash'] = net_cash
        self.decision_dict[self.sim_year]['pl'] = self.compute.get_pl(net_cash)

        if net_cash < 0:
            self.farm_details["liquid_capital"] = self.farm_details["liquid_capital"] + net_cash
            self.decision_dict[self.sim_year]['liquid_capital'] = self.farm_details["liquid_capital"]
        else:
            savings = net_cash * self.compute.savings_percent
            net_left = net_cash - savings
            self.farm_details["liquid_capital"] = self.farm_details["liquid_capital"] + net_left
            self.decision_dict[self.sim_year]['liquid_capital'] = self.farm_details["liquid_capital"]

    def parse_decision_from_response(self, response):
        response = response['lu_decision']
        id_r = response.find("<|assistant|>")
        reasoning = response[id_r:]

        print(reasoning)

        id_d = reasoning.find("Final decision:")
        lu_verbose = reasoning[id_d:]

        print(lu_verbose)

        if (("climate smart agriculture" in lu_verbose or "climate-smart agriculture" in lu_verbose) and
                "food" in lu_verbose):
            lu_decision = "csa food"
        elif (("climate smart agriculture" in lu_verbose or "climate-smart agriculture" in lu_verbose) and
              "bioenergy" in lu_verbose):
            lu_decision = "csa bio"
        elif "food" in lu_verbose:
            lu_decision = "food crops"
        elif "bioenergy" in lu_verbose:
            lu_decision = "bioenergy crops"
        elif "afforestation" in lu_verbose:
            lu_decision = "afforestation"
        elif "rewilding" in lu_verbose:
            lu_decision = "rewilding"

        print(lu_decision)
        return lu_decision, reasoning, lu_verbose

    def get_neighbour_info(self):
        pass
