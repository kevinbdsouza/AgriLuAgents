from market import Market
import random
from gekko import GEKKO
import numpy as np


class Compute:
    def __init__(self):
        self.market = Market()
        self.lus = ["food crops", "bioenergy crops", "afforestation", "rewilding", "csa food", "csa bio"]
        self.lu_details = ["agriculture for food crops", "agriculture for bioenergy crops",
                           "afforestation", "rewilding", "climate smart agriculture for food crops",
                           "climate smart agriculture for bioenergy crops"]
        self.factor_demands = {"food crops": {"labour": 1, "technology": self.r_ch(2, 3)},
                               "bioenergy crops": {"labour": 1, "technology": self.r_ch(2, 3)},
                               "afforestation": {"carbon": {"labour": 1, "technology": self.r_ch(2, 3)},
                                                 "timber": {"labour": 4, "technology": self.r_ch(2, 4)}},
                               "rewilding": {"labour": 1, "technology": 1},
                               "csa food": {"labour": 2, "technology": self.r_ch(2, 4)},
                               "csa bio": {"labour": 2, "technology": self.r_ch(2, 4)}}  # in units/tons/hectare
        self.max_yields_hectare = {"food crops": 4.4, "bioenergy crops": 8.7,
                                   "afforestation": {"carbon": 2, "timber": 2},
                                   "csa food": 4.4, "csa bio": 8.7, "rewilding": 0.5}  # per hectare
        self.lq_factor = {"poor": 0.2, "satisfactory": 0.4, "good": 0.6, "very good": 0.8, "excellent": 1}
        self.costs = {"labour_price": 20000, "technology_price": 5000, "raw_materials": 20, "maintenance": 200,
                      "other_costs": 1000, "dormancy_costs": 100}
        self.changing_costs = {"food crops": 0, "bioenergy crops": 0,
                               "afforestation": {"rewilding": 0, "others": 500}, "rewilding": 500,
                               "csa food": 0, "csa bio": 0}
        self.interest_rate = 4
        self.inflation_rate = 0.02
        self.savings_percent = 0.6
        self.labour_tech_capacity = {"food crops": {}, "bioenergy crops": {},
                                     "afforestation": {"carbon": {}, "timber": {}}, "rewilding": {},
                                     "csa food": {}, "csa bio": {}}
        self.labour_scale_factor = 50
        self.tech_scale_factor = 100
        self.income_dict = {}
        self.pl_dict = {}
        self.annual_cost_dict = {}
        self.init_cost_dict = {}
        self.other_scores = {}

    def r_ch(self, a, b):
        choice = random.choice([a, b])
        return choice

    def get_pl(self, num):
        if num >= 0:
            pl = "profit"
        else:
            pl = "loss"
        return pl

    def get_sign(self, pl):
        if pl == "profit":
            sign = 1
        elif pl == "loss":
            sign = -1
        return sign

    def compute_agent_income(self, farmer_agent):
        farm_details = farmer_agent.farm_details
        sim_year = farmer_agent.sim_year

        lq_factor_crops = self.lq_factor[farm_details['land_quality_crops'][0]]
        lq_factor_trees = self.lq_factor[farm_details['land_quality_trees'][0]]
        size = farm_details["land_size"]
        interest = farm_details["liquid_capital"] * (self.interest_rate / 100)
        compound_inflation = np.power((1 + self.inflation_rate), sim_year - 1)  # should add inflation to market as well
        liquid_capital = farm_details["liquid_capital"]
        labour_scale = max(1, np.ceil(size / self.labour_scale_factor))
        tech_scale = max(1, np.ceil(size / self.tech_scale_factor))

        for lu in self.lus:
            m = GEKKO(remote=False)
            if lu == "afforestation":
                n = GEKKO(remote=False)

                labour_capacity_carbon = m.Var(integer=True, lb=0)
                technology_capacity_carbon = m.Var(integer=True, lb=0)
                labour_capacity_timber = n.Var(integer=True, lb=0)
                technology_capacity_timber = n.Var(integer=True, lb=0)

                changing_cost = farmer_agent.rewild_continuous * self.changing_costs["rewilding"]

                if 1 <= farmer_agent.aff_continuous < 40:
                    labour_capacity_timber = 0
                    technology_capacity_timber = 0
                elif farmer_agent.aff_continuous == 0:
                    labour_capacity_timber = self.factor_demands[lu]["timber"]["labour"]
                    technology_capacity_timber = self.factor_demands[lu]["timber"]["technology"]

                if lu in farmer_agent.inventory_dict:
                    technology_capacity_carbon = farmer_agent.inventory_dict[lu]["lab_tech"]["carbon"]["technology"]
                    self.factor_demands[lu]["carbon"]["technology"] = technology_capacity_carbon

                    technology_capacity_timber = farmer_agent.inventory_dict[lu]["lab_tech"]["timber"]["technology"]
                    self.factor_demands[lu]["timber"]["technology"] = technology_capacity_timber

                    dorm_cost = self.costs["dormancy_costs"] * (sim_year - farmer_agent.inventory_dict[lu]["year"] - 1)
                    if dorm_cost > self.costs["technology_price"]:
                        tech_costs_carbon = self.costs["technology_price"] * technology_capacity_carbon * tech_scale
                        tech_costs_timber = self.costs["technology_price"] * technology_capacity_timber * tech_scale
                        farmer_agent.inventory_dict[lu]["year"] = sim_year
                    else:
                        tech_costs_carbon = (dorm_cost * technology_capacity_carbon * tech_scale)
                        tech_costs_timber = (dorm_cost * technology_capacity_timber * tech_scale)
                else:
                    tech_costs_carbon = self.costs["technology_price"] * technology_capacity_carbon * tech_scale
                    tech_costs_timber = self.costs["technology_price"] * technology_capacity_timber * tech_scale

                prod_capacity_carbon = self.max_yields_hectare[lu]["carbon"] * m.min3(1, labour_capacity_carbon /
                                                                                      self.factor_demands[lu]["carbon"][
                                                                                          "labour"] *
                                                                                      technology_capacity_carbon /
                                                                                      self.factor_demands[lu]["carbon"][
                                                                                          "technology"])
                prod_i_carbon = m.Intermediate(prod_capacity_carbon)
                labour_costs_carbon = self.costs["labour_price"] * labour_capacity_carbon * labour_scale
                cost_carbon = compound_inflation * (labour_costs_carbon + tech_costs_carbon +
                                                    self.costs["maintenance"] *
                                                    technology_capacity_carbon * tech_scale + self.costs[
                                                        "other_costs"] + changing_cost)
                init_cost_carbon = compound_inflation * (tech_costs_carbon + changing_cost)

                if farmer_agent.aff_continuous >= 40:
                    prod_capacity_timber = self.max_yields_hectare[lu]["timber"] * n.min3(1,
                                                                                          labour_capacity_timber /
                                                                                          self.factor_demands[lu][
                                                                                              "timber"][
                                                                                              "labour"] *
                                                                                          technology_capacity_timber /
                                                                                          self.factor_demands[lu][
                                                                                              "timber"][
                                                                                              "technology"])

                    prod_i_timber = n.Intermediate(prod_capacity_timber)

                labour_costs_timber = self.costs["labour_price"] * labour_capacity_timber * labour_scale
                cost_timber = compound_inflation * (labour_costs_timber + tech_costs_timber + self.costs[
                    "maintenance"] * technology_capacity_timber * tech_scale + self.costs["other_costs"])
                init_cost_timber = compound_inflation * tech_costs_timber
            else:
                labour_capacity = m.Var(integer=True, lb=0)
                technology_capacity = m.Var(integer=True, lb=0)

                if lu in farmer_agent.inventory_dict:
                    technology_capacity = farmer_agent.inventory_dict[lu]["lab_tech"]["technology"]
                    self.factor_demands[lu]["technology"] = technology_capacity

                    dorm_cost = self.costs["dormancy_costs"] * (sim_year - farmer_agent.inventory_dict[lu]["year"] - 1)
                    if dorm_cost > self.costs["technology_price"]:
                        tech_costs = self.costs["technology_price"] * technology_capacity * tech_scale
                        farmer_agent.inventory_dict[lu]["year"] = sim_year
                    else:
                        tech_costs = (dorm_cost * technology_capacity * tech_scale)
                else:
                    tech_costs = self.costs["technology_price"] * technology_capacity * tech_scale

                if lu != "rewilding":
                    prod_capacity = self.max_yields_hectare[lu] * m.min3(1,
                                                                         labour_capacity / self.factor_demands[lu][
                                                                             "labour"] *
                                                                         technology_capacity / self.factor_demands[lu][
                                                                             "technology"])
                    prod_i = m.Intermediate(prod_capacity)

                    changing_cost = farmer_agent.aff_continuous * self.changing_costs["afforestation"]["others"]
                else:
                    changing_cost = farmer_agent.aff_continuous * self.changing_costs["afforestation"]["rewilding"]

                labour_costs = self.costs["labour_price"] * labour_capacity * labour_scale
                cost = compound_inflation * (labour_costs + tech_costs + self.costs[
                    "maintenance"] * technology_capacity * tech_scale +
                                             self.costs["other_costs"] + changing_cost)
                init_cost = compound_inflation * (tech_costs + changing_cost)

            if lu not in ["afforestation", "rewilding"]:
                lq_factor = lq_factor_crops
                if (lu == "csa food" and farmer_agent.csa_food_continuous >= 5) or (
                        lu == "csa bio" and farmer_agent.csa_bio_continuous >= 5):
                    lq_factor += random.choice([0.2, 0.4])
                    if lq_factor > 1:
                        lq_factor = 1
            else:
                lq_factor = lq_factor_trees

            if lu == "afforestation":
                self.income_dict[lu] = {}
                self.pl_dict[lu] = {}
                self.init_cost_dict[lu] = {}
                self.annual_cost_dict[lu] = {}

                cost_carbon_i = m.Intermediate(cost_carbon)

                net_cash_carbon = prod_i_carbon * size * lq_factor * (
                        self.market.market_lu_dict[lu]["carbon"] - self.costs[
                    "raw_materials"] / 2) + interest - cost_carbon_i

                netcash_i = m.Intermediate(net_cash_carbon)
                init_cost_carbon_i = m.Intermediate(init_cost_carbon)
                init_cost_timber_i = m.Intermediate(init_cost_timber)
                lc = m.Const(liquid_capital)

                m.Maximize(netcash_i)
                if isinstance(init_cost_carbon + init_cost_timber, float):
                    m.Equations([labour_capacity_carbon >= self.factor_demands[lu]["carbon"]["labour"]])
                else:
                    m.Equations([labour_capacity_carbon >= self.factor_demands[lu]["carbon"]["labour"],
                                 technology_capacity_carbon >= 1,
                                 lc >= init_cost_carbon_i + init_cost_timber_i])
                m.options.SOLVER = 1
                m.solve()

                self.income_dict[lu]["carbon"] = np.abs(netcash_i.value[0])
                self.pl_dict[lu]["carbon"] = self.get_pl(netcash_i.value[0])
                self.init_cost_dict[lu]["carbon"] = init_cost_carbon_i.value[0]
                self.annual_cost_dict[lu]["carbon"] = cost_carbon_i.value[0]
                self.labour_tech_capacity[lu]["carbon"]["labour"] = labour_capacity_carbon.value[0]
                if isinstance(technology_capacity_carbon, float):
                    self.labour_tech_capacity[lu]["carbon"]["technology"] = technology_capacity_carbon
                else:
                    self.labour_tech_capacity[lu]["carbon"]["technology"] = technology_capacity_carbon.value[0]

                if 0 <= farmer_agent.aff_continuous < 40:
                    netcash = 0
                elif farmer_agent.aff_continuous >= 40:
                    cost_timber_i = m.Intermediate(cost_timber)

                    net_cash_timber = prod_i_timber * size * lq_factor * (self.market.market_lu_dict[lu]["timber"] -
                                                                          self.costs[
                                                                              "raw_materials"] / 2) + interest - cost_timber_i
                    netcash_i = n.Intermediate(net_cash_timber)
                    lc = n.Const(liquid_capital)

                    n.Maximize(netcash_i)
                    if isinstance(init_cost_carbon + init_cost_timber, float):
                        n.Equations([labour_capacity_timber >= self.factor_demands[lu]["timber"]["labour"]])
                    else:
                        n.Equations([labour_capacity_timber >= self.factor_demands[lu]["timber"]["labour"],
                                     technology_capacity_timber >= 1,
                                     lc >= init_cost_carbon_i + init_cost_timber_i])
                    n.options.SOLVER = 1
                    n.solve()
                    netcash = netcash_i.value[0]

                self.income_dict[lu]["timber"] = np.abs(netcash)
                self.pl_dict[lu]["timber"] = self.get_pl(netcash)
                self.init_cost_dict[lu]["timber"] = init_cost_timber_i.value[0]
                if 0 <= farmer_agent.aff_continuous < 40:
                    self.labour_tech_capacity[lu]["timber"]["labour"] = labour_capacity_timber
                    self.labour_tech_capacity[lu]["timber"]["technology"] = technology_capacity_timber
                    self.annual_cost_dict[lu]["timber"] = cost_timber
                    self.init_cost_dict[lu]["timber"] = init_cost_timber
                else:
                    self.labour_tech_capacity[lu]["timber"]["labour"] = labour_capacity_timber.value[0]
                    self.labour_tech_capacity[lu]["timber"]["technology"] = technology_capacity_timber.value[0]
                    self.annual_cost_dict[lu]["timber"] = cost_timber_i.value[0]
                    self.init_cost_dict[lu]["timber"] = init_cost_timber_i.value[0]
            else:
                cost_i = m.Intermediate(cost)

                if lu == "rewilding":
                    net_cash = size * lq_factor * (
                            self.max_yields_hectare[lu] * self.market.market_lu_dict[lu]["carbon"] +
                            self.market.market_lu_dict[lu]["land"]) + interest - cost_i
                else:
                    net_cash = prod_i * size * lq_factor * (
                            self.market.market_lu_dict[lu] - self.costs["raw_materials"]) + interest - cost_i

                netcash_i = m.Intermediate(net_cash)
                init_cost_i = m.Intermediate(init_cost)
                lc = m.Const(liquid_capital)

                m.Maximize(netcash_i)
                if isinstance(init_cost, float):
                    m.Equations([labour_capacity >= self.factor_demands[lu]["labour"]])
                else:
                    m.Equations([labour_capacity >= self.factor_demands[lu]["labour"],
                                 technology_capacity >= 1,
                                 lc >= init_cost_i])
                m.options.SOLVER = 1
                m.solve()

                self.labour_tech_capacity[lu]["labour"] = labour_capacity.value[0]
                if isinstance(technology_capacity, float):
                    self.labour_tech_capacity[lu]["technology"] = technology_capacity
                else:
                    self.labour_tech_capacity[lu]["technology"] = technology_capacity.value[0]
                self.income_dict[lu] = np.abs(netcash_i.value[0])
                self.pl_dict[lu] = self.get_pl(netcash_i.value[0])
                self.init_cost_dict[lu] = init_cost_i.value[0]
                self.annual_cost_dict[lu] = cost_i.value[0]

    def get_eco_score(self):
        pass

    def get_carbon_score(self):
        pass

    def get_food_security_score(self):
        pass

    def get_energy_security_score(self):
        pass
