import os
import numpy as np


class Config:
    def __init__(self):
        cwd = os.getcwd()
        #self.data_dir = os.path.join(cwd, "data")
        self.data_dir = os.path.join(cwd, "agri_abm", "data")

        # data
        self.agri_limitations_path = os.path.join(self.data_dir, "cli_land_limitation_for_ag_1M.geojson")
        self.eco_connectivity_path = ""
        self.carbon_sequestration_path = ""
        self.csa_path = ""

        # library
        self.land_quality = [
            {"poor": ["adverse climate", "undesirable soils structure", "low permeability", "erosion", "low fertility",
                      "inundation by streams or lakes", "moisture limitation", "salinity issues", "stoniness issues",
                      "consolidated bedrock limited", "topography limited", "excess water"],
             "satisfactory": ["minor characteristics", "unclassified", "combination of subclasses"],
             "good": "no limitations", "very good": "no limitations", "excellent": "no limitations"}]
        self.traits = {"self": ["self-serving", "altruistic"], "moral": ["conscientious", "unprincipled"],
                       "empathy": ["empathetic", "apathetic"],
                       "open": [["adventurous", "risk-taker", "open-minded"], ["change-averse"]],
                       "group": ["family oriented", "individualistic", "community minded"],
                       "politics": ["conservative", "liberal"],
                       "env": [["nature lover", "strong environmentalist", "sustainability champion"],
                               ["nature neutral person", "climate change denier"]],
                       "school": ["traditional", "modern"], "tech": ["technology averse", "technology adopter"],
                       "adoption": ["early adopter", "trend follower"],
                       "firmness": ["firm", "easily swayed"],
                       "public": ["care about public opinion", "do not care about public opinion"],
                       "incentives": ["use government incentives", "do not use government incentives"],
                       "future": ["prepare for the future", "do not prepare for the future"],
                       "market": ["keep tabs on the market", "do not keep tabs on the market"]}
        self.goals = [["maximize profit", "minimize cost"],
                      ["act sustainably", "preserve environment"], ["take care of family", "explore new avenues"]]
        self.government_policies = ["carbon credits for carbon dioxide sequestration",
                                    "credits for creating ecological connectivity",
                                    "credits for preserving food security",
                                    "partial upfront grant for afforestation",
                                    "tax for carbon emissions from land use",
                                    "small grants for improving land quality"]
        self.managerial_ability = ["poor", "satisfactory", "good", "very good", "excellent"]
        self.farmer_age = np.arange(23, 76)
        self.liquid_capital = np.arange(100000, 1000000, 100000)
        self.base_year = 2024
