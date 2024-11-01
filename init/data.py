import json
from config import Config


class Data:
    def __init__(self):
        self.cfg = Config()
        self.agri_limitations_data = self._get_agri_data()
        self.eco_connectivity_data = self._get_eco_connectivity_data()
        self.carbon_sequestration_potential = self._get_carbon_sequestration_potential()
        self.csa_potential = self._get_csa_potential()

    def _get_agri_data(self):
        with open(self.cfg.agri_limitations_path, 'r') as j:
            agri_limitations_data = json.loads(j.read())
        return agri_limitations_data

    def _get_eco_connectivity_data(self):
        eco_connectivity_data = ""
        return eco_connectivity_data

    def _get_carbon_sequestration_potential(self):
        carbon_sequestration_potential = ""
        return carbon_sequestration_potential

    def _get_csa_potential(self):
        csa_potential = ""
        return csa_potential
