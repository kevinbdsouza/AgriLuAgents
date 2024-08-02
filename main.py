import sys
from gov import GovAgent
import numpy as np

if __name__ == "__main__":
    model_dir = sys.argv[1]
    st_dir = sys.argv[2]
    #model_dir = ""

    gov_agent = GovAgent(model_dir)
    farms = gov_agent.get_all_farms_info()

    sim_years = np.arange(1, 3)
    for sim_year in sim_years:
        for farm_id, farm in farms.items():
            if sim_year == 1:
                farmer_agent = gov_agent.create_farmer_agent(sim_year, farm_id, farm)
            else:
                farmer_agent = farm["agent"]
                farmer_agent = gov_agent.update_farmer_agent(sim_year, farmer_agent)

            farm["agent"] = farmer_agent
