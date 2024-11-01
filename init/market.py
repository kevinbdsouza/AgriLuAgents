# crops in $/tonne, carbon in $/tonne, timber in $/m3, rewilding in $/hectare

class Market:
    def __init__(self):
        self.market_lu_dict = {"food crops": 200, "bioenergy crops": 300,
                               "afforestation": {"carbon": 50, "timber": 20000},
                               "rewilding": {"land": 1000, "carbon": 20},
                               "csa food": 200, "csa bio": 300}
        self.labour_price = 0
        self.technology_price = 0

    def update_market(self):
        pass
