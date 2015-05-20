###############################################################################
# What could be interesting is creating different sets of parameters, corres-
# ponding to different policies, that we could choose. That is the purpose of 
# seperating the paramters depending on the level (region, institution, facili-
# ty). It will be possible to select one (or even several ?) options.							    #
# Need to add specific inflation rates
###############################################################################

import pandas as pd

########################
# Financial Parameters #
########################

# see d'Haeseleer (2013 data)
"""
# Region level
tax_rate = # %
depreciation_schedule = # depreciation type
depreciation_time = # years
external_cost = # $/MWh (e.g. CO2)
CO2_price = # $/tonCO2
ini_BC = # $/MWh initial busbar cost
delta_BC = # $/MWh increment of the BC in the loop

# Institution level
fixedOM = # $/MWh-yr
variableOM = # $/MWh
"""
# Facility level
overnight = 5000 # $/MW cap or maybe overnight cap cost
"""
construction_time = # years
decommissioning_cost = # $/MW
decommissioning_time = # years

##############
# Fuel costs #
##############

# see d'Haeseleer (2013 data)

# mining
u_ore_price = 
# processing
yellow_cake_price = 130 # $/kg
# conversion from U308 to UF6
conversion_cost = 9 # $/kg
# enrichment
swu_cost = 140 # $/swu
# Example : for 4.95 % enrichment, fuel cost 300 $/kg
# fabrication
## reconversion
UO2_cost =
## fuel reprocessing
Pu_price =
reprocessed_ur =
# dict with prices for different reactors (pwr, phwr, bwr, fr... differentiate 
# reenriched uox with uox using natural uranium)
fuel_price = {} 
# a function could give the price of uranium as a function of availability (see 
# Arnaud's work)
# "The back-end cost elements include the interim storage facilities, 
# construction of reprocessing facilities, SNF encapsulation and final 
# disposal" (d'Haeseleer)
"""

#####################
# Power plant costs #
#####################

shapes = ["linear"]

def capital_shape(t0, duration, shape):
    """Returns the values corresponding to annual costs corresponding to  
    capital cots (per kW). Duration could be optional, and then one thousand 
    simulation with random durations could be calculated. An important feature 
    is the possibility to change both the date before commissioning when the 
    paiement starts and the duration (increases costs, for the future, an could
    be to make the the price variable, as a function of the duration).
    """
    if not isinstance(t0, int):
        raise Exception("Year begin for paiement must be an integer")
    if t0 < 0:
        raise Exception("Year begin for paiement must be positive")
    if not isinstance(duration, int):
        raise Exception("Duration of paiement must be an integer")
    if duration < 0:
        raise Exception("Duration of paiement must be positive")            
    if "LINEAR" in shape.upper():
        step1 = pd.Series(list(range(t0))).apply(lambda x: 2/(t0*duration)*x)
        step2 = pd.Series(list(range(duration-t0+1))).apply(lambda x: -2/((
              duration-t0)*duration)*x+2/duration)
        return pd.concat([step1, step2]).reset_index()[0]      
    #else if shape == :   
    else:
        raise Exception("Wrong shape, valid shapes are in the following list : " + str(shapes))