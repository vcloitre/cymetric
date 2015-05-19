#############################################################################################
# What could be interesting is creating different sets of parameters, corresponding to      #
# different policies, that we could choose. That is the purpose of seperating the paramters #
# depending on the level (region, institution, facility). It will be possible to select     #
# one (or even several ?) options.							    #
# 											    #
# Need to add specific inflation rates 							    #
#############################################################################################

########################
# Financial Parameters #
########################

# see d'Haeseleer (2013 data)

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

# Facility level
capital_cost = # $/MW or maybe overnight cap cost
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
fuel_price = {} # dict with prices for different reactors (pwr, phwr, bwr, fr... differentiate reenriched uox with uox using natural uranium)
# a function could give the price of uranium as a function of availability (see Arnaud's work)
# "The back-end cost elements include the interim storage facilities, construction of reprocessing facilities, SNF encapsulation and final disposal" (d'Haeseleer)


#####################
# Power plant costs #
#####################
