# ASIC centralization analysis

#=========#
# Imports #
#=========#

import pandas as pd
import numpy as np


#===========#
# Set Logic #
#===========#

M = {}  # set of all current miners
A = {}  # set of all current ASIC miners
G = {}  # set of all current GPU (non-ASIC) miners

# A + G = M

NM = {}  # set of non-miners
AA = {}  # set of non-miners who could access ASIC miners (N.B. ACCESS LIMITS)
AG = {}  # set of non-miners who could access GPU miners
AN = {}  # unable to access mining / not used in sim other than as math check

# AA + AG + AN = NM
# M + NM = population of world
# N.B. "nodes" here are individuals with access to financial capital


#===================#
# System Parameters #  // ALL NUMBERS PLACEHOLDER FOR NOW -- RESEARCH THESE
#===================#

# percentages of miners by hardware type
a = 0.25  # percentage of miners with access to ASICs == |A| / |M|
g = 0.75  # percentage of miners with access to GPUs == |G| / |M|
assert a + g == 1

# similarly for non-miners; percentage w/ financial access to hardware types
aa = 0.1  # percentage of non-miners who could access ASICs
ag = 0.3  # percentage of non-miners who could access GPUs
an = 0.6  # percentage of non-miners who could access neither ASICs nor GPUs
assert aa + ag + an == 1



# N.B. hashrates can be indexed rather than real rates
a_hr = 250  # ASIC hashrate - placeholder, not realistic
g_hr = 100  # GPU hashrate

'''
Maybe instead of fixing A, G from the start, have a distribution over participation rates for sampling people from AA, AG and have initial NM == world population
'''

participation_rate_a = 0.05
participation_rate_g = 0.15

# A = {sample(AA, participation_rate_a)}
# G = {sample(AG, participation_rate_g)}

# Minimum buy-in prices (not realistic)
p_a = 11000
p_g = 600

# Maximum assumed capital that an individual would devote to miners
# N.B. these could again be sampled per-node from some distribution
max_cap_a = 100000
max_cap_g = p_a - 1

# Capital Distributions
cd_a = np.linspace(p_a, max_cap_a)
# lay a distribution over these values - left skewed
cd_g = np.linspace(p_g, max_cap_g)
# lay a distribution over these values - possibly uniform

# these seem kinda unnecessary now...
def get_beta(a, b, low, high):
    return np.random.beta(a, b) * (high - low)
def get_uniform(low, high):
    return np.random.uniform(low=low, high=high)

class Miner:
    def __init__(self, **kwargs):
        self.id             = kwargs.get('id', None)
        self.capital        = kwargs.get('capital', None)
        self.access         = kwargs.get('access', None)
        self.participating  = kwargs.get('participating', False)
        self.hardware_units = kwargs.get('hardware_units', 0)
        self.hashpower      = kwargs.get('hashpower', None)
        self.spent          = kwargs.get('spent', 0)

    def __repr__(self):
        infostr = 'Miner \
                   \n======\n \
                   \nID:\t\t{} \
                   \nAccess:\t\t{} \
                   \nCapital:\t{} \
                   \nParticipating:\t{} \
                   \n\nHardware \
                   \n--------\n \
                   \nUnits:\t\t{} \
                   \nHashpower:\t{} \
                   \nCost:\t\t{} \
                   '.format(self.id, self.access, self.capital, \
                            self.participating, self.hardware_units, \
                            self.hashpower, self.spent)
        return infostr

    def purchase_hardware(self, gpu_price, asic_price, allocation=1):
        '''
        Convert capital into funded hardware, based on capital allocation.
        '''
        # could extend this to allow additional funding of gpus with funded remainder, or just simplify to only take 1 price parameter
        assert (allocation > 0) & (allocation <= 1)
        funded = allocation * self.capital
        hw_price = None
        if self.access == 'GPU':
            hw_price = gpu_price
        if self.access == 'ASIC':
            hw_price = asic_price
        # must fund at least one hardware unit if participating
        if funded < hw_price:
            self.hardware_units = 1
            self.spent = hw_price
        else:
            self.hardware_units = funded // hw_price
            self.spent = self.hardware_units * hw_price

    def calculate_hashpower(self, gpu_hashrate, asic_hashrate):
        if self.access == 'GPU':
            self.hashpower = self.hardware_units * gpu_hashrate
        if self.access == 'ASIC':
            self.hashpower = self.hardware_units * asic_hashrate

def sample_initial_pop(aa, ag, an, pop_n):
    '''
    For a specified population size and breakdown by access to miners, samples capital distributions of miners.
    '''
    assert aa + ag + an == 1, 'population percentages must sum to 1'
    aa_n = int(pop_n * aa)
    ag_n = int(pop_n * ag)
    an_n = pop_n - aa_n - ag_n
    AA = set()
    AG = set()
    gid = 1
    for i in range(aa_n):
        # use a beta distribution for now; no real justification for this
        mid = str(gid).zfill(len(str(pop_n)))
        AA.add(Miner(id=mid, access='ASIC', \
                     capital=get_beta(2, 3 ,p_a, max_cap_a)))  # DO NOT HARDCODE THIS
        i += 1
        gid += 1
    for i in range(ag_n):
        mid = str(gid).zfill(len(str(pop_n)))
        AG.add(Miner(id=mid, access='GPU', capital=get_uniform(p_g, p_a)))
        i += 1
        gid += 1
    return list(AA), list(AG)

#============#
# Simulation #
#============#

'''
Simulation might proceeed something like as follows:

- sample capital distributions for all nodes in AA, AG
- sample initially participating miners to get M = A + G
- calculate hashpower per miner based on allocated capital (either use full distribution from step 1, or sample capital commitment within allowable bounds, e.g. a node selected for set A must commit enough capital to buy at least 1 ASIC)
- calculate an HHI-style scale of % of nodes controlling % of hashpower

run this simulation across a grid of parameters for differing distributions / hashrate indices / participation rates and see if there is a way to cleanly visualize the results (can use multiple graphs as needed)
'''

asics, gpus = sample_initial_pop(0.1, 0.3, 0.6, 1000000)
miners = asics + gpus

# purchase hardware and calculate hashpower
for m in miners:
    allocation = np.random.uniform(0,1)  # just to get a number for now
    m.purchase_hardware(p_g, p_a, allocation)
    m.calculate_hashpower(g_hr, a_hr)

# calculate concentration under assumption of all particpating, or sample participating
def get_concentration(miners):
    # miners should be a list of Miner objects
    hps = pd.DataFrame({'hashpower': [m.hashpower for m in miners]}, \
                       index=[m.id for m in miners])
    hps.sort_values(by='hashpower', ascending=False, inplace=True)
    hps['cumpct_hp'] = hps['hashpower'].cumsum() / hps['hashpower'].sum()
    hps['cumpct_m'] = [i / len(hps) for i in list(range(len(hps)))]
    cumpct = hps[['cumpct_m', 'cumpct_hp']]
    cumpct.set_index('cumpct_m', inplace=True)  # maybe
    simple = pd.DataFrame()
    for i in [j/100 for j in list(range(1,100))]:
        simple.loc[i, 'cumpct_hp'] = cumpct.loc[i, 'cumpct_hp']
    return simple

# run get_concentration on various parameterizations / averages across samplings of the same parmeterization i guess, and then combine them into surfaces of mining concentration based on changes in those parameters (average scenario per parameterization makes the most sense, I guess)
