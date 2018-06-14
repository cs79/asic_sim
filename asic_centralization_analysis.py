# ASIC centralization analysis

#=========#
# Imports #
#=========#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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


#=========#
# Classes #
#=========#

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


#===========#
# Functions #
#===========#

def get_beta(a, b, low, high):
    '''
    Helper function used by simulation to sample ASIC miners.
    '''
    return np.random.beta(a, b) * (high - low)

def get_uniform(low, high):
    '''
    Helper function used by simulation to sample GPU miners.
    '''
    return np.random.uniform(low=low, high=high)

# TODO: consider why exactly we are hardcoding the capital distributions here, in the manner they are hardcoded
def sample_initial_pop(aa, ag, an, pop_n, gpu_price, asic_price, max_cap_a):
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
                     capital=get_beta(2, 3, asic_price, max_cap_a)))  # DO NOT HARDCODE a AND b HERE
        i += 1
        gid += 1
    for i in range(ag_n):
        mid = str(gid).zfill(len(str(pop_n)))
        AG.add(Miner(id=mid, access='GPU', \
                     capital=get_uniform(gpu_price, asic_price)))
        i += 1
        gid += 1
    # TODO: make this return a dict w/ ids as keys so I can sim and easily aggregate simulated capital across instances of the same ID (maybe)
    return list(AA), list(AG)

# calculate concentration under assumption of all particpating, or sample participating
# TODO: fix this - does not work for some values
def get_concentration(miners):
    '''
    Calculates cumulative percentage of miners that control cumulative percentage of network hashpower.
    '''
    # miners should be a list of Miner objects
    hps = pd.DataFrame({'hashpower': [m.hashpower for m in miners]}, \
                       index=[m.id for m in miners])
    hps.sort_values(by='hashpower', ascending=False, inplace=True)
    hps['cumpct_hp'] = hps['hashpower'].cumsum() / hps['hashpower'].sum()
    hps['cumpct_m'] = [i / len(hps) for i in list(range(len(hps)))]
    cumpct = hps[['cumpct_m', 'cumpct_hp']]
    #cumpct.set_index('cumpct_m', inplace=True)  # maybe
    simple = pd.DataFrame()
    # approximation rather than relying on calculation of exact pcts
    grid = [i/100 for i in list(range(1,100))] + [1.0]
    j = 0
    for i in range(len(cumpct)):
        pct_m = cumpct['cumpct_m'][i]
        if pct_m < grid[j]:
            i += 1
        else:
            #print('i: {}\n\nj: {}'.format(i, j))
            simple.loc[grid[j], 'cumpct_hp'] = cumpct.loc[cumpct.index[i], 'cumpct_hp']
            j += 1
    #for i in [j/100 for j in list(range(1,100))]:
    #    simple.loc[i, 'cumpct_hp'] = cumpct.loc[i, 'cumpct_hp']
    return simple



# run get_concentration on various parameterizations / averages across samplings of the same parmeterization i guess, and then combine them into surfaces of mining concentration based on changes in those parameters (average scenario per parameterization makes the most sense, I guess)

def run_sim(n=10000, alloc='uniform', avg_result=True, **kwargs):
    '''
    '''
    tr =[]
    gpu_price = kwargs.get('gpu_price')
    asic_price = kwargs.get('asic_price')
    for i in range(n):
        a, g = sample_initial_pop(aa=kwargs.get('aa'), ag=kwargs.get('ag'), \
                                  an=kwargs.get('an'), \
                                  pop_n=kwargs.get('pop_n'), \
                                  gpu_price=gpu_price, asic_price=asic_price, \
                                  max_cap_a=kwargs.get('max_cap_a'))
        if i % 100 == 0:
            print('sampled pop for run {}'.format(i))
        miners = a + g
        for m in miners:
            if alloc == 'uniform':
                alloc = np.random.uniform(0,1)  # change this later
            m.purchase_hardware(gpu_price=gpu_price, asic_price=asic_price, \
                                allocation=alloc)
            m.calculate_hashpower(gpu_hashrate=kwargs.get('gpu_hashrate'), \
                                  asic_hashrate=kwargs.get('asic_hashrate'))
        if i % 100 == 0:
            print('allocated capital to hardware for run {}'.format(i))
        tr.append(get_concentration(miners))
        if i % 100 == 0:
            print('calculated miner concentration for run {}'.format(i))
        i += 1
    if avg_result:
        nm = tr[0].columns[0]
        sims = pd.DataFrame(index=tr[0].index)
        j = 1
        for s in tr:
            sims = sims.join(s.rename(columns=lambda x: x + '_{}'.format(j)), \
                             how='outer')
            j += 1
        sims = sims.mean(1)
        sims.name = nm
        tr = sims.to_frame()
    return tr

# knobs to tune here: aa / ag, a_hr (indexed), p_g, p_a, max_cap_a
def panel_sim(p=[], which=None, **kwargs):
    '''
    '''
    w = ('aas', 'ags', 'a_hrs', 'p_gs', 'p_as', 'mc_as')
    assert which in w, 'which must be one of {}'.format(w)
    assert type(p) == list, 'p must be a list of values to run a panel over'
    assert p, 'p must not be empty'
    # unpack sim kwargs
    n               = kwargs.get('n', None)
    alloc           = kwargs.get('alloc', None)
    avg_result      = True  # forced for this case
    aa              = kwargs.get('aa', None)
    ag              = kwargs.get('ag', None)
    an              = kwargs.get('an', None)
    pop_n           = kwargs.get('pop_n', None)
    gpu_price       = kwargs.get('gpu_price', None)
    asic_price      = kwargs.get('asic_price', None)
    max_cap_a       = kwargs.get('max_cap_a', None)
    gpu_hashrate    = kwargs.get('gpu_hashrate', None)
    asic_hashrate   = kwargs.get('asic_hashrate', None)

    tr = pd.DataFrame()

    # big dumb switch for now until I can figure out a clever way to do this
    if which == 'aas':
        for i in p:
            # keep an value constant (at 0) - eventually factor an out
            aa, ag, an = recalc_pcts(aa=i, an=an)
            sim = run_sim(n=n, alloc=alloc, aa=i, ag=ag, an=an, pop_n=pop_n, \
                          gpu_price=gpu_price, asic_price=asic_price, \
                          max_cap_a=max_cap_a, gpu_hashrate=gpu_hashrate, \
                          asic_hashrate=asic_hashrate)
            tr = tr.join(sim.rename(columns=lambda x: \
                                    x + '_aa_{}'.format(i)), how='outer')
            print('Ran sim for {} - value {}'.format(which, i))
    # etc.
    # TODO: FINISH THIS FUNCTION
    #if which == '':

    return tr

# there is a less dumb way to do this, but this should work OK
def recalc_pcts(aa=None, ag=None, an=None):
    '''
    Recalculate population percentages to keep combined split at 100%.
    '''
    bits = [1 if i is not None else 0 for i in [aa, ag, an]]
    assert sum(bits) < 3
    # check individual cases:
    if sum(bits) == 1:
        if aa is not None:
            assert aa <= 1
            ag = an = (1 - aa) / 2  # split remainder evenly, I guess
            return aa, ag, an
        if ag is not None:
            assert ag <= 1
            aa = an = (1 - ag) / 2
            return aa, ag, an
        if an is not None:
            assert an <= 1
            aa = ag = (1 - an) / 2
            return aa, ag, an
    # check double cases
    if sum(bits) == 2:
        if aa is not None and ag is not None:
            assert (aa + ag) <= 1
            an = 1 - (aa + ag)
            return aa, ag, an
        if aa is not None and an is not None:
            assert (aa + an) <= 1
            ag = 1 - (aa + an)
            return aa, ag, an
        if ag is not None and an is not None:
            assert (ag + an) <= 1
            aa = 1 - (ag + an)
            return aa, ag, an

def plot_surface(df, pcts):
    '''
    Plot the output of a panel simulation in 3D.
    '''
    surf = pd.DataFrame()
    for p in pcts:
        col = [col for col in df if col.endswith(str(p))][0]
        temp = df[[col]]
        temp['x'] = temp.index
        temp['y'] = p
        temp.rename(columns={col: 'z'}, inplace=True)
        temp.index = range(len(temp))
        surf = surf.append(temp)
    surf.index = range(len(surf))

    fig = plt.figure()
    ax = Axes3D(fig)
    plot = ax.plot_trisurf(surf['x'], surf['y'], surf['z'], cmap=cm.jet, \
                           linewidth=0.2)
    return plot


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

#testsim = run_sim(n=1000, aa=0.1, ag=0.3, an=0.6, pop_n=100000, gpu_price=p_g, asic_price=p_a, max_cap_a=max_cap_a, gpu_hashrate=g_hr, asic_hashrate=a_hr)

#testpanel = panel_sim(p=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25], which='aas', n=1000, alloc='uniform', aa=0.1, ag=0.3, an=0.6, pop_n=100000, gpu_price=p_g, asic_price=p_a, max_cap_a=max_cap_a, gpu_hashrate=g_hr, asic_hashrate=a_hr)

pcts = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

testpanel = panel_sim(p=pcts, which='aas', n=500, alloc='uniform', aa=0, ag=1, an=0, pop_n=10000, gpu_price=p_g, asic_price=p_a, max_cap_a=max_cap_a, gpu_hashrate=g_hr, asic_hashrate=a_hr)

testpanel.to_csv('C:/Users/cloud/Desktop/testpanel_AA.csv')

plot_surface(testpanel, pcts=pcts)

# what this seems to show is that if NOBODY has an ASIC, decentralization is "better" (whatever that means), but if ANYONE has an ASIC, decentralization "improves" the more that other miners also have them
