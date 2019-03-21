''' This file contains some generator functions 
to filter the events '''

from itertools import islice, combinations
from operator import itemgetter

##########################################
# Produce a limited number of events
def n_events(ev_iter, n):
    return islice(ev_iter, n)

def after_n_events(ev_iter, n):
    return islice(ev_iter, n, None)

#########################################
# Cuts on muon 

def eta_max_muon(ev_iter, etamax):
    return filter(lambda e: abs(e.muon.Eta()) <= etamax, ev_iter)

def pt_min_muon(ev_iter, ptmin):
    return filter(lambda e: e.muon.Pt() >= ptmin, ev_iter)

########################################
# Filters on number of jets

def min_njets(ev_iter, njets):
    ''' Events with more than njets'''
    return filter(lambda e: e.njets >= njets, ev_iter)

def max_njets(ev_iter, njets):
    ''' Events with less than njets'''
    return filter(lambda e: e.njets <= njets, ev_iter)

def eq_njets(ev_iter, njets):
    ''' Events with njets'''
    return filter(lambda e: e.njets == njets, ev_iter)

###########################################à
# Cuts over jets, they MODIFY the event

def atleastone_mjj_M(ev_iter, M):
    #We add the condition that the event has at least one pair 
    #of jets with invariant mass > M GeV
    for event in ev_iter:
        l = []
        for i ,k  in combinations(range(len(event.jets)),2):
            l.append( ([i,k], (event.jets[i]+ event.jets[k]).M() ))
        l = sorted(l, key=itemgetter(1), reverse=True)
        if l[0][1] > M:
            yield event

def pt_min_jets(ev_iter, ptmin):
    #Knowing that jets are ordered by Pt we can use a early
    # exit while
    for event in ev_iter:
        icut = 0
        jetsiter = iter(event.jets)
        while(icut < event.njets ):
            jet = next(jetsiter)
            if jet.Pt() < ptmin:
                break
            icut+=1
        # Cut the list of jets in the event object
        event.jets = event.jets[:icut]
        yield event

def eta_max_jets(ev_iter, etamax):
    #We have to cycle on all the jets
    for event in ev_iter:
        okjets = []
        for j in event.jets:
            if abs(j.Eta()) <= etamax:
                okjets.append(j)
        # Cut the list of jets in the event object
        event.jets = okjets
        yield event
    

#################################################################
# Pairing flags
def gt_flag(e_iterator, flag ):
    ''' Events with flag greater than parameter '''
    return filter(lambda e: e.flag >= flag and e.pass_jets_cuts, e_iterator)

def eq_flag(e_iterator, flag):
    ''' Events with flag equal to parameter.
    The filter checks if the paired jets are still not cut by other filters in the event
    '''
    return filter(lambda e: e.flag == flag and e.pass_jets_cuts, e_iterator)

def eq_flags(e_iterator, flags):
    return filter(lambda e: e.flag in flags and e.pass_jets_cuts, e_iterator)

