from .Utils import JetSelectors as jsel
from collections import namedtuple

TaggedJets = namedtuple("TaggedJets", ["vbsjets", "vjets"])
TaggedPartons = namedtuple("TaggedPartons", ["vbsjets", "vjets", "vbs_pair", "w_pair"])

#CHECK PER ASSOCIAZIONE PARTONI GETTI

def check_association(event, partons_ids, jets):
    '''This function check the event pairings between
    partons and jets. It requires the partons indexes and a list of 
    jet objects'''
    pairs = []
    for i in partons_ids:
        ass_jet = event.paired_jet(i)
        ok = False
        for j in jets:
            if j.IsEqual(ass_jet):
                ok = True
                break
        if not ok: 
            return False
    return True

#STRATEGIE PER PARTONI IN BASE A NEAREST W, NEAREST Z, OPPURE NEAREST W O Z, RESTITUISCONO 
#INDICI E QUADRIMOMENTI

def strategy_partons(pts):
    c = [0,1,2,3]
    partons = pts.copy()  #duplicate the jets
    w_pair = jsel.nearest_W_pair(partons)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vpartons = [ partons.pop(w_pair[0]), partons.pop(w_pair[1]-1)]
    c.pop(w_pair[0])
    c.pop(w_pair[1]-1)
    # W jet by closest mass to W
    vbs_pair = c
    vbspartons = [ partons[0], partons[1] ]
    # Return the result
    return TaggedPartons(vbspartons, vpartons, vbs_pair, w_pair)

def strategy_partons1(pts):
    c = [0,1,2,3]
    partons = pts.copy()  #duplicate the jets
    w_pair = jsel.nearest_Z_pair(partons)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vpartons = [ partons.pop(w_pair[0]), partons.pop(w_pair[1]-1)]
    c.pop(w_pair[0])
    c.pop(w_pair[1]-1)
    # W jet by closest mass to W
    vbs_pair = c
    vbspartons = [ partons[0], partons[1] ]
    # Return the result
    return TaggedPartons(vbspartons, vpartons, vbs_pair, w_pair)

def strategy_partons2(pts):
    c = [0,1,2,3]
    partons = pts.copy()  #duplicate the jets
    w_pair = jsel.nearest_Z_or_W(partons)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vpartons = [ partons.pop(w_pair[0]), partons.pop(w_pair[1]-1)]
    c.pop(w_pair[0])
    c.pop(w_pair[1]-1)
    # W jet by closest mass to W
    vbs_pair = c
    vbspartons = [ partons[0], partons[1] ]
    # Return the result
    return TaggedPartons(vbspartons, vpartons, vbs_pair, w_pair)



#DA QUI IN POI STRATEGIE PER I GETTI,
#RESTITUISCONO SOLO I QUADRIVETTORI E NON PIU GLI INDICI

def strategy_wz_mjj(jets):
    jets = jets.copy()  #duplicate the jets
    w_pair = jsel.nearest_Z_or_W(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # W jet by closest mass to W
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_wz(jets):
    jets = jets.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # W jet by closest mass to W
    w_pair = jsel.nearest_Z_or_W(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_mz(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # W jet by closest mass to W
    w_pair = jsel.nearest_Z_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mz_mjj(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.nearest_Z_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # W jet by closest mass to W
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_mw(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # W jet by closest mass to W
    w_pair = jsel.nearest_W_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mw_mjj(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.nearest_W_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # W jet by closest mass to W
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mw_deltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.nearest_W_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_deltaeta_mw(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.nearest_W_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxpt_mjj(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.max_pt_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_maxpt(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.max_pt_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxpt_deltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.max_pt_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_deltaeta_maxpt(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.max_pt_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mindeltaeta_mjj(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.min_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_mindeltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.min_deltaeta_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxdeltaeta_mindeltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.min_deltaeta_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mindeltaeta_maxdeltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.min_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_Weta_maxeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.deltaeta_mw2_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_Weta_mjj(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.deltaeta_mw2_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_Weta(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.deltaeta_mw2_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxeta_Weta(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.deltaeta_mw2_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_proviamo_mjj(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.combined_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_proviamo(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.combined_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_deltaeta_proviamo(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.combined_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_proviamo_deltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.combined_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

# MAX SIZE AND MINSIZE 

def strategy_minsize_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.jet_min_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_minsize(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.jet_min_size_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_minsize_mjj(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.jet_min_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_minsize(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.jet_min_size_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_minsize_maxdeltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.jet_min_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxdeltaeta_minsize(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.jet_min_size_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_W(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.nearest_W_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_W_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.nearest_W_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_Z(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.nearest_Z_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_Z_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.nearest_Z_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_maxpt(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.max_pt_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxpt_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.max_pt_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_WZ(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.nearest_Z_or_W(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_WZ_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.nearest_Z_or_W(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_mindeltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.min_deltaeta_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mindeltaeta_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.min_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

#MANCANTI


def strategy_proviamo_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.combined_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_proviamo(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.combined_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_Weta_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.deltaeta_mw2_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_Weta(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.deltaeta_mw2_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxdeltaeta_Z(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.nearest_Z_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_Z_maxdeltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.nearest_Z_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxdeltaeta_WZ(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.nearest_Z_or_W(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_WZ_maxdeltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.nearest_Z_or_W(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

#NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW 
def strategy_minetaWZ_mjj(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.mindeltaeta_wz_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_mjj_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_mjj_minetaWZ(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_mjj_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.mindeltaeta_wz_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_minetaWZ_maxdeltaeta(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.mindeltaeta_wz_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.max_deltaeta_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxdeltaeta_minetaWZ(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.max_deltaeta_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.mindeltaeta_wz_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_minetaWZ_maxsize(jts):
    jets = jts.copy()  #duplicate the jets
    w_pair = jsel.mindeltaeta_wz_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    #selected max deltaeta pair
    vbs_pair = jsel.jet_max_size_pair(jets)
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)

def strategy_maxsize_minetaWZ(jts):
    jets = jts.copy()  #duplicate the jets
    vbs_pair = jsel.jet_max_size_pair(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    vbsjets = [ jets.pop(vbs_pair[0]), jets.pop(vbs_pair[1]-1)]
    #selected max deltaeta pair
    w_pair = jsel.mindeltaeta_wz_pair(jets)
    vjets = [ jets.pop(w_pair[0]), jets.pop(w_pair[1]-1)]
    # Return the result
    return TaggedJets(vbsjets, vjets)
