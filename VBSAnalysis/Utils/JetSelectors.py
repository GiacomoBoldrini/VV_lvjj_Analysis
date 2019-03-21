from operator import attrgetter, itemgetter
from itertools import combinations
import math



def max_deltaeta_pair(jets):
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], abs(jets[i].Eta() - jets[k].Eta())))
    l = sorted(l, key=itemgetter(1), reverse=True)
    return l[0][0]

def min_deltaeta_pair(jets):
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], abs(jets[i].Eta() - jets[k].Eta())))
    l = sorted(l, key=itemgetter(1), reverse=False)
    return l[0][0]

def max_mjj_pair(jets):
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], (jets[i]+ jets[k]).M() ))
    l = sorted(l, key=itemgetter(1), reverse=True)
    return l[0][0]

def max_pt_pair(jets):
    ''' Returns the pair with highest Pt'''
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append(( [i,k], (jets[i]+ jets[k]).Pt() ))
    l = sorted(l, key=itemgetter(1), reverse=True)
    return l[0][0]

def nearest_W_pair(jets): 
    ''' Returns the pair of jets with Mjj nearest
    to Mw '''
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append(([i,k], abs(80 - (jets[i]+ jets[k]).M() )))
    l = sorted(l, key=itemgetter(1))
    return l[0][0]

#NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW 
#NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW 
def nearest_Z_pair(jets): 
    ''' Returns the pair of jets with Mjj nearest
    to Mw '''
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append(([i,k], abs(91 - (jets[i]+ jets[k]).M() )))
    l = sorted(l, key=itemgetter(1))
    return l[0][0]

def nearest_W_jet(jets):
    ''' Returns the signle jets with Mjj nearest to Mw '''
    return sorted(enumerate(jets), key=lambda x: x[1].M(), reverse=True)[0]

def nearest_Z_or_W (pts):
    
    def nearest_Z_partons ( pts):
        ''' Returns the pair of jets with Mjj nearest
            to Mw '''
        l = []
        for i ,k  in combinations(range(len(pts)),2):
            l.append(([i,k], abs(91 - (pts[i]+ pts[k]).M() )))
        l = sorted(l, key=itemgetter(1))
        return l[0]
    
    def nearest_W_partons(pts): 
        ''' Returns the pair of jets with Mjj nearest
        to Mw '''
        l = []
        for i ,k  in combinations(range(len(pts)),2):
            l.append(([i,k], abs(80 - (pts[i]+ pts[k]).M() )))
        l = sorted(l, key=itemgetter(1))
        return l[0]
    
    z = nearest_Z_partons(pts)
    w = nearest_W_partons(pts)
    if z[1] > w[1]:
        return w[0]
    else:
        return z[0]
    

def tag_parton_pair(jets):
    ''' Returns Tag Partons '''
    c = [0,1,2,3]
    w_pair = nearest_Z_or_W(jets)
    #always remove one because the second element on mjj_pair
    # is after the first one
    c.pop(w_pair[0])
    c.pop(w_pair[1]-1)
    # W jet by closest mass to W
    # Return the result
    return c

#NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW 
#NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW 

def deltaeta_mjj_pair(jets):
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], abs(jets[i].Eta() - jets[k].Eta()), (jets[i]+ jets[k]).M()))
    l = sorted(l, key=itemgetter(1), reverse=True)
    l1 = []
    for i ,k  in combinations(range(len(jets)),2):
        l1.append( ([i,k], (jets[i]+ jets[k]).M(), abs(jets[i].Eta() - jets[k].Eta()) ))
    l1 = sorted(l1, key=itemgetter(1), reverse=True)
    if l[0][0]==l1[0][0]:
        return l[0][0]
    else:
        if (1+3/abs(l[0][1])+ l[0][2]/1000) > (l1[0][1]/1000 + (1+3/abs(l1[0][1]))):
            return l[0][0]
        else:
            return l1[0][0]

#questo fa schifo
def deltaeta_mw_pair(jets):
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], abs(jets[i].Eta() - jets[k].Eta()), abs(80 - (jets[i]+ jets[k]).M() ) ))
    l = sorted(l, key=itemgetter(1), reverse=False)
    l1 = []
    for i ,k  in combinations(range(len(jets)),2):
        l1.append(([i,k], abs(80 - (jets[i]+ jets[k]).M() )))
    l1 = sorted(l1, key=itemgetter(1))
    if l[0][0]==l1[0][0]:
        return l[0][0]
    else:
        if abs(l[0][1]) < l1[0][1]:
            return l1[0][0]
        else:
            return l[0][0]

#altro tentativo di definire meglio mw
def deltaeta_mw2_pair(jets):
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], abs(jets[i].Eta() - jets[k].Eta()), abs(80 - (jets[i]+ jets[k]).M() ) ))
    l = sorted(l, key=itemgetter(1), reverse=False)
    l1 = []
    for i ,k  in combinations(range(len(jets)),2):
        l1.append(([i,k], abs(80 - (jets[i]+ jets[k]).M() ), abs(jets[i].Eta() - jets[k].Eta())))
    l1 = sorted(l1, key=itemgetter(1))
    if l[0][0]==l1[0][0]:
        return l[0][0]
    else:
        #cerco di dare un peso alle due variabili sommando le loro distanze dal punto ideale
        if (l[0][1]+l[0][2]) > (l1[0][1]+l1[0][2]) :
            return l1[0][0]
        else:
            return l[0][0]
        
#Tentativo WZ e min deltaeta

def mindeltaeta_wz_pair(jets):
    w = []
    for i ,k  in combinations(range(len(jets)),2):
        w.append( ([i,k], abs(jets[i].Eta() - jets[k].Eta()), abs(80 - (jets[i]+ jets[k]).M() ) ))
    w = sorted(w, key=itemgetter(1), reverse=False)
    z = []
    for i ,k  in combinations(range(len(jets)),2):
        z.append( ([i,k], abs(jets[i].Eta() - jets[k].Eta()), abs(91 - (jets[i]+ jets[k]).M()) ))
    z = sorted(z, key=itemgetter(1), reverse=False)
    if (z[0][1]+z[0][2]) > (w[0][1]+w[0][2]):
        return w[0][0]
    else:
        return z[0][0]
        
    """
    l1 = []
    for i ,k  in combinations(range(len(jets)),2):
        l1.append(([i,k], abs(80 - (jets[i]+ jets[k]).M() ), abs(jets[i].Eta() - jets[k].Eta())))
    l1 = sorted(l1, key=itemgetter(1))
    if l[0][0]==l1[0][0]:
        return l[0][0]
    else:
        #cerco di dare un peso alle due variabili sommando le loro distanze dal punto ideale
        if (l[0][1]+l[0][2]) > (l1[0][1]+l1[0][2]) :
            return l1[0][0]
        else:
            return l[0][0]
    """
#combining everything attempt! molto alta selezione con /4000
def combined_pair(jets):
    #list sorted for mindeltaeta
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], abs(jets[i].Eta() - jets[k].Eta()), abs(80 - (jets[i]+ jets[k]).M()), jets[i].DeltaR(jets[k])  ))
    l = sorted(l, key=itemgetter(1), reverse=False)
    #list sorted for nearest W
    l1 = []
    for i ,k  in combinations(range(len(jets)),2):
        l1.append(([i,k], abs(80 - (jets[i]+ jets[k]).M() ), abs(jets[i].Eta() - jets[k].Eta()), jets[i].DeltaR(jets[k])))
    l1 = sorted(l1, key=itemgetter(1))
    #list sorted for max pt
    l2 = []
    for i ,k  in combinations(range(len(jets)),2):
        l2.append(( [i,k], jets[i].DeltaR(jets[k]), abs(80 - (jets[i]+ jets[k]).M()), abs(jets[i].Eta() - jets[k].Eta()) ))
    l2 = sorted(l2, key=itemgetter(1), reverse=True)
    
    if l[0][0]==l1[0][0]==l2[0][0]:
        return l[0][0]
    else:
        c = []
        c.append( (l[0][0], l[0][1]+l[0][2]+l[0][3]) )
        c.append( (l1[0][0], l1[0][1]+l1[0][2]+l1[0][3]) )
        c.append( (l2[0][0], l2[0][1]+l2[0][2]+l2[0][3]) )
        c = sorted(c, key= itemgetter(1), reverse=False)
        return c[0][0]

#Jet size selection

def jet_min_size_pair(jets):
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], jets[i].DeltaR(jets[k])))
    l = sorted(l, key=itemgetter(1))
    return l[0][0]

def jet_max_size_pair(jets):
    l = []
    for i ,k  in combinations(range(len(jets)),2):
        l.append( ([i,k], jets[i].DeltaR(jets[k])))
    l = sorted(l, key=itemgetter(1), reverse = True)
    return l[0][0]
