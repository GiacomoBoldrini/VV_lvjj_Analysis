import json
import os
from functools import reduce
from operator import mul

eff = json.load(open(os.path.dirname(__file__) + "/efficiencyMCFileFull2016.json")) 

minpt = 20
maxpt = 300 
mineta = 0
maxeta = 2.4

def calculate_efficiency(flavour, workpoint, pt, eta):
        # fix underflow and overflow of pt
        if pt < minpt:
            pt = minpt
        if pt > maxpt:
            pt = maxpt
        # Check eta limit -> no tag if greater
        if eta>maxeta:
            #print("maxeta")
            return 0
        
        # get the efficiency
        for point in eff[workpoint][flavour] :
            #   pt           eta          eff 
            # (( 0.0, 10.0), (0.0, 1.5), 0.980 ),
            if ( pt  >= point[0][0] and pt  < point[0][1] and
                eta >= point[1][0] and eta < point[1][1] ) :
                #print("flavour = {}, wp = {}, pt= {}, eta={}, eff={}".format(
                #        flavour,workpoint, pt, eta, point[2]))
                return point[2]

        # If not found... it should never happen
        #print (" default ???", pt, eta, flavour)
        return 0


def event_btag_efficiency(event, workpoint):
    # Add a default 0 in case there aren't 
    # associated jets after the cuts
    effs = [0]
    # Get all the pairs of jet/partons not cut by selections
    for pair in event.paired_jets_not_cut:
        # u, d, s, g (9 or 21)
        if pair.flavour in [1,2,3,9,21]:
            flav = "l"
        elif pair.flavour == 4:
            flav = "c"
        elif pair.flavour == 5:
            flav = "b"
        
        effs.append(calculate_efficiency(flav, workpoint, 
                    pair.jet.Pt(), abs(pair.jet.Eta())))

    # extract the final efficiency as
    # 1 - product for each jet( 1 - eff)
    return 1 - reduce (mul, map(lambda e: 1-e, effs))
    
