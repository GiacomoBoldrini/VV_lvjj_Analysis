from rootpy.vector import LorentzVector
from array import array
    
from .Utils.TreeDriver import *
from .Utils import EventFilters 

def EventIterator(file, criteria, partons=False, pairing=False, treename="tree"):
    ''' This method loads a TreeDriver and apply the 
    selection criteria to generate the requested events. '''
    tree = file.Get(treename)
    tree_pairs = file.Get(treename +"_pairs")
    if partons and not pairing:
        tree_driver = PartonsTreeDriver(tree)
    elif pairing:
        tree_driver = PairingTreeDriver(tree, tree_pairs)
    else:
        tree_driver = BaseTreeDriver(tree)

    current_generator = tree_driver.all()
    # Build the chain of generators
    for f, par in criteria:
        current_generator = getattr(EventFilters, f)(current_generator, par)
    
    return current_generator


def count_events(event_iterator):
    return sum(1 for _ in event_iterator)



