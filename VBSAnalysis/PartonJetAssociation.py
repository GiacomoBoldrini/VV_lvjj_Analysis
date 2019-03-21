from .EventIterator import EventIterator
from .Utils.TreeDriver import *
from operator import itemgetter
from tqdm import tqdm
import numpy as np
from ROOT import TTree
from array import array
from rootpy.tree import Tree, TreeModel, IntCol, IntArrayCol, FloatArrayCol

class JetPair(TreeModel):
    ''' Model for the Tree of pairs lhe-fastjet jets'''
    npartons = IntCol()
    pairs = IntArrayCol(4, length_name="npartons")
    dist = FloatArrayCol(4, length_name="npartons")
    flag = IntCol()  
    # 0 normal; 1 overlapping jets; 2 non-associated

def associate_vectors(jets, partons, params):
    ''' The params influences the flag of the event:
    0 = OK
    1 = Overlapping partons
    2 = At least one parton not associated 
    '''
    flag = 0
    ntotjets = len(jets)
    ntotpartons = len(partons)
    comb = []
    '''enumerate crea una lista con indice posizione nella lista e valore'''
    for nj, j in enumerate(jets):
        for njr, jr in enumerate(partons):
            '''append aggiunge elementi a una lista già esistente: indice getto, indice partone, distanza angolare partone e getto'''
            comb.append( (nj, njr, j.DrEtaPhi(jr)))
    comb = sorted(comb, key=itemgetter(2))
    results = [[-1]*ntotpartons,[0.]*ntotpartons]
    assigned_part = 0
    for nj, njr, distance  in comb:        
        # the fastjet jet can be reused if the lhe jet
        # is nearer than the max_distance
        if results[0][njr] == -1 and distance <= params["max_distance"]:
                if nj in results[0]:
                    # the jet is already associated with a parton
                    # This is an overlapping parton
                    flag = 1
                results[0][njr] = nj
                results[1][njr] = distance 
                assigned_part+=1
        if assigned_part == ntotpartons:
            break  #early exit when partons are all assigned
    # Check if at least one parton is not associated
    if -1 in results[0]:
        flag = 2
    return results, flag

def associate_parton_to_jets(tree, params):
    # Create tree to save the pairs
    # The name of the pairs tree is the name of the tree + _pairs
    tree_pairs = Tree(tree.GetName() + "_pairs", model=JetPair)
    tree_iterator = PartonsTreeDriver(tree)
    with tqdm(total=tree.GetEntries()) as pbar:
        for event in tree_iterator.all():
            r, flag = associate_vectors(event.jets, 
                        event.partons, params)
            tree_pairs.npartons = event.npartons
            tree_pairs.pairs = r[0]
            tree_pairs.dist = r[1]
            tree_pairs.flag = flag
            tree_pairs.fill()
            pbar.update()
    return tree_pairs
    
        
