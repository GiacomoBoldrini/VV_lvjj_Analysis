'''
This script associate jets to partons in each event. 
The jet radius is used as a parameter for the association 
algorithm. 
'''

import sys
import argparse
import os
import ROOT as R
from VBSAnalysis import PartonJetAssociation as partonjet 

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="Input file") 
parser.add_argument('-t', '--tree', type=str, required=False, default="tree", help="Input tree name")
parser.add_argument('-r', '--jet-radius', type=float, required=True,  help="Jet radius")
args = parser.parse_args()

f = R.TFile(args.input, "UPDATE")
tree = f.Get(args.tree)

params= {
    "max_distance" : args.jet_radius
}

# Associate fastjet with lhe jets
tree_pair = partonjet.associate_parton_to_jets(tree, params)
tree_pair.Write("", R.TObject.kOverwrite)

# Save output file
f.Close()
