import sys
import argparse
import ROOT as R
from tqdm import tqdm
from VBSAnalysis.EventIterator import EventIterator, count_events
from VBSAnalysis import JetTagging as tag  
from VBSAnalysis.TreeWriter import TreeWriter
from VBSAnalysis.JER import JEResolution

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="Input file")  
parser.add_argument('-t', '--tree-name', type=str, required=True,  help="Output tree names")
args = parser.parse_args()

f = R.TFile(args.input)

cuts = [
    ("pt_min_jets", 30),
    ("eta_max_jets", 4.7),
    ("min_njets", 4)
]


event_iterator = EventIterator(f, cuts)
nevents = 0
ievent = 0
with tqdm(total=f.tree.GetEntries()) as pbar:
    for event in event_iterator:        
        nevents+=1
        pbar.update(event.evId - ievent)
        ievent = event.evId

print("Selected events: {}".format(nevents))

f.Close()
