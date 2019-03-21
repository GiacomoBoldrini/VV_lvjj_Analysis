import sys
import yaml
import argparse
import ROOT as R
from tqdm import tqdm
from VBSAnalysis.EventIterator import EventIterator
from VBSAnalysis import JetTagging as tag  
from VBSAnalysis.OutputTree import OutputTree
from VBSAnalysis.JER import JEResolution

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="Input file") 
parser.add_argument('-o', '--output', type=str, required=True,  help="Output file")  
parser.add_argument('-it','--input-tree', type=str, required=False, default="tree", help="Input Tree name")
parser.add_argument('-ot','--output-tree', type=str, required=True,  help="Output tree name")
parser.add_argument('-xs','--xs', type=str, required=True, help="sample XS id:xs file")
parser.add_argument('-jr', '--jer', type=str, required=False, help="JER up/down")
parser.add_argument('-bt','--btagging', action="store_true", help="Btagging")
parser.add_argument('-c', '--csv', action="store_true" , help="Export to csv") 
parser.add_argument('-n', '--nevents', type=int, required=False, help="Only nevent")
args = parser.parse_args()

# Load btagging if required
if args.btagging:
    from VBSAnalysis.bTagging import bTagger

# Load the tree
f = R.TFile(args.input)
input_tree = f.Get(args.input_tree)

# Output file and OutputTree
fout = R.TFile(args.output, "UPDATE")
output_tree = OutputTree(args.output_tree)

# Calculate events xs_weight before cuts
xsfile = yaml.load(open(args.xs.split(":")[1]))
xs = xsfile[args.xs.split(":")[0]]
xs_weight = xs / input_tree.GetEntries()
print("Sample xs_weight: ", xs_weight)

cuts = [
    ("pt_min_muon", 20),
    ("eta_max_muon", 2.1),
    ("pt_min_jets", 30),
    ("eta_max_jets", 4.7),
    ("min_njets", 4)
]
# Elaborate only nevents if required
if args.nevents != None:
    cuts.append(("n_events", args.nevents))

# Apply JEResolution if required
if args.jer != None:
    event_iterator = JEResolution.apply_JER(
        EventIterator(f, cuts, pairing=args.btagging,treename=args.input_tree),
        args.jer)
else:
    event_iterator = EventIterator(f, cuts, pairing=args.btagging, 
                                    treename=args.input_tree)

ievent = 0
with tqdm(total=input_tree.GetEntries()) as pbar:
    for event in event_iterator:    
        # tag jets    
        tagged_jets = tag.strategy_mw_mjj(event.jets)
        # bveto weights for evert working point
        bveto_weights = [1.]*3
        if args.btagging:
            # the bveto_weight is calculated as 1-btag-efficienc
            for i,wp in enumerate(["L", "M", "T"]):
                bveto_weights[i] = 1 - bTagger.event_btag_efficiency(event, wp) 

        # Extract variables and write the tree
        output_tree.write_event(event,tagged_jets.vbsjets, tagged_jets.vjets,
                                xs_weight, bveto_weights)
        pbar.update(event.evId - ievent)
        ievent = event.evId

# Write the tree on file
output_tree.write()

# Write csv if required
if args.csv :
    output_tree.to_csv(sys.argv[3], sys.argv[4])

f.Close()
fout.Close()
