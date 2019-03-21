'''
This macro saves some info about the lhe-fastjet
association '''

import sys 
import ROOT as R 
from tqdm import tqdm
from VBSAnalysis.EventIterator import EventIterator

# Open the Root file
f = R.TFile(sys.argv[1], "UPDATE")

# Define some histo
h_last = R.TH1F("last_jet", "last_jet", 30 , 0, 30 )
h_lastpt = R.TH1F("last_jet_pt", "Pt of last jet", 50, 0, 400)
h_pt = R.TH1F("jetspt_all", "Pt of all jets",100, 0, 2000 )
h_selpt = R.TH1F("jetspt_sel", "Pt of selected jets", 100, 0, 2000)

'''
Cuts on events
------------------------ 
The cuts are applied in the specified order. 
For example you can first filter out jets with Pt under 30 GeV, 
and than require a minimum number of jets. 

The complete list of filters is in the module VBSAnalysis.Utils.EventFilters.
A cut is specified by the name of the filter function and the parameter 
inside a tuple: ("filter_function", param)
'''
cuts = [
    ("min_njets", 3)
]

# tqdm module is used to create a progress bar
with tqdm(total=f.tree.GetEntries()) as pbar:

    # get the events from EventIterator, passing the root file, 
    # the desidered list of cuts and the parameters pairing=True
    # to get info about jet/parton pairing
    for event in EventIterator(f, cuts, pairing=True):
        last = max(event.pairs)
        h_last.Fill(last)
        h_lastpt.Fill(event.jets[last].Pt())
        for j in event.jets:
            h_pt.Fill(j.Pt()) 
        #event.paired_jets returns only the lhe paired jets
        for k in event.paired_jets:
            h_selpt.Fill(k.Pt())
        pbar.update()
    
h_last.Write()
h_lastpt.Write()
h_pt.Write()
h_selpt.Write()
f.Close()