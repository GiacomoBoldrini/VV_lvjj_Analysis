'''
This script check the charateristic of the 
jets with multiple lhe jets assigned'''
import sys
import ROOT as R
from tqdm import tqdm
from VBSAnalysis.EventIterator import EventIterator

f = R.TFile(sys.argv[1], "UPDATE")

hm = R.TH1D("mjet", "M overlap jet", 50, 0, 200)
hpt = R.TH1D("pt", "Pt overlap jet", 30, 0 , 1400)
heta = R.TH1D("eta", "eta overlap jet", 30, 0, 6)
pt_partons = R.TH2D("pt_partons", "Pt overlap partons", 
        60, 0, 200, 20, 1, 10)

# Here we check first the pairing flag, than the number of jets
cuts = [
    ("eq_flag", 1),
    ("min_njets", 3)
]

jevent = 0 
with tqdm(total=f.tree.GetEntries()) as pbar:
    for event in EventIterator(f, cuts, pairing=True):
        #we have to find the double selected jet
        jets = []
        partons = []
        # remember that the pairs array contains the association
        # parton: id jets (in pt order)
        for i in range(len(event.pairs)-1):
            for j in range(i+1, len(event.pairs)):
                # check if the jet is the same for two partons
                if event.pairs[i] == event.pairs[j]:
                    partons += event.get_partons([i,j])
                    jets.append(event.jets[event.pairs[i]])
        hm.Fill(jets[0].M())
        hpt.Fill(jets[0].Pt())
        heta.Fill(jets[0].Eta())
        #Filling pt_partons with the M and ration of Pt
        #of the partons
        pt_partons.Fill(jets[0].M(), partons[0].Pt() / partons[1].Pt())
        pbar.update(event.evId - jevent)
        jevent = event.evId

hm.Write()
hpt.Write()
heta.Write()
pt_partons.Write()
f.Close()
