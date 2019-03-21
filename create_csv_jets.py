import sys
import os
import ROOT as R
from tqdm import tqdm
from collections import defaultdict
from VBSAnalysis.EventIterator import EventIterator

if __name__ == "__main__":
    f = R.TFile(sys.argv[1])
    tree = f.tree

   
    files = defaultdict(list)
    for i in range(3,8):
        legend = []
        for j in range(i):
            legend+= [f"E{j}",f"P{j}",f"Pt{j}",f"Eta{j}"]
        files[i] = ["|".join(legend)]

    with tqdm(total=tree.GetEntries()) as pbar:
        for event in EventIterator(tree):
            nj = min([8, event.njets])
            output_line = []
            for i in range(nj):
                jet = event.jets[i]
                output_line += [
                    str(jet.E()),
                    str(jet.P()),
                    str(jet.Pt()),
                    str(jet.Eta()),                     
                ]
            files[nj].append("|".join(output_line))
            pbar.update()
    
    for key, data in files.items():
        with open(f"output_jets_{key}.csv", "w") as fo:
            fo.write("\n".join(data))
    



