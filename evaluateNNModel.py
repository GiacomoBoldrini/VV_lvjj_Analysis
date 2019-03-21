import sys
from array import array
import argparse
from tqdm import tqdm
import ROOT as R
import numpy as np
from VBSAnalysis.MVAModel import Model
import uproot
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True,
                    help="Root file")
parser.add_argument('-t', '--tree', type=str, required=False,
                    default="mw_mjj", help="Tree name")
parser.add_argument('-m', '--model-config', type=str, required=True,
                    help="Model configuration file")  
parser.add_argument('-b', '--branchname', type=str, required=False, 
                    default="score", help="Output branch name")
parser.add_argument('-bs', '--batch-size', type=int, required=False, default=4096,
                    help="Batch size") 
parser.add_argument("-nt", '--new-tree', action="store_true",
                    help="Not overwrite but create new tree")
parser.add_argument("-ss", "--score-size", type=int, required=False,
                    default=1, help="Size of the score vector")
args =parser.parse_args()

model = Model(args.model_config)

tfile = R.TFile(args.file, "update")
data_tree = tfile.Get(args.tree)

tree = uproot.open(args.file)[args.tree]

score = array("f", [0.]*args.score_size)
if args.score_size == 1:
    score_branch = data_tree.Branch(args.branchname, score, args.branchname+"/F")
else:
    score_branch = data_tree.Branch(args.branchname, score, 
                            "{0}[{1}]/F".format(args.branchname, args.score_size))
i = 0

with tqdm(total=data_tree.GetEntries()) as pbar:
    # Get variables from the model
    for events in tree.iterate(model.variables, 
            entrysteps=args.batch_size, outputtype=pd.DataFrame):
        scores = model.evaluate(events, args.batch_size) 
        for s in scores:
            data_tree.GetEntry(i)
            for i,v in enumerate(s):
                score[i] = v
            score_branch.Fill()
            i+=1
            pbar.update()
   

if not args.new_tree:
    data_tree.Write("",R.TObject.kOverwrite)
else:
    data_tree.Write()
    
tfile.Close()
