import ROOT as r
import numpy as np
import itertools
import sys
from tqdm import tqdm
from array import array
import argparse
from VBSAnalysis.EventIterator import EventIterator
import math 
from operator import attrgetter, itemgetter
import matplotlib.pyplot as plt 
import random

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help="File") 
parser.add_argument('-n', '--nevents', type=int, required=False, help="Number of events to process") 
parser.add_argument('-o', '--output', type=str, required=True, help="Output file")
parser.add_argument('-c', '--correlation', required=False, help="Plotting correlationmatrix", action = "store_true")
parser.add_argument('-l', '--listtype', type = str, required=True, help="List type")
parser.add_argument('-b', '--balanced', required=False, help="for a dataset with equal number of 1 and 0",  action = "store_true")

args = parser.parse_args()

f = r.TFile(args.file, "OPEN")
   
cuts = [
    ("pt_min_muon", 20),
    ("eta_max_muon", 2.1),
    ("pt_min_jets", 30),
    ("eta_max_jets", 4.7),
    ("eq_flag", 0),
    ("min_njets",4),
    ("atleastone_mjj_M", 250),
    ]

if args.nevents != None: 
    cuts.append(("n_events", args.nevents))

"""
l = []

ievent = 0
ngood = 0 
nbad = 0


with tqdm(total=f.Get("tree").GetEntries()) as pbar:
    for evento in EventIterator(f,criteria = cuts, pairing = True):
        pts = [j.Pt() for j in evento.jets]
        etas = [abs(j.Eta()) for j in evento.jets]
        ms = [j.M() for j in evento.jets]
        maxpt = max(pts)
        minpt = min(pts)
        maxeta = max(etas)
        maxm = max(ms)
        minm = min(ms)
        for j in evento.jets:
            if evento.paired_parton(j) == None:
                l.append([j.Px(),j.Py(),j.Pz(),j.Pt(),j.E(), 
                                maxpt, minpt, maxeta, maxm, minm, 0])
                nbad += 1
            else:
                l.append([j.Px(),j.Py(),j.Pz(),j.Pt(),j.E(), 
                                maxpt, minpt, maxeta, maxm, minm, 1] )
                ngood += 1
        # Update progress bar
        pbar.update(evento.evId - ievent)
        ievent = evento.evId

a = np.array(l)
np.save(args.output, a)
print("N. good: {} | N. bad: {}".format(ngood, nbad))

f.Close()

"""

def momcons(evento, j):
    l = []
    mom = (evento.muon + evento.neutrino).Pt()
    for i in (range(len(evento.jets))):
        if evento.jets[i] != j:
            l.append(abs(mom-(evento.jets[i]+j).Pt()))
    l = sorted(l)
    return l[0]
    
    
    
def inbetweenD(getti, j):
    inb = 0
    for i in (range(len(getti))):
        if getti[i] != j:
            if math.sqrt((getti[i].X()-j.X())**2 + (getti[i].Y()-j.Y())**2 + (getti[i].Z()-j.Z())**2) < 100:
                inb += 1
    
    return inb


def inbetweenR(getti, j):
    inb = 0
    for i in (range(len(getti))):
        if getti[i] != j:
            if j.DeltaR(getti[i]) < 2:
                inb += 1
    
    return inb


def nearW(getti, j):
    l = []
    for i in (range(len(getti))):
        if getti[i] != j:
            l.append(abs(80-(getti[i]+j).M()))
    
    l = sorted(l)
    return l[0]

def MaxM(getti, j):
    l = []
    for i in (range(len(getti))):
        if getti[i] != j:
            l.append((getti[i]+j).M())
    l = sorted(l, reverse = True)
    return l[0]
    
def Maxdeltaeta(getti, j):
    l = []
    for i in (range(len(getti))):
        if getti[i] != j:
            l.append(abs(getti[i].Eta()-j.Eta()))
    l = sorted(l, reverse = True)
    return l[0]

def Mindeltaeta(getti, j):
    l = []
    for i in (range(len(getti))):
        if getti[i] != j:
            l.append(abs(getti[i].Eta()-j.Eta()))
    l = sorted(l)
    return l[0]

def MinR(getti, j):
    l = []
    for i in (range(len(getti))):
        if getti[i] != j:
            l.append(j.DeltaR(getti[i]))
    l = sorted(l)
    return l[0]

def MaxR(getti, j):
    l = []
    for i in (range(len(getti))):
        if getti[i] != j:
            l.append(j.DeltaR(getti[i]))
    l = sorted(l, reverse = True)
    return l[0]

def size(getti, j):
    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            distance.append([j.DeltaR(getti[i]) , math.sqrt((getti[i].X()-j.X())**2 + (getti[i].Y()-j.Y())**2 + (getti[i].Z()-j.Z())**2) ])
    distance = sorted(distance, key = itemgetter(1))
    
    return distance[0][0]
    
def distance_nearest(getti, j):
    
    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            #distance.append(math.sqrt((getti[i].X()-j.X())**2 + (getti[i].Y()-j.Y())**2 + (getti[i].Z()-j.Z())**2) )
            distance.append(j.DeltaR(getti[i]))
    distance = sorted(distance)
    
    return distance[0]

def nearest_Pt(getti, j):
    
    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            distance.append([ i, j.DeltaR(getti[i])])
            
    distance = sorted(distance, key = itemgetter(1))
    
    return getti[distance[0][0]].Pt()

def min_deltaeta(getti, j):
    
    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            distance.append([ i, j.DeltaR(getti[i]) ])
    distance = sorted(distance, key = itemgetter(1))
    
    return getti[distance[0][0]].Eta()

def min_deltaphi(getti, j):

    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            distance.append([ i, j.DeltaR(getti[i])])
    distance = sorted(distance, key = itemgetter(1))
    
    return getti[distance[0][0]].Phi()

def variable_list1(evento,j):
    pts = [j.Pt() for j in evento.jets]
    etas = [abs(j.Eta()) for j in evento.jets]
    ms = [j.M() for j in evento.jets]
    maxpt = max(pts)
    minpt = min(pts)
    maxeta = max(etas)
    maxm = max(ms)
    minm = min(ms)
    if evento.paired_parton(j) == None:
        return [j.Px(),j.Py(),j.Pz(),j.Pt(),j.E(), maxpt, minpt, maxeta, maxm, minm, 0]
    else:
        return [j.Px(),j.Py(),j.Pz(),j.Pt(),j.E(), maxpt, minpt, maxeta, maxm, minm, 1]
        
def variable_list2(evento, j):
    totaljet = evento.njets
    pts = [j.Pt() for j in evento.jets]
    etas = [abs(j.Eta()) for j in evento.jets]
    ms = [j.M() for j in evento.jets]
    maxpt = max(pts)
    minpt = min(pts)
    maxeta = max(etas)
    mineta = min(etas)
    maxm = max(ms)
    minm = min(ms)
    near_mineta = min_deltaeta(evento.jets, j)
    near_minphi = min_deltaphi(evento.jets, j)
    near_Pt = nearest_Pt(evento.jets, j)
    distance = distance_nearest(evento.jets, j)
    near_size = size(evento.jets,j)
    if evento.paired_parton(j) == None:
        return [totaljet, near_mineta, near_minphi, near_Pt,  maxm, maxpt, minpt , j.Pt(), j.E(), distance, 0]
    else:
        return [totaljet, near_mineta, near_minphi, near_Pt, maxm, maxpt, minpt , j.Pt(), j.E(), distance, 1]
        
# CANDIDATE--------------------------
def variable_list3(evento, j):
    totaljet = evento.njets
    pts = [j.Pt() for j in evento.jets]
    etas = [abs(j.Eta()) for j in evento.jets]
    ms = [j.M() for j in evento.jets]
    maxpt = max(pts)
    minpt = min(pts)
    maxm = max(ms)
    minm = min(ms)
    W = nearW(evento.jets, j)
    Maxjj = MaxM(evento.jets,j)
    distance = distance_nearest(evento.jets, j)
    Mindelta= Maxdeltaeta(evento.jets, j)
    Maxdelta=Mindeltaeta(evento.jets, j)
    inb = inbetweenR(evento.jets, j)
    inbd = inbetweenD(evento.jets,j)
    near_Pt = nearest_Pt(evento.jets, j)
    MinDR = MinR(evento.jets, j)
    MaxDR= MaxR(evento.jets, j)
    if evento.paired_parton(j) == None:
        return [ totaljet, MinDR, Maxdelta,  W, Maxjj, near_Pt, maxm, minm, maxpt, minpt , j.Pt(), j.E(), distance, inb, inbd, 0]
    else:
        return [ totaljet, MinDR, Maxdelta,  W, Maxjj, near_Pt, maxm, minm, maxpt, minpt , j.Pt(), j.E(), distance, inb, inbd, 1]
    
    
#nevent ha correlazione bassissima non gli altri
    
def variable_list4(evento, j, nevent):
    totaljet = evento.njets
    distance = distance_nearest(evento.jets, j)
    inb = inbetweenR(evento.jets, j)
    inbd = inbetweenD(evento.jets,j)
    pts = [j.Pt() for j in evento.jets]
    near_size = size(evento.jets,j)
    near_Pt = nearest_Pt(evento.jets, j)
    minpt = min(pts)
    maxpt = max(pts)
    if evento.paired_parton(j) == None:
        return [ totaljet, near_Pt, minpt, maxpt,  j.Pt(), j.E(), distance, near_size, inb, inbd, 0]
    else:
        return [ totaljet, near_Pt, minpt, maxpt,  j.Pt(), j.E(), distance, near_size, inb, inbd, 1]
    
def variable_list5(evento, j):
    totaljet = evento.njets
    distance = distance_nearest(evento.jets, j)
    inb = inbetweenR(evento.jets, j)
    inbd = inbetweenD(evento.jets,j)
    pts = [j.Pt() for j in evento.jets]
    etas = [abs(j.Eta()) for j in evento.jets]
    ms = [j.M() for j in evento.jets]
    maxpt = max(pts)
    minpt = min(pts)
    maxm = max(ms)
    #near_size = size(evento.jets,j)
    near_Pt = nearest_Pt(evento.jets, j)
    if evento.paired_parton(j) == None:
        return [ totaljet ,j.Px(), j.Py(), j.Pz(), j.Eta(), j.Phi(), maxm, maxpt, minpt, near_Pt, j.Pt(), j.E(), distance, inb, inbd, 0]
    else:
        return [ totaljet ,j.Px(), j.Py(), j.Pz(), j.Eta(), j.Phi(), maxm, maxpt, minpt, near_Pt, j.Pt(), j.E(), distance, inb, inbd, 1]
        
def variable_list6(evento):
    l = []
    combo_jets = list(itertools.permutations(evento.jets, 4))
    for j in combo_jets:
        count = 0
        for i in j:
            if evento.paired_parton(i) != None:
                count += 1
        if count == 4:
            l.append([j[0].E(),j[1].E(), j[2].E(), j[3].E(), j[0].Pt(),j[1].Pt(), j[2].Pt(), j[3].Pt(), 1])
        else:
             l.append([j[0].E(),j[1].E(), j[2].E(), j[3].E(), j[0].Pt(),j[1].Pt(), j[2].Pt(), j[3].Pt(), 0])
      
    return l 
                
        
l = []

ievent = 0
ngood = 0 
nbad = 0
nevent = 0 

if not args.balanced:
    with tqdm(total=f.Get("tree").GetEntries()) as pbar:
        for evento in EventIterator(f,criteria = cuts, pairing = True):
            if args.listtype == "variable_list6":
                variab = eval(args.listtype + '(evento)')
                #print(variab)
                l.append(variab)
                nevent += 1
else:
    for j in evento.jets:
        variables = []
        if args.listtype == "variable_list4":
            variab = eval(args.listtype + '(evento,j, nevent)')
        else:
            variab = eval(args.listtype + '(evento,j)')
        l.append(variab)
        
        if evento.paired_parton(j) == None:
            nbad += 1
        else:
            ngood += 1
        nevent += 1
                
    else:
        print('sono qui')
        count_zero = 0
        count_zero_print = 0
        count_uno = 0
        count_uno_print = 0
            
        for evento in EventIterator(f,criteria = cuts, pairing = True):
            nevent += 1
            for j in evento.jets:
                variables = []
                if evento.paired_parton(j) == None:
                    count_zero += 1
                    if count_zero < 855314:
                    #if count_zero < 50001:
                        count_zero_print += 1
                        if args.listtype == "variable_list4":
                            variab = eval(args.listtype + '(evento,j, nevent)')
                        else:
                            variab = eval(args.listtype + '(evento,j)')
                        l.append(variab)
                        #print(variab)
                else:
                    count_uno += 1
                    if count_uno < 855314:
                        count_uno_print += 1
                        if args.listtype == "variable_list4":
                            variab = eval(args.listtype + '(evento,j, nevent)')
                        else:
                            variab = eval(args.listtype + '(evento,j)')
                        l.append(variab)
                if evento.paired_parton(j) == None:
                    nbad += 1
                else:
                    ngood += 1
                
        # Update progress bar
        pbar.update(evento.evId - ievent)
        ievent = evento.evId
       
if args.balanced:
    print('>>>number of 0 jets: {}'.format(count_zero_print))
    print('>>>number of 1 jets: {}'.format(count_uno_print))

#da qui è un test per variable_list5 solo per vedere come agisce random.shuffle.
#se non si usa variable_list5 bisogna inserire nel ciclo for il numero della colonna del target
#(a.shape[1]-1)    
a = np.array(l)
if args.listtype == "variable_list6":
    a = np.concatenate(a)
np.random.shuffle(a)
print(">>>Array shape: {}".format(a.shape))
"""
count_zero = 0
count_uno = 0
for i in a[:,16]:
    if i == 0.:
        count_zero += 1 
    else:
        count_uno += 1
print('>>>number of 0 jets after shuffle: {}'.format(count_zero))
print('>>>number of 1 jets after shuffle: {}'.format(count_uno))


np.save(args.output, a)
print("N. good: {} | N. bad: {}".format(ngood, nbad))

f.Close()
"""

if args.correlation:
    if args.listtype == "variable_list1":
        axisname = ["Px","Py","Pz","Pt","E", "maxpt", "minpt", "maxeta", "maxm", "minm", "match"]
    elif args.listtype == "variable_list2":
        axisname = ["totaljet", "near_mineta", "near_minphi", "near_Pt", "maxm", "maxPt", "minPt" ,"Pt","E", "distance", "match"]
    elif args.listtype == "variable_list3":
        axisname = ["totaljet", "MinR", "Maxdeltaeta","NearW", "MaxMjj", "nearest_Pt", "maxm", "minm", "maxPt", "minPt" ,"Pt","E", "distance", "inside DeltaR", "inside distance", "match"]
    #>>>>>>>>>>>>>>>BEST CANDIDATE
    elif args.listtype == "variable_list4":
        axisname = ["totaljet", "Near_Pt","minpt", "maxpt","nevent" ,"Pt","E", "distance", "DeltaR", "inside DeltaR1", "bet. distance", "match"]
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    elif args.listtype == "variable_list5":
        axisname = ["totaljet", "Px", "Py", "Pz", "Eta", "Phi", "maxm", "minm", "maxPt", "minPt" "near_Pt" ,"Pt","E", "distance", "inside DeltaR1", "bet. distance", "match"]
    elif args.listtype == "variable_list6":
        axisname = ["E1", "E2", "E3", "E4", "Pt1", "Pt2", "Pt3", "Pt4", "match"]
        
        
    print(">>>Plotting correlation matrix")
    corr = np.corrcoef(a, rowvar = False)
    #mov_data = ["Px", "Py", "Pz", "Pt", "E", "MaxPt", "MinPt", "MaxEta", "MaxM", "MinM", "Match"]
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(axisname)), axisname, fontsize=12, rotation = 90)
    plt.yticks(range(len(axisname)), axisname, fontsize=12)
    plt.title("Correlation matrix jets")
    for i in range(len(axisname)):
        for j in range(len(axisname)):
            text = plt.text(j, i, "%.2f" % corr[i, j], ha="center", va="center", color="w")
    plt.show()

    
