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

args = parser.parse_args()

#>>>>>File<<<<<<<<<<<<<
f = r.TFile(args.file, "OPEN")

#>>>>>>>>>Cuts<<<<<<<<<<
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

#>>>>>>>>Dataset Variables<<<<<<<<<<

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

def max_deltaphi(getti, j):

    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            distance.append([ i, j.DeltaR(getti[i])])
    distance = sorted(distance, key = itemgetter(1), reverse=True)
    
    return getti[distance[0][0]].Phi()


#>>>>Normalization of dataset<<<<<<
def normalizer(data):
    media = np.zeros(25)
    for row in data:
        i = 0
        while i < data.shape[1]-1:
            media[i] += row[i]
            i += 1
    i = 0 
    dev = np.zeros(25)
    print(dev)
    i = 0
    while i < data.shape[1]-1:
        for e in data[:,i]:
            dev[i] += (e-media[i])**2
        i += 1
    print(dev)
    i = 0
    while i<len(dev):
        dev[i] = np.sqrt(dev[i]/(data.shape[0]-1))
        i += 1
    print(dev)
    i = 0
    while i < data.shape[1]-1:
        z = 0
        while  z < data[:,i].shape[0]:
            data[z,i] = (data[z,i]-media[i])/dev[i]
            z+=1
        i+=1
    
    return data


#>>>>>>>>>>>List creators<<<<<<<
def variable_list_creator(evento, j):
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
    near_Pt = nearest_Pt(evento.jets, j)
    min_phi=min_deltaphi(evento.jets, j)
    max_phi = max_deltaphi(evento.jets,j)
    if evento.paired_parton(j) == None:
        return [ totaljet ,j.Px(), j.Py(), j.Pz(), maxm, near_Pt, j.Pt(), j.E(), distance, 0]
    else:
        return [ totaljet ,j.Px(), j.Py(), j.Pz(), maxm, near_Pt, j.Pt(), j.E(), distance, 1]
    
#>>>>>>>>>>>MAIN<<<<<<<<<

l = []

ievent = 0
ngood = 0 
nbad = 0
nevent = 0 

print(">>>Building array...")
with tqdm(total=f.Get("tree").GetEntries()) as pbar:
    for evento in EventIterator(f,criteria = cuts, pairing = True):
        nevent += 1
        for j in evento.jets:
            variab = variable_list_creator(evento, j)
            l.append(variab)
            if evento.paired_parton(j) == None:
                nbad += 1
            else:
                ngood += 1
    #Update progress bar
    pbar.update(evento.evId - ievent)
    ievent = evento.evId
        
a = np.array(l)
print(a.shape)
np.random.shuffle(a)
print(a[:][:2])
print(">>>Normalizing Dataset...")
a = normalizer(a)
print(a[:][:5])
print(">>>Array shape: {}".format(len(a)))
np.save(args.output, a)

#>>>>>>>>>Plotting correlation matrix<<<<<<<
axisname = ["totaljet", "Px", "Py", "Pz", "maxm", "near_Pt" ,"Pt","E", "distance", "match"]
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
