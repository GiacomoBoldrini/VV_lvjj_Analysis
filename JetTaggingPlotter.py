import ROOT as r
import sys
from operator import attrgetter, itemgetter
from array import array
from VBSAnalysis.EventIterator import EventIterator
from VBSAnalysis.Utils import JetSelectors as jsel
from VBSAnalysis import JetTagging 
import argparse
import myplotter as plotter
from collections import namedtuple
from ROOT import gSystem
from ROOT import gStyle
plotter.setStyle()
tdrStyle =  r.TStyle("tdrStyle","Style for P-TDR")
import argparse
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import math

def to_xy(df, target):
    y = df[:,target]
    x = np.delete(df, target, 1)
    return x,y

def mkdir_p(mypath):
    #crea una directory

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

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
            distance.append(math.sqrt((getti[i].X()-j.X())**2 + (getti[i].Y()-j.Y())**2 + (getti[i].Z()-j.Z())**2) )
    distance = sorted(distance)
    
    return distance[0]

def nearest_Pt(getti, j):
    
    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            distance.append([ i, math.sqrt((getti[i].X()-j.X())**2 + (getti[i].Y()-j.Y())**2 + (getti[i].Z()-j.Z())**2) ])
    distance = sorted(distance, key = itemgetter(1))
    
    return getti[distance[0][0]].Pt()

def min_deltaeta(getti, j):
    
    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            distance.append([ i, math.sqrt((getti[i].X()-j.X())**2 + (getti[i].Y()-j.Y())**2 + (getti[i].Z()-j.Z())**2) ])
    distance = sorted(distance, key = itemgetter(1))
    
    return abs(getti[distance[0][0]].Eta()-j.Eta())

def min_deltaphi(getti, j):

    distance = []
    for i in (range(len(getti))):
        if getti[i] != j:
            distance.append([ i, math.sqrt((getti[i].X()-j.X())**2 + (getti[i].Y()-j.Y())**2 + (getti[i].Z()-j.Z())**2) ])
    distance = sorted(distance, key = itemgetter(1))
    
def variable_list1(evento,j):
    pts = [j.Pt() for j in evento.jets]
    etas = [abs(j.Eta()) for j in evento.jets]
    ms = [j.M() for j in evento.jets]
    maxpt = max(pts)
    minpt = min(pts)
    maxeta = max(etas)
    maxm = max(ms)
    minm = min(ms)
    
    return np.array([[j.Px(),j.Py(),j.Pz(),j.Pt(),j.E(), maxpt, minpt, maxeta, maxm, minm]])

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

    return np.array([[totaljet, maxm, maxpt, minpt , j.Pt(), j.E(), distance]])
"""
def variable_list3(evento, j):
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
    inb = inbetweenR(evento.jets, j)
    inbd = inbetweenD(evento.jets,j)
    near_size = size(evento.jets,j)
    if evento.paired_parton(j) == None:
        return [j.Pz(), totaljet, maxm, maxpt, minpt , j.Pt(), j.E(), distance, near_size, inb, inbd, 0]
    else:
        return [j.Pz(), totaljet, maxm, maxpt, minpt , j.Pt(), j.E(), distance, near_size, inb, inbd, 1]
"""
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
    return np.array([[ totaljet, MinDR, MaxDR, Maxdelta, Mindelta,  W, Maxjj, near_Pt, maxm, minm, maxpt, minpt , j.Pt(), j.E(), distance, inb, inbd]])
    
def variable_list4(evento, j):
    totaljet = evento.njets
    pts = [j.Pt() for j in evento.jets]
    etas = [abs(j.Eta()) for j in evento.jets]
    ms = [j.M() for j in evento.jets]
    maxpt = max(pts)
    minpt = min(pts)
    maxm = max(ms)
    distance = distance_nearest(evento.jets, j)
    inb = inbetweenR(evento.jets, j)
    inbd = inbetweenD(evento.jets,j)
    return np.array([[ totaljet, maxm, maxpt, minpt , j.Pt(), j.E(), distance, inb, inbd]])

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
    MinDR = MinR(evento.jets, j)
    MaxDR= MaxR(evento.jets, j)
    Mindelta= Maxdeltaeta(evento.jets, j)
    Maxdelta=Mindeltaeta(evento.jets, j) 
    #near_size = size(evento.jets,j)
    near_Pt = nearest_Pt(evento.jets, j)
    W = nearW(evento.jets, j)
    return np.array([[ totaljet ,MinDR, MaxDR, Mindelta, Maxdelta, W, maxm, maxpt, minpt, near_Pt, j.Pt(), j.E(), distance, inb, inbd]])
    
    
    

f = r.TFile("data/ewk_giacomo.root")

cuts = [
    ("pt_min_jets",30),
    ("min_njets",4),
    #("eq_njets",4),
    #("eta_max_jets", 2),
    ("eq_flag", 0),
    ("n_events", 1000),
    ("atleastone_mjj_M", 250)
]

parser = argparse.ArgumentParser()

parser.add_argument('-dnn', '--neuralnetwork',  required=False, help = "check if we want to use neural network model", action = "store_true")
parser.add_argument('-in', '--inpt', nargs='+', required = False, help = "Input  network models")
#parser.add_argument('-o', '--output', type = str, required = False, help = "output path for neural network prediction")
parser.add_argument('-d', '--data', type = str, required = False, help = "Dataset for neural network computation")
parser.add_argument('-v', '--val', required = False, help = "Check if we want njets or th", action = "store_true")
parser.add_argument('-l', '--list', type = str, required = False, help = 'Check on dataset variables')

args = parser.parse_args()

StratList = []
"""
StratList.append(["strategy_wz_mjj", "Nearest W/Z jets", "Nearest W/Z good", "Nearest W/Z bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj W/Z jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mjj_wz", "Nearest W/Z jets", "Nearest W/Z good", "Nearest W/Z bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj W/Z jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mjj_mz", "Nearest Z jets", "Nearest Z good", "Nearest Z bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Z jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mz_mjj", "Nearest Z jets", "Nearest Z good", "Nearest Z bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Z jets (GeV)", "Max Mjj jets (GeV)"])
"""
StratList.append(["strategy_mjj_mw", "Nearest W jets", "Nearest W good", "Nearest W bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj W jets (GeV)", "Max Mjj jets (GeV)"])
"""
StratList.append(["strategy_mw_mjj", "Nearest W jets", "Nearest W good", "Nearest W bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj W jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mw_deltaeta", "Nearest W jets", "Nearest W good", "Nearest W bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj W jets (GeV)", "Max #Delta#eta Mjj jets (GeV)"])

StratList.append(["strategy_deltaeta_mw", "Nearest W jets", "Nearest W good", "Nearest W bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj W jets (GeV)", "Max #Delta#eta Mjj jets (GeV)"])

StratList.append(["strategy_maxpt_mjj", "Max P_{t} jets", "Max P_{t} good", "Max P_{t} bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Max P_{t} jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mjj_maxpt", "Max P_{t} jets", "Max P_{t} good", "Max P_{t} bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Max P_{t} jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_maxpt_deltaeta", "Max P_{t} jets", "Max P_{t} good", "Max P_{t} bad", "Max Mjj jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Max P_{t} jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_deltaeta_maxpt", "Max P_{t} jets", "Max P_{t} good", "Max P_{t} bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Max P_{t} jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_mindeltaeta_mjj", "Min #Delta#eta jets", "Min #Delta#eta good", "Min #Delta#eta bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Min #Delta#eta jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mjj_mindeltaeta", "Min #Delta#eta jets", "Min #Delta#eta good", "Min #Delta#eta bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Min #Delta#eta jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_maxdeltaeta_mindeltaeta", "Min #Delta#eta jets", "Min #Delta#eta good", "Min #Delta#eta bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Min #Delta#eta jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_mindeltaeta_maxdeltaeta", "Min #Delta#eta jets", "Min #Delta#eta good", "Min #Delta#eta bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Min #Delta#eta jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_Weta_maxeta", "Min #Delta#eta/ Nearest W / Max P_{t} jets", "Min #Delta#eta/ Nearest W / Max P_{t} good", "Min #Delta#eta/ Nearest W / Max P_{t} bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Min #Delta#eta/ Nearest W / Max P_{t} jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_maxeta_Weta", "Min #Delta#eta/ Nearest W jets", "Min W#Delta#eta good", "Min W#Delta#eta bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Min W#Delta#eta jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_Weta_mjj", "Min #Delta#eta/ Nearest W jets", "Min W#Delta#eta good", "Min W#Delta#eta bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Min W#Delta#eta jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mjj_Weta", "Min #Delta#eta/ Nearest W jets", "Min W#Delta#eta good", "Min W#Delta#eta bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Min W#Delta#eta jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_proviamo_mjj", "Min #Delta#eta/ Nearest W / jets", "Min W#Delta#eta good", "Min W#Delta#eta bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Min W#Delta#eta jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mjj_proviamo", "Min #Delta#eta/ Nearest W / jets", "Min W#Delta#eta good", "Min W#Delta#eta bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Min W#Delta#eta jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_deltaeta_proviamo", "Min #Delta#eta/ Nearest W / jets", "Min W#Delta#eta good", "Min W#Delta#eta bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Min W#Delta#eta jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_proviamo_deltaeta", "Min #Delta#eta/ Nearest W / jets", "Min W#Delta#eta good", "Min W#Delta#eta bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Min W#Delta#eta jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_minsize_maxsize", "Min #DeltaR jets", "Min #DeltaR good", "Min #DeltaR bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Min #DeltaR jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxsize_minsize", "Min #DeltaR jets", "Min #DeltaR good", "Min #DeltaR bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Min #DeltaR jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_minsize_mjj", "Min #DeltaR jets", "Min #DeltaR good", "Min #DeltaR bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Min #DeltaR jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mjj_minsize", "Min #DeltaR jets", "Min #DeltaR good", "Min #DeltaR bad", "Max Mjj jets", "Max Mjj jets good", "Max Mjj jets bad", "Mjj Min #DeltaR jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_minsize_maxdeltaeta", "Min #DeltaR jets", "Min #DeltaR good", "Min #DeltaR bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Min #DeltaR jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_maxdeltaeta_minsize", "Min #DeltaR jets", "Min #DeltaR good", "Min #DeltaR bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Min #DeltaR jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_maxsize_W", "Nearest W jets", "Nearest W good", "Nearest W bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Nearest W jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_W_maxsize", "Nearest W jets", "Nearest W good", "Nearest W bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Nearest W jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxsize_Z", "Nearest Z jets", "Nearest Z good", "Nearest Z bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Nearest Z jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_Z_maxsize", "Nearest Z jets", "Nearest Z good", "Nearest Z bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Nearest Z jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxsize_maxpt", "Max P_{t} jets", "Max P_{t} good", "Max P_{t} bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Max P_{t} jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxpt_maxsize", "Max P_{t} jets", "Max P_{t} good", "Max P_{t} bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Max P_{t} jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxsize_WZ", "Nearest W or Z jets", "Nearest W or Z  good", "Nearest W or Z  bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Nearest W or Z  jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_WZ_maxsize", "Nearest W or Z jets", "Nearest W or Z  good", "Nearest W or Z  bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Nearest W or Z  jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_mindeltaeta_maxsize", "Min #Delta#eta jets", "Min #Delta#eta good", "Min #Delta#eta bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Min #Delta#eta  jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxsize_mindeltaeta", "Min #Delta#eta jets", "Min #Delta#eta good", "Min #Delta#eta bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj Min #Delta#eta  jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_proviamo_maxsize", "Min #Delta#eta/ Nearest W/ Max P_{t} jets", "Min #Delta#eta/ Nearest W/ Max P_{t} good", "Min #Delta#eta/ Nearest W/ Max P_{t} bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj #Delta#eta/W/Max P_{t}  jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxsize_proviamo", "Min #Delta#eta/ Nearest W/ Max P_{t} jets", "Min #Delta#eta/ Nearest W/ Max P_{t} good", "Min #Delta#eta/ Nearest W/ Max P_{t} bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj #Delta#eta/W/Max P_{t}  jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxsize_Weta", "Min #Delta#eta/ Nearest W jets", "Min #Delta#eta/ Nearest W good", "Min #Delta#eta/ Nearest W bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj #Delta#eta/W  jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_Weta_maxsize", "Min #Delta#eta/ Nearest W jets", "Min #Delta#eta/ Nearest W good", "Min #Delta#eta/ Nearest W bad", "Max #DeltaR jets", "Max #DeltaR jets good", "Max #DeltaR jets bad", "Mjj #Delta#eta/W  jets (GeV)", "Mjj Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxdeltaeta_Z", "Nearest Z jets", "Nearest Z good", "Nearest Z bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Nearest Z jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_Z_maxdeltaeta", "Nearest Z jets", "Nearest Z good", "Nearest Z bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Nearest Z jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_WZ_maxdeltaeta", "Nearest W/Z jets", "Nearest W/Z good", "Nearest W/Z bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Nearest W/Z jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_maxdeltaeta_WZ", "Nearest W/Z jets", "Nearest W/Z good", "Nearest W/Z bad", "Max #Delta#eta jets", "Max #Delta#eta jets good", "Max #Delta#eta jets bad", "Mjj Nearest W/Z jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_minetaWZ_mjj", "Nearest W/Z/min #Delta#eta jets", "Nearest W/Z/min #Delta#eta  good", "Nearest W/Z/min #Delta#eta  bad", "Max Mjj jets", "Max Mjj good", "Max Mjj bad", "Mjj Nearest W/Z/min #Delta#eta  jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_mjj_minetaWZ", "Nearest W/Z/min #Delta#eta jets", "Nearest W/Z/min #Delta#eta  good", "Nearest W/Z/min #Delta#eta  bad", "Max Mjj jets", "Max Mjj good", "Max Mjj bad", "Mjj Nearest W/Z/min #Delta#eta  jets (GeV)", "Max Mjj jets (GeV)"])

StratList.append(["strategy_maxdeltaeta_minetaWZ", "Nearest W/Z/min #Delta#eta jets", "Nearest W/Z/min #Delta#eta  good", "Nearest W/Z/min #Delta#eta  bad", "Max #Delta#eta jets", "Max #Delta#eta good", "Max #Delta#eta bad", "Mjj Nearest W/Z/min #Delta#eta  jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_minetaWZ_maxdeltaeta", "Nearest W/Z/min #Delta#eta jets", "Nearest W/Z/min #Delta#eta  good", "Nearest W/Z/min #Delta#eta  bad", "Max #Delta#eta jets", "Max #Delta#eta good", "Max #Delta#eta bad", "Mjj Nearest W/Z/min #Delta#eta  jets (GeV)", "Max #Delta#eta jets (GeV)"])

StratList.append(["strategy_minetaWZ_maxsize", "Nearest W/Z/min #Delta#eta jets", "Nearest W/Z/min #Delta#eta  good", "Nearest W/Z/min #Delta#eta  bad", "Max #DeltaR jets", "Max #DeltaR good", "Max #DeltaR bad", "Mjj Nearest W/Z/min #Delta#eta  jets (GeV)", "Max #DeltaR jets (GeV)"])

StratList.append(["strategy_maxsize_minetaWZ", "Nearest W/Z/min #Delta#eta jets", "Nearest W/Z/min #Delta#eta  good", "Nearest W/Z/min #Delta#eta  bad", "Max #DeltaR jets", "Max #DeltaR good", "Max #DeltaR bad", "Mjj Nearest W/Z/min #Delta#eta  jets (GeV)", "Max #DeltaR jets (GeV)"])
"""

if not args.neuralnetwork:
    l = []
    for strat in StratList:
        print(">>>>computing for {0}".format(strat[0]))
        cwrt = 0
        cwrg = 0
        #c1, pad1, pad2 = plotter.createCanvasPads()
        #c2, pad1, pad2 = plotter.createCanvasPads()
        strategy = getattr(JetTagging, strat[0])
        output_dir = "/home/giacomo/tesi1/VBSAnalysis/images/prova/{}".format(strat[0])
        mkdir_p(output_dir)
        #hs_v, h_v, h_v_bad = plotter.StackCreator1(strat[1], strat[2], strat[3])
        hs_v, h_v, h_v_bad = plotter.StackCreator2(strat[1], strat[2], strat[3])
        hs_vbs, h_vbs, h_vbs_bad = plotter.StackCreator2(strat[4],  strat[5], strat[6])
        print(">>>Checking on events")
        for evento in EventIterator(f,criteria = cuts, pairing = True) :
            #strategy partons2 restituisce la coppia di partoni più vicina a Z o W
            partons_pair1 = JetTagging.strategy_partons2(evento.partons)
            jets_pair1 = strategy(evento.jets)
            #associazione vbs
            associazione = JetTagging.check_association(evento, partons_pair1.vbs_pair, jets_pair1.vbsjets)
            #associazione vector boson
            associazione2 = JetTagging.check_association(evento, partons_pair1.w_pair,  jets_pair1.vjets)
            if associazione:
                #due getti vbs indici corrispondono a partoni
                h_vbs.Fill((jets_pair1.vbsjets[0]+jets_pair1.vbsjets[1]).M())
            else:
                #due getti vbs indici NON corrispondono a partoni
                h_vbs_bad.Fill((jets_pair1.vbsjets[0]+jets_pair1.vbsjets[1]).M())
            if associazione2:
                #due getti V indici corrispondono a partonieff
                h_v.Fill((jets_pair1.vjets[0]+jets_pair1.vjets[1]).M())
            else:
                #due getti V indici NON corrispondono a partoni
                h_v_bad.Fill((jets_pair1.vjets[0]+jets_pair1.vjets[1]).M())
            if associazione and associazione2:
                cwrt = cwrt +1
            else:
                cwrg = cwrg +1
        print(">>>Saving efficiencies")
        eff = cwrt/(cwrt+cwrg)
        
        c1, pad1, pad2 = plotter.createCanvasPads()

        pad1.cd()
        hs_v.Draw("nostack hist")
        legend = plotter.createLegend(h_v, h_v_bad,1)
        legend.Draw("same")
        pad2.cd()
        hratio = plotter.createRatio(h_v, h_v +h_v_bad, "eff.")
        hratio.Draw("hist")
        hratio.SetFillColor(0)
        hratioerror = hratio.DrawCopy("E2 same")
        hratioerror.SetFillStyle(3013)
        hratioerror.SetFillColor(13)
        hratioerror.SetMarkerStyle(1)
        hratio = plotter.RatioDrawOptions(hratio, strat[7])
        hratio.GetXaxis().SetLabelSize(0.07)
        c1.Draw()
        c1.SaveAs(output_dir + "/Vjets.png")
        
        
        c2, pad3, pad4 = plotter.createCanvasPads()
        
        pad3.cd()
        hs_vbs.Draw("nostack hist")
        legend = plotter.createLegend(h_vbs, h_vbs_bad,1)
        legend.Draw("same")
        pad4.cd()
        hratio = plotter.createRatio(h_vbs, h_vbs +h_vbs_bad, "eff.")
        hratio.Draw("hist")
        hratio.SetFillColor(0)
        hratioerror = hratio.DrawCopy("E2 same")
        hratioerror.SetFillStyle(3013)
        hratioerror.SetFillColor(13)
        hratioerror.SetMarkerStyle(1)
        hratio = plotter.RatioDrawOptions(hratio, strat[8])
        hratio.GetXaxis().SetLabelSize(0.07)
        c2.Draw()
        c2.SaveAs(output_dir + "/VBSjets.png")
        
        l.append([eff, strat[0] ])


    l = sorted(l, key=itemgetter(0), reverse = True)
    print(">>>Saving parameters")
    f = open("/home/giacomo/tesi1/VBSAnalysis/images/TaggingEfficiencies.txt", "w")
    i = 0
    while i < len(l):
        f.write("{}\n".format(l[i]))
        i = i+1
    f.close()
    
else:
    for models in args.inpt:
        print(">>>DNN application testing model {0}".format(models))
        input_dir = "/home/giacomo/tesi1/DNN_test/third dataset/DNNoptimizer/megadataset/1010/modello_prova_10_10"
        model = load_model(input_dir)
        #data = np.load('Dataset1.npy')
        #print(data[:10][:])
        #x,y = to_xy(data, 7)
        #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
        #data.shape
        l = []
        l1 = []
        dir_output = "/home/giacomo/tesi1/DNN_test/Convolutional"
        #mkdir_p(dir_output)
        for strat in StratList:
            print(">>>>Computing strategy {}".format(strat[0]))
            if not args.val:
                cwrt = 0
                cwrg = 0
                g = 4
                strategy = getattr(JetTagging, strat[0])
                output_dir = dir_output + "{}".format(strat[0])
                #hs_v, h_v, h_v_bad = plotter.StackCreator1(strat[1], strat[2], strat[3])
                #hs_vbs, h_vbs, h_vbs_bad = plotter.StackCreator2(strat[4],  strat[5], strat[6])
                while g < 8:
                    print("g {}".format(g))
                    nevento = 0
                    print(">>>>Checking on jets for event")
                    for evento in EventIterator(f,criteria = cuts, pairing = True) :
                        nevento += 1
                        pts = [j.Pt() for j in evento.jets]
                        etas = [abs(j.Eta()) for j in evento.jets]
                        ms = [j.M() for j in evento.jets]
                        maxpt = max(pts)
                        minpt = min(pts)
                        maxeta = max(etas)
                        maxm = max(ms)
                        minm = min(ms)
                        evento_jets_score = []
                        for j in evento.jets:
                            if evento.paired_parton(j) == None:
                                    label = 0
                            else:
                                    label = 1
                            
                            if args.list == "var_1":
                                datas = variable_list1(evento, j)
                                
                                #datas = np.array([[j.Px(),j.Py(),j.Pz(),j.Pt(),j.E(), 
                                                   # maxpt, minpt, maxeta, maxm, minm]])
                            elif args.list == "var_2":
                                datas = variable_list2(evento, j)
                            elif args.list ==  "var_3":
                                datas = variable_list3(evento, j)
                            elif args.list == "var_4":
                                datas = variable_list4(evento, j)
                            pred = model.predict(datas)[0]
                            evento_jets_score.append(([pred, j, label, nevento]))
                        evento_jets_score= sorted(evento_jets_score, key= itemgetter(0), reverse=True)
                        #print([j[2] for j in evento_jets_score[:5]])
                        evento_jets_score = [j[1] for j in evento_jets_score[:g]]


                        #applying strategies
                        partons_pair1 = JetTagging.strategy_partons2(evento.partons)
                        jets_pair1 = strategy(evento_jets_score)
                        associazione = JetTagging.check_association(evento, partons_pair1.vbs_pair, jets_pair1.vbsjets)
                        #associazione vector boson
                        associazione2 = JetTagging.check_association(evento, partons_pair1.w_pair,  jets_pair1.vjets)
                        
                        #if associazione:
                            #due getti vbs indici corrispondono a partoni
                         #   h_vbs.Fill((jets_pair1.vbsjets[0]+jets_pair1.vbsjets[1]).M())
                        #else:
                            #due getti vbs indici NON corrispondono a partoni
                      #      h_vbs_bad.Fill((jets_pair1.vbsjets[0]+jets_pair1.vbsjets[1]).M())
                       # if associazione2:
                            #due getti V indici corrispondono a partonieff
                        #    h_v.Fill((jets_pair1.vjets[0]+jets_pair1.vjets[1]).M())
                        #else:
                            #due getti V indici NON corrispondono a partoni
                         #   h_v_bad.Fill((jets_pair1.vjets[0]+jets_pair1.vjets[1]).M())
                        
                        if associazione and associazione2:
                            cwrt = cwrt +1
                        else:
                            cwrg = cwrg +1
                    print(">>>Saving efficiency")
                    eff = cwrt/(cwrt+cwrg)
                    l.append([eff, g, models, strat[0]])
                    g = g+1
                    
            
            else:
                cwrt = 0
                cwrg = 0
                strategy = getattr(JetTagging, strat[0])
                output_dir = dir_output + "{}".format(strat[0])
                #hs_v, h_v, h_v_bad = plotter.StackCreator1(strat[1], strat[2], strat[3])
                #hs_vbs, h_vbs, h_vbs_bad = plotter.StackCreator2(strat[4],  strat[5], strat[6])
                th = 0.1
                while th < 1.:
                    print(">>>testing th {0}".format(th))
                    print(">>>>>Computing for event")
                    nevento = 0
                    re = 0
                    cwrt = 0
                    cwrg = 0
                    for evento in EventIterator(f,criteria = cuts, pairing = True) :
                        nevento += 1
                        """
                        pts = [j.Pt() for j in evento.jets]
                        etas = [abs(j.Eta()) for j in evento.jets]
                        ms = [j.M() for j in evento.jets]
                        maxpt = max(pts)
                        minpt = min(pts)
                        maxeta = max(etas)
                        maxm = max(ms)
                        minm = min(ms)
                        """
                        evento_jets_score = []
                        score_jets = []
                        for j in evento.jets:
                            if evento.paired_parton(j) == None:
                                    label = 0
                            else:
                                    label = 1
                            if args.list == "var_1":
                                datas = variable_list1(evento, j)
                                #datas = np.array([[j.Px(),j.Py(),j.Pz(),j.Pt(),j.E(), 
                                                   # maxpt, minpt, maxeta, maxm, minm]])
                                
                            elif args.list == "var_2":
                                datas = variable_list2(evento, j)
                                
                            elif args.list ==  "var_3":
                                datas = variable_list3(evento, j)
                            elif args.list == "var_4":
                                datas = variable_list4(evento, j)
                            elif args.list == "var_5":
                                datas = variable_list5(evento, j)
                            pred = model.predict(datas)[0]
                            evento_jets_score.append(([pred, j, label, nevento]))
                            
                        
                        evento_jets_score= sorted(evento_jets_score, key= itemgetter(0), reverse=True)
                        score_jets = [j[1] for j in evento_jets_score if j[0]>th]
                        while len(score_jets) < 4:
                            k = len(score_jets) + 1
                            score_jets = [j[1] for j in evento_jets_score[:k]]
                            
                        ###### computa quanti eventi hanno tutti e 4 i getti giusti ########
                        counter = 0
                        for j in evento_jets_score:
                            if j[0]>th:
                                if j[2] == 1:
                                    counter += 1
                        if counter == 4:
                            re+=1
                        for j in evento_jets_score:
                            if j[0]<th:
                                if j[2] == 0:
                                    print("ciao")
                        #applying strategies
                            
                        partons_pair1 = JetTagging.strategy_partons2(evento.partons)
                        jets_pair1 = strategy(score_jets)
                        associazione = JetTagging.check_association(evento, partons_pair1.vbs_pair, jets_pair1.vbsjets)
                        #associazione vector boson
                        associazione2 = JetTagging.check_association(evento, partons_pair1.w_pair,  jets_pair1.vjets)
                        if associazione and associazione2:
                            cwrt = cwrt +1
                        else:
                            cwrg = cwrg +1
                    print(">>> Saving efficiency")
                    eff = cwrt/(cwrt+cwrg)
                    num = re/nevento
                    l1.append([num, eff, th, models, strat[0]])
                    th = th + 0.1
                    th = round(th, 1)
                    
        print(">>> Saving Parameters")
        
        if args.val:
            
            f = open("/home/giacomo/tesi1/DNN_test/DNNTESTth_{}.txt".format(models), "w")
            i = 0
            while i < len(l1):
                f.write("{}\n".format(l1[i]))
                i = i+1
            f.close()
        else: 
            
            f = open("/home/giacomo/tesi1/DNN_test/DNNnjets_{}.txt".format(models), "w")
            i = 0
            while i < len(l):
                f.write("{}\n".format(l[i]))
                i = i+1
            f.close()
                    
