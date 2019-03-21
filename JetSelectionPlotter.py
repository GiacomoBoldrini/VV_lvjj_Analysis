import ROOT as r
import sys
from operator import attrgetter, itemgetter
from itertools import combinations
from VBSAnalysis.EventIterator import EventIterator
from VBSAnalysis.Utils import JetSelectors as jsel
from VBSAnalysis import JetTagging 
import argparse
import myplotter as plotter
plotter.setStyle()
from tqdm import tqdm

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
    

f = r.TFile("data/ewk_giacomo.root")

cuts = [
    ("pt_min_jets",30),
    ("min_njets",4),
    #("eq_njets",4),
    #("eta_max_jets", 2),
    ("eq_flag", 0),
    #("n_events", 100),
    ("atleastone_mjj_M", 250)
]

Vselectionlist = []
VBSselectionlist = []
"""
Vselectionlist.append(["nearest_W_pair", "Nearest W jets", "Nearest W good", "Nearest W bad", "Mjj Nearest W jets (GeV)"])

Vselectionlist.append(["nearest_Z_pair", "Nearest Z jets", "Nearest Z good", "Nearest Z bad", "Mjj Nearest Z jets (GeV)"])

Vselectionlist.append(["nearest_Z_or_W", "Nearest W/Z jets", "Nearest W/Z good", "Nearest W/Z bad", "Mjj Nearest W/Z jets (GeV)"])

Vselectionlist.append(["min_deltaeta_pair", "Min #Delta#eta jets", "Min #Delta#eta good", "Min #Delta#eta bad", "Mjj Min #Delta#eta jets (GeV)"])

Vselectionlist.append(["jet_min_size_pair", "Min #DeltaR jets", "Min #DeltaR good", "Min #DeltaR bad", "Mjj Min #DeltaR jets (GeV)"])

Vselectionlist.append(["max_pt_pair", "Max P_{t} jets", "Max P_{t} good", "Max P_{t} bad", "Mjj Max P_{t} jets (GeV)"])

Vselectionlist.append(["deltaeta_mw2_pair", "Composite Min #Delta#eta/Nearest W jets", "Min #Delta#eta/Nearest W good", "Min #Delta#eta/Nearest W bad", "Mjj Min #Delta#eta/Nearest W jets (GeV)"])

Vselectionlist.append(["combined_pair", "Composite Min #Delta#eta/Nearest W/Max P_{t} jets", "Min #Delta#eta/Nearest W/Max P_{t} good", "Min #Delta#eta/Nearest W/Max P_{t} bad", "Mjj Min #Delta#eta/Nearest W/Max P_{t} jets (GeV)"])

Vselectionlist.append(["mindeltaeta_wz_pair", "Composite Min #Delta#eta/Nearest W or Z jets", "Min #Delta#eta/Nearest W or Z good", "Min #Delta#eta/Nearest W or Z bad", "Mjj Min #Delta#eta/Nearest W or Z jets (GeV)"])

"""
VBSselectionlist.append(["max_mjj_pair", "Max Mjj jets", "Max Mjj good", "Max Mjj bad", "Mjj Max Mjj jets (GeV)"])

VBSselectionlist.append(["max_deltaeta_pair", "Max #Delta#eta jets", "Max #Delta#eta good", "Max #Delta#eta bad", "Mjj Max #Delta#eta jets (GeV)"])

VBSselectionlist.append(["jet_max_size_pair", "Max #DeltaR jets", "Max #DeltaR good", "Max #DeltaR bad", "Mjj Max #DeltaR jets (GeV)"])

VBSselectionlist.append(["deltaeta_mjj_pair", "Max #Delta#eta/ Mjj jets", "Max #Delta#eta/ Mjj good", "Max #Delta#eta/ Mjj bad", "Mjj Max #Delta#eta/ Mjj jets (GeV)"])



#ciclo per selezioni di getti da bosoni
previous_id = 0
l = []

for selection in Vselectionlist:

    strategy = getattr(jsel, selection[0])
    output_dir = "/home/giacomo/tesi1/VBSAnalysis/images/selections/V"
    mkdir_p(output_dir)
    hs_v, h_v, h_v_bad = plotter.StackCreator1(selection[1], selection[2], selection[3])
    print("computing...{}".format(selection[0]))
    previous_id = 0
    with tqdm(total=f.tree.GetEntries()) as pbar:
        for evento in EventIterator(f,criteria = cuts, pairing = True) :
            #pairing = True quindi PairingTreeDriver
            #jsel selettore di getti il cui metodo viene applicato anche ai partoni per creare le coppie
            partons_pair = jsel.nearest_Z_or_W(evento.partons)
            jets_pair = strategy(evento.jets)
            associazione = JetTagging.check_association(evento, partons_pair, [evento.jets[i] for i in jets_pair])
            if associazione:
                #se l'associazione va a buon fine restituisce TRUE e riempio l'histo con la massa dei due getti
                h_v.Fill( (evento.jets[jets_pair[0]]+ evento.jets[jets_pair[1]]).M())
            else :
                #se l'associazione non va a buon fine restituisce FALSE e inserisco nel secondo histo la massa dei due getti
                h_v_bad.Fill((evento.jets[jets_pair[0]]+ evento.jets[jets_pair[1]]).M())
            pbar.update(evento.evId - previous_id)
            previous_id = evento.evId
                
    eff = h_v.GetEntries()/(h_v.GetEntries()+h_v_bad.GetEntries())
    
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
    hratio = plotter.RatioDrawOptions(hratio, selection[4])
    hratio.GetXaxis().SetLabelSize(0.07)
    c1.Draw()
    c1.SaveAs(output_dir + "/{}.png".format(selection[0]))
    l.append([eff, selection[0]])

#ciclo per selezioni di getti di tag
l1 = []

for selection in VBSselectionlist:

    strategy = getattr(jsel, selection[0])
    output_dir = "/home/giacomo/tesi1/VBSAnalysis/images/selections/VBS"
    mkdir_p(output_dir)
    hs_vbs, h_vbs, h_vbs_bad = plotter.StackCreator2(selection[1], selection[2], selection[3])
    print("computing...{}".format(selection[0]))
    previous_id = 0
    with tqdm(total=f.tree.GetEntries()) as pbar:
        for evento in EventIterator(f,criteria = cuts, pairing = True) :
            #pairing = True quindi PairingTreeDriver
            #jsel selettore di getti il cui metodo viene applicato anche ai partoni per creare le coppie
            partons_pair = jsel.tag_parton_pair(evento.partons)
            jets_pair = strategy(evento.jets)
            associazione = JetTagging.check_association(evento, partons_pair, [evento.jets[i] for i in jets_pair])
            if associazione:
                #se l'associazione va a buon fine restituisce TRUE e riempio l'histo con la massa dei due getti
                h_vbs.Fill( (evento.jets[jets_pair[0]]+ evento.jets[jets_pair[1]]).M())
            else :
                #se l'associazione non va a buon fine restituisce FALSE e inserisco nel secondo histo la massa dei due getti
                h_vbs_bad.Fill((evento.jets[jets_pair[0]]+ evento.jets[jets_pair[1]]).M())
            pbar.update(evento.evId - previous_id)
            previous_id = evento.evId
                
    eff = h_vbs.GetEntries()/(h_vbs.GetEntries()+h_vbs_bad.GetEntries())
    
    c1, pad1, pad2 = plotter.createCanvasPads()

    pad1.cd()
    hs_vbs.Draw("nostack hist")
    legend = plotter.createLegend(h_vbs, h_vbs_bad,1)
    legend.Draw("same")
    pad2.cd()
    hratio = plotter.createRatio(h_vbs, h_vbs +h_vbs_bad, "eff.")
    hratio.Draw("hist")
    hratio.SetFillColor(0)
    hratioerror = hratio.DrawCopy("E2 same")
    hratioerror.SetFillStyle(3013)
    hratioerror.SetFillColor(13)
    hratioerror.SetMarkerStyle(1)
    hratio = plotter.RatioDrawOptions(hratio, selection[4])
    hratio.GetXaxis().SetLabelSize(0.07)
    c1.Draw()
    c1.SaveAs(output_dir + "/{}.png".format(selection[0]))
    l1.append([eff, selection[0]])
    
l = sorted(l, key=itemgetter(0), reverse = True)
l1 = sorted(l1, key = itemgetter(0), reverse = True)

f = open("/home/giacomo/tesi1/VBSAnalysis/images/SelectionVEfficiencies.txt", "w")
i = 0
while i < len(l):
    f.write("{}\n".format(l[i]))
    i = i+1
f.close()

g = open("/home/giacomo/tesi1/VBSAnalysis/images/SelectionVBSEfficiencies.txt", "w")
i = 0
while i < len(l1):
    g.write("{}\n".format(l1[i]))
    i = i+1
g.close()
