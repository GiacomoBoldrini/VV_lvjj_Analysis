from rootpy.tree import Tree, TreeModel
from rootpy.tree import IntCol, FloatCol, DoubleCol,DoubleArrayCol, BoolCol
from ROOT import TLorentzVector
from math import pi
from . import RecoNeutrino
from .Utils.TreeWriter import TreeWriter


class OutputTree:
    def __init__(self, tree_name):
        self.tree = TreeWriter(tree_name)
        self.tree.define_branches({
            1: {
                float: ["xs_weight", "mjj_vbs", "mjj_vjet",
                    "vbs_pt_high", "vbs_pt_low", "vbs_etaprod",
                    "vjet_pt_high", "vjet_pt_low", "mu_pt", 
                    "mu_phi", "mu_eta", "met", "met_phi", "deltaeta_vbs",
                    "deltaphi_vbs", "deltaeta_vjet", "deltaphi_vjet", 
                    "deltaphi_mu_vbs_high", "deltaphi_mu_vbs_low", "deltaeta_mu_vbs_high",
                    "deltaeta_mu_vbs_low", "deltaphi_mu_vjet_high", "deltaphi_mu_vjet_low",
                    "deltaeta_mu_vjet_high", "deltaeta_mu_vjet_low",
                    "deltaR_mu_vbs", "deltaR_mu_vjet",
                    "deltaphi_mu_nu", "deltaeta_mu_nu",
                    "deltaR_mu_nu", "deltaR_vbs", "deltaR_vjet",
                    "Rvjets_high", "Rvjets_low", "Zvjets_high", "Zvjets_low", "Zmu",
                    "A_vbs", "A_vjet", "Mw_lep", "w_lep_pt", "Mww", "R_ww", "R_mw", "A_ww",
                    "C_vbs", "C_ww", "L_p", "L_pw", "Ht"          
                    ],
                int: ["N_jets", "N_jets_forward", "N_jets_central"]
                },
            2: { float: ["eta_vjet", "eta_vbs"] },
            3: { float: ["bveto_weights"]}
        })
    
    def write_event(self, event, vbsjets, vjets, xs_weight, bveto_weights):
        # event weights
        self.tree.xs_weight = xs_weight
        self.tree.bveto_weights = bveto_weights

        # variables extraction
        total_vbs = TLorentzVector(0,0,0,0)
        vbs_etas = []
        vbs_phis = []
        vbs_pts = []
        for i, j in enumerate(vbsjets):
            total_vbs+= j
            vbs_etas.append(j.Eta())
            vbs_phis.append(j.Phi())
            vbs_pts.append(j.Pt())
        deltaeta_vbs = abs(vbs_etas[0]- vbs_etas[1])
        mean_eta_vbs = sum(vbs_etas) / 2 
        self.tree.vbs_pt_high = vbs_pts[0]
        self.tree.vbs_pt_low = vbs_pts[1]
        self.tree.mjj_vbs = total_vbs.M()
        self.tree.deltaeta_vbs = deltaeta_vbs
        self.tree.deltaphi_vbs = abs(vbsjets[0].DeltaPhi(vbsjets[1]))
        self.tree.deltaR_vbs = vbsjets[0].DrEtaPhi(vbsjets[1])
        self.tree.vbs_etaprod = vbs_etas[0]*vbs_etas[1]
        self.tree.eta_vbs = list(map(abs, vbs_etas))

        total_vjet = TLorentzVector(0,0,0,0)
        vjet_etas = []
        vjet_phis = []
        vjet_pts = []
        for i, j in enumerate(vjets):
            total_vjet += j
            vjet_etas.append(j.Eta())
            vjet_phis.append(j.Phi())
            vjet_pts.append(j.Pt())
        self.tree.vjet_pt_high = vjet_pts[0]
        self.tree.vjet_pt_low = vjet_pts[1]
        self.tree.mjj_vjet = total_vjet.M()
        self.tree.deltaphi_vjet =  abs(vjets[0].DeltaPhi(vjets[1]))
        self.tree.deltaeta_vjet = abs(vjet_etas[0] - vjet_etas[1])
        self.tree.deltaR_vjet = vjets[0].DrEtaPhi(vjets[1])
        self.tree.eta_vjet = list(map(abs, vjet_etas))

        #Save muon info
        muon_vec = event.muon
        nu_vec = RecoNeutrino.reconstruct_neutrino(muon_vec, event.P_met)
        self.tree.mu_pt = muon_vec.Pt()
        self.tree.mu_eta = abs(muon_vec.Eta())
        self.tree.mu_phi = abs(muon_vec.Phi())
        self.tree.met = nu_vec.Pt()
        self.tree.met_phi = abs(nu_vec.Phi())
        self.tree.deltaphi_mu_nu = abs(muon_vec.DeltaPhi(nu_vec)) 
        self.tree.deltaeta_mu_nu = abs(muon_vec.Eta() - nu_vec.Eta())
        self.tree.deltaR_mu_nu = muon_vec.DrEtaPhi(nu_vec)

        # Delta Phi with muon
        self.tree.deltaphi_mu_vbs_high = abs(muon_vec.DeltaPhi(vbsjets[0]))
        self.tree.deltaphi_mu_vbs_low = abs(muon_vec.DeltaPhi(vbsjets[1]))
        self.tree.deltaphi_mu_vjet_high = abs(muon_vec.DeltaPhi(vjets[0]))
        self.tree.deltaphi_mu_vjet_low = abs(muon_vec.DeltaPhi(vjets[1]))

        # Delta Eta with muon
        self.tree.deltaeta_mu_vbs_high = abs(muon_vec.Eta() - vbs_etas[0])
        self.tree.deltaeta_mu_vbs_low= abs(muon_vec.Eta() - vbs_etas[1])
        self.tree.deltaeta_mu_vjet_high = abs(muon_vec.Eta() - vjet_etas[0])
        self.tree.deltaeta_mu_vjet_low = abs(muon_vec.Eta() - vjet_etas[1])
         
        # Look for nearest vbs jet from muon
        self.tree.deltaR_mu_vbs = min( [ muon_vec.DrEtaPhi(vbsjets[0]), 
                        muon_vec.DrEtaPhi(vbsjets[1])])
        self.tree.deltaR_mu_vjet = min( [ muon_vec.DrEtaPhi(vjets[0]), 
                        muon_vec.DrEtaPhi(vjets[1])])


        # Zeppenfeld variables
        self.tree.Zvjets_high = (vjet_etas[0] - mean_eta_vbs)/ deltaeta_vbs
        self.tree.Zvjets_low = (vjet_etas[1] - mean_eta_vbs)/ deltaeta_vbs
        self.tree.Zmu = (muon_vec.Eta() - mean_eta_vbs)/ deltaeta_vbs

        #R variables
        ptvbs12 = vbsjets[0].Pt() * vbsjets[1].Pt() 
        self.tree.Rvjets_high = (muon_vec.Pt() * vjets[0].Pt()) / ptvbs12
        self.tree.Rvjets_low = (muon_vec.Pt() * vjets[1].Pt()) / ptvbs12

        #Asymmetry
        self.tree.A_vbs = (vbs_pts[0] - vbs_pts[1]) / sum(vbs_pts)
        self.tree.A_vjet = (vjet_pts[0] - vjet_pts[1]) / sum(vjet_pts)

        #WW variables
        w_lep = muon_vec + nu_vec
        w_had = vjets[0] + vjets[1]
        w_lep_t = w_lep.Vect()
        w_lep_t.SetZ(0)
        w_had_t = w_had.Vect()
        w_had_t.SetZ(0)
        ww_vec = w_lep + w_had
        self.tree.w_lep_pt = w_lep.Pt()
        self.tree.Mw_lep = w_lep.M()
        self.tree.Mww = ww_vec.M()
        self.tree.R_ww = (w_lep.Pt() * w_lep.Pt()) / ptvbs12
        self.tree.R_mw = ww_vec.M() / ptvbs12
        self.tree.A_ww = (w_lep_t + w_had_t).Pt() / (w_lep.Pt() + w_had.Pt())
        
        #Centrality
        eta_ww = (w_lep.Eta() + w_had.Eta())/2
        self.tree.C_vbs = abs(vbs_etas[0] - eta_ww - vbs_etas[1]) / deltaeta_vbs
        deltaeta_plus = max(vbs_etas) - max([w_lep.Eta(), w_had.Eta()])
        deltaeta_minus = min([w_lep.Eta(), w_had.Eta()]) - min(vbs_etas)
        self.tree.C_ww = min([deltaeta_plus, deltaeta_minus])

        #Lepton projection
        muon_vec_t = muon_vec.Vect()
        muon_vec_t.SetZ(0)
        self.tree.L_p = (w_lep_t * muon_vec_t) / w_lep.Pt()
        self.tree.L_pw = (w_lep_t * muon_vec_t) / (muon_vec.Pt() * w_lep.Pt())

        # Ht and number of jets with Pt> 20
        # using uncut jets
        Njets = 0
        N_jets_forward = 0
        N_jets_central = 0
        Ht = 0.
        for j in event.jets:
            if j not in vbsjets and j not in vjets:
                # Looking only to jets != vbs & vjets
                Z = abs((j.Eta() - mean_eta_vbs)/ deltaeta_vbs)
                Njets += 1
                if Z > 0.5:
                    N_jets_forward += 1
                else:
                    N_jets_central += 1
            # Ht totale
            Ht += j.Pt()
                
        self.tree.N_jets = Njets 
        self.tree.N_jets_central = N_jets_central
        self.tree.N_jets_forward = N_jets_forward
        self.tree.Ht = Ht
       
        #Fill the tree
        self.tree.fill()

    def write(self):
        self.tree.write()


    def to_csv(self, filename, nevents):
       self.tree.csv(  sep="|",
                       stream=open(filename, "w"),
                       limit = nevents )
