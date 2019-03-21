
int JES(char* input, char* output, char* param, int direction){
    TFile * inputfile = new TFile(input, "read");
    TTree * inputtree = (TTree*) inputfile->Get("tree");
    // Update if the outputfile exists
    TFile * outputfile = new TFile(output, "update");
    string outtree_name = "";
    if (direction == +1){
        outtree_name = "jes_up";
    }else if(direction == -1){
        outtree_name = "jes_down";
    }
    TTree * outputtree = new TTree(outtree_name.c_str(),outtree_name.c_str());
    int njets = 0;  
    double E_jets[25];
    double px_jets[25];
    double py_jets[25];
    double pz_jets[25];
    double p_mu[4];
    double p_nu[4];
    double p_mu_lhe[4];
    double met[4];
    double unclustered_met[4]; 
    double E_parton[4];
    double px_parton[4];
    double py_parton[4];
    double pz_parton[4];
    int partons_flavour[4];
    int npartons = 0; 
    inputtree->SetBranchAddress("njets", &njets);
    inputtree->SetBranchAddress("E_jets", &E_jets);
    inputtree->SetBranchAddress("px_jets", &px_jets);
    inputtree->SetBranchAddress("py_jets", &py_jets);
    inputtree->SetBranchAddress("pz_jets", &pz_jets);
    inputtree->SetBranchAddress("p_mu", &p_mu);
    inputtree->SetBranchAddress("p_mu_lhe", &p_mu_lhe);
    inputtree->SetBranchAddress("p_nu", &p_nu);
    inputtree->SetBranchAddress("met", &met);
    inputtree->SetBranchAddress("unclustered_met", &unclustered_met);
    inputtree->SetBranchAddress("npartons", &npartons);
    inputtree->SetBranchAddress("E_parton", &E_parton);
    inputtree->SetBranchAddress("px_parton", &px_parton);
    inputtree->SetBranchAddress("py_parton", &py_parton);
    inputtree->SetBranchAddress("pz_parton", &pz_parton);
    inputtree->SetBranchAddress("partons_flavour", &partons_flavour);
    outputtree->Branch("njets", &njets, "njets/I");
    outputtree->Branch("E_jets", &E_jets, "E_jets[njets]/D");
    outputtree->Branch("px_jets", &px_jets, "px_jets[njets]/D");
    outputtree->Branch("py_jets", &py_jets, "py_jets[njets]/D");
    outputtree->Branch("pz_jets", &pz_jets, "pz_jets[njets]/D");
    outputtree->Branch("p_mu", &p_mu, "p_mu[4]/D");
    outputtree->Branch("p_mu_lhe", &p_mu_lhe, "p_mu_lhe[4]/D"); 
    outputtree->Branch("p_nu", &p_nu, "p_nu[4]/D"); 
    outputtree->Branch("met", &met, "met[4]/D");
    outputtree->Branch("unclustered_met", &unclustered_met, "unclustered_met[4]/D");
    outputtree->Branch("npartons", &npartons, "npartons/I");
    outputtree->Branch("E_parton", &E_parton, "E_parton[npartons]/D");
    outputtree->Branch("px_parton", &px_parton, "px_parton[npartons]/D");
    outputtree->Branch("py_parton", &py_parton, "py_parton[npartons]/D");
    outputtree->Branch("pz_parton", &pz_parton, "pz_parton[npartons]/D");
    outputtree->Branch("partons_flavour", &partons_flavour, "partons_flavour[4]/I");
  
    JetCorrectionUncertainty * jes = new JetCorrectionUncertainty(param);
    int entries = inputtree->GetEntries();
    cout << "N entries: " << entries << endl;

    for (int i = 0; i< entries; i++){
        inputtree->GetEntry(i);
        for (int j = 0; j<njets; j++){
            auto v = TLorentzVector(px_jets[j], py_jets[j], pz_jets[j], E_jets[j]);
            if (abs(v.Eta())>=5.4){
                continue;
            }
            jes->setJetEta(v.Eta());
            jes->setJetPt(v.Pt());
            float uncer = jes->getUncertainty(true);
            if (uncer > 0){
                double new_Pt = v.Pt()*(1+ direction*uncer);
                double new_M = v.M()*(1+ direction*uncer);
                v.SetPtEtaPhiM(new_Pt, v.Eta(), v.Phi(), new_M);
                px_jets[j] = v.Px();
                py_jets[j] = v.Py();
                pz_jets[j] = v.Pz();
                E_jets[j] = v.E();
                if (i% 500000==0){
                    cout << i <<">>> Pt: "<< v.Pt() <<" eta: " << v.Eta() << " unc: "<< uncer <<endl;
                }
            }else{
                cout <<"Out-of-range >>> Pt: "<< v.Pt() <<" eta: " << v.Eta() <<endl;
            }
        }
        outputtree->Fill();
           
    }

    outputfile->Write("", TFile::kOverwrite);
    inputfile->Close();
    outputfile->Close();
    return 0;

}
