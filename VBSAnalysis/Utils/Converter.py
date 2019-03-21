from ROOT import TLorentzVector

def convertToLorentzVector(njets, px, py, pz, E):
    return [TLorentzVector(
                px[i], py[i],
                pz[i], E[i] 
            ) for i in range(njets)]

def convertToLorentzVector_single(px, py, pz, E):
    return TLorentzVector(px, py, pz, E)