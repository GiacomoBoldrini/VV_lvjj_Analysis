import ROOT as r 
import os

# Load the map file with the 4 parameters
map_file = r.TFile(os.path.dirname(__file__) + "/maps.root")
m_par_0 = map_file.Get("m_par_0")
m_par_1 = map_file.Get("m_par_1")
m_par_2 = map_file.Get("m_par_2")
m_par_3 = map_file.Get("m_par_3")
# resolution formula
resolution_f = r.TF1("resolution", 
    "sqrt([0]*abs([0])/(x*x)+[1]*[1]*pow(x,[3])+[2]*[2])",
    15, 3000);  # The Pt range is between 15 and 3000 GeV


# Default value pileup density rho=20
def resolution( eta, pt, rho=25 ):
    eta_bin = m_par_0.GetXaxis().FindBin(eta)
    rho_bin = m_par_0.GetYaxis().FindBin(rho)

    val_0 = m_par_0.GetBinContent(eta_bin, rho_bin)
    val_1 = m_par_1.GetBinContent(eta_bin, rho_bin)
    val_2 = m_par_2.GetBinContent(eta_bin, rho_bin)
    val_3 = m_par_3.GetBinContent(eta_bin, rho_bin)

    resolution_f.SetParameter(0, val_0)
    resolution_f.SetParameter(1, val_1)
    resolution_f.SetParameter(2, val_2)
    resolution_f.SetParameter(3, val_3)

    return resolution_f.Eval(pt)


def apply_JER(events_iterator, direction):
    '''
    Apply the JER up/down directly on the jets of
    the event object.
    '''
    for event in events_iterator:
        for jet in event.jets:
            res = resolution(jet.Eta(), jet.Pt())   
            oldE = jet.E()
            if direction=="up":
                newE = oldE + oldE*res
            elif direction=="down":
                newE = oldE - oldE*res
            jet.SetE(newE)
            #print("Pt: {} | Eta: {} >>> {} | Test: {}".format(jet.Pt(), jet.Eta(), res, jet.E()/ oldE ))
        #Return the modified event
        yield event

def apply_JER_single(jet, direction):
    res = resolution(jet.Eta(), jet.Pt())
    oldE = jet.E()
    if direction=="up":
        newE = oldE + oldE*res
    elif direction=="down":
        newE = oldE - oldE*res
    jet.SetE(newE)
    return jet
