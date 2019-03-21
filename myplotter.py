import ROOT as ROOT
import argparse

def setStyle():
  style = ROOT.gStyle
  style.SetPalette(ROOT.kDarkRainBow)
  style.SetOptStat(1111)
  style.SetOptFit(1111)
  style.cd()
  
def getCanvas():
    H = 800; 
    W = 1000; 
    T = 0.08
    B = 0.11 
    L = 0.13
    R = 0.04
    canvas = ROOT.TCanvas("c1","c1",50,50,W,H)
    canvas.SetFillColor(0)
    canvas.SetBorderMode(0)
    canvas.SetFrameFillStyle(0)
    canvas.SetFrameBorderMode(0)
    #canvas.SetMargin(L,R, B, T)
    canvas.SetTickx(1)
    canvas.SetTicky(1) 
    return canvas

def createCanvasPads():	# Create Canvas having two pads
    H = 800; 
    W = 1000; 
    T = 0.08
    B = 0.10 
    L = 0.13
    R = 0.04
    c = ROOT.TCanvas("c", "canvas",50, 50, W, H)
    # Upper histogram plot is pad1
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.28, 1, 1)
    pad1.SetBottomMargin(0)  # joins upper and lower plot
    pad1.SetTickx(1)
    pad1.SetTicky(1)
    pad1.SetGridx()
    pad1.Draw()
    # Lower ratio plot is pad2
    c.cd()  # returns to main canvas before defining pad2
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.28)
    pad2.SetTopMargin(0)  # joins upper and lower plot
    pad2.SetBottomMargin(0.392)
    pad2.SetGridx()
    pad2.SetGridy()
    pad2.SetTickx(1)
    pad2.SetTicky(1)
    pad2.Draw()

    return c, pad1, pad2

def createRatio(h1, h2, label="ratio"):	# h1/h2
    h3 = h1.Clone("h3")
    h3.SetMarkerStyle(21)
    h3.SetMarkerColor(ROOT.kBlack)
    h3.SetLineColor(ROOT.kBlack)
    h3.SetTitle("")
    # Set up plot for markers and errors
    #h3.Sumw2()
    h3.SetStats(0)
    h3.Divide(h2)
    #Fix automatically the ratio range
    h3.SetMaximum(h3.GetMaximum() + 0.05)
    h3.SetMinimum(h3.GetMinimum() - 0.05)
    
    # Adjust y-axis settings
    h3.GetYaxis().SetTitle(label)
    return h3

def createLegend(h1, h2, s):
    if s == 1 :
        legend = ROOT.TLegend(.9,.9,0.7,0.7)
    if s == 2 :
        legend = ROOT.TLegend(0.1,0.6,0.3,0.9)
        legend.SetBorderSize(0)
    legend.AddEntry(h1)
    legend.AddEntry(h2)
    
    return legend

def RatioDrawOptions(h1, string):
    h1.GetXaxis().SetTitle(string)
    h1.GetXaxis().SetTitleSize(0.1)
    h1.GetYaxis().CenterTitle()
    h1.GetYaxis().SetTitleSize(0.1)
    h1.GetYaxis().SetTitleOffset(0.3)
    h1.GetYaxis().SetLabelSize(0.05)
    h1.GetXaxis().SetLabelSize(0.07)
    
    return h1

def StackCreator1(hslabel, h1label, h2label):
    hs= ROOT.THStack(hslabel,hslabel);
    h1 = ROOT.TH1F(h1label, h1label, 60, 20, 150 )
    h1.SetLineColor(ROOT.kRed)
    h1.SetLineWidth(2)
    h1.SetFillColor(ROOT.kRed)
    h1.SetFillStyle(3003)
    hs.Add(h1)
    h2 = ROOT.TH1F(h2label, h2label, 60, 20, 150 )
    h2.SetLineColor(ROOT.kBlue)
    h2.SetFillColor(ROOT.kBlue)
    h2.SetLineWidth(2)
    h2.SetFillStyle(3003)
    hs.Add(h2)
    
    return hs, h1, h2

def StackCreator2(hslabel, h1label, h2label):
    hs= ROOT.THStack(hslabel,hslabel);
    h1 = ROOT.TH1F(h1label, h1label, 60, 20, 600 )
    h1.SetLineColor(ROOT.kRed)
    h1.SetLineWidth(2)
    h1.SetFillColor(ROOT.kRed)
    h1.SetFillStyle(3003)
    hs.Add(h1)
    h2 = ROOT.TH1F(h2label, h2label, 60, 20, 600 )
    h2.SetLineColor(ROOT.kBlue)
    h2.SetLineWidth(2)
    h2.SetFillColor(ROOT.kBlue)
    h2.SetFillStyle(3003)
    hs.Add(h2)
    
    return hs, h1, h2
"""
def StackCreator1(hslabel, h1label, h2label):
    hs= ROOT.THStack(hslabel,hslabel);
    h1 = ROOT.TH1F(h1label, h1label, 60, -6, 6 )
    h1.SetLineColor(ROOT.kRed)
    h1.SetLineWidth(2)
    h1.SetFillColor(ROOT.kRed)
    h1.SetFillStyle(3003)
    hs.Add(h1)
    h2 = ROOT.TH1F(h2label, h2label, 60, -6, 6 )
    h2.SetLineColor(ROOT.kBlue)
    h2.SetFillColor(ROOT.kBlue)
    h2.SetLineWidth(2)
    h2.SetFillStyle(3003)
    hs.Add(h2)
    
    return hs, h1, h2

def StackCreator2(hslabel, h1label, h2label):
    hs= ROOT.THStack(hslabel,hslabel);
    h1 = ROOT.TH1F(h1label, h1label, 60, -10, 10 )
    h1.SetLineColor(ROOT.kRed)
    h1.SetLineWidth(2)
    h1.SetFillColor(ROOT.kRed)
    h1.SetFillStyle(3003)
    hs.Add(h1)
    h2 = ROOT.TH1F(h2label, h2label, 60, -10, 10 )
    h2.SetLineColor(ROOT.kBlue)
    h2.SetLineWidth(2)
    h2.SetFillColor(ROOT.kBlue)
    h2.SetFillStyle(3003)
    hs.Add(h2)
    
    return hs, h1, h2
"""
def StackCreator3(hslabel, h1label, h2label):
    hs= ROOT.THStack(hslabel,hslabel);
    h1 = ROOT.TH1F(h1label, h1label, 60, 20, 600 )
    h1.SetLineColor(ROOT.kRed)
    h1.SetLineWidth(2)
    h1.SetFillColor(ROOT.kRed)
    h1.SetFillStyle(3003)
    hs.Add(h1)
    h2 = ROOT.TH1F(h2label, h2label, 60, 20, 600 )
    h2.SetLineColor(ROOT.kBlue)
    h2.SetLineWidth(2)
    h2.SetFillColor(ROOT.kBlue)
    h2.SetFillStyle(3003)
    hs.Add(h2)
    
    return hs, h1, h2
