import ROOT as ROOT
import argparse

#############	Define Legend, Canvas, 	##########################

def getLegends(pos,ncol,nvar, textsize=0.04, xmin= 0.77):
    if pos == "tr":
    	legend = ROOT.TLegend(xmin-(0.15*(ncol-1)), 0.70-(0.02*(nvar/ncol-1)) ,.90 ,.90)
    elif pos == "tl":
    	legend = ROOT.TLegend(0.15, 0.70-(0.02*(nvar/ncol-1)),.40+(0.15*(ncol-1)) ,.88)
    elif pos == "tc":
    	legend = ROOT.TLegend(0.25, 0.70-(0.02*(nvar/ncol-1)) ,.80 ,.88)
    elif pos == "bl":
    	legend = ROOT.TLegend(0.11, 0.25+(0.02*(nvar/ncol-1)) ,.36+(0.15*(ncol-1)) ,.130)
    elif pos == "bc":
    	legend = ROOT.TLegend(0.30, 0.05+(0.02*(nvar/ncol-1)) ,.65 ,.20)
    elif pos == "br":
    	legend = ROOT.TLegend(.45-(0.15*(ncol-1)), 0.25+(0.02*(nvar/ncol-1)) ,.95 ,.130)
    else:
    	print ("Invalid default position: Switching to default legend position top-right")
    	legend = ROOT.TLegend(.85-(0.15*(ncol-1)), 0.88-(0.02*(nvar/ncol-1)) ,.50 ,.950)
    legend.SetFillColor(ROOT.kWhite)
    legend.SetFillStyle(0)
    legend.SetLineColor(0)
    legend.SetTextSize(textsize)
    legend.SetNColumns(ncol)
    return legend

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
    canvas.SetMargin(L,R, B, T)
    canvas.SetTickx(1)
    canvas.SetTicky(1) 
    return canvas

def createCanvasPads():	# Create Canvas having two pads
    H = 800; 
    W = 800; 
    T = 0.08
    B = 0.10 
    L = 0.13
    R = 0.04
    c = ROOT.TCanvas("c", "canvas",50, 50, W, H)
    # Upper histogram plot is pad1
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.28, 1, 1)
    pad1.SetBottomMargin(0)  # joins upper and lower plot
    pad1.SetRightMargin(R)
    pad1.SetLeftMargin(L)
    pad1.SetTickx(1)
    pad1.SetTicky(1)
    pad1.Draw()
    # Lower ratio plot is pad2
    c.cd()  # returns to main canvas before defining pad2
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.28)
    pad2.SetTopMargin(0)  # joins upper and lower plot
    pad2.SetRightMargin(R)
    pad2.SetLeftMargin(L)
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

def fixOverflowBins(histo):
    ''' This function add the content of the underflow and overflow bins
    in the first and last bin
    '''
    nbin = histo.GetNbinsX()
    #print(histo.GetBinContent(nbin+1), histo.GetBinContent(nbin))
    histo.AddBinContent(nbin, histo.GetBinContent(nbin+1))
    histo.SetBinContent(nbin+1, 0)
    histo.AddBinContent(1, histo.GetBinContent(0))
    histo.SetBinContent(0, 0)

def setHistoStyle(hist, ratio):
    label_size = 26
    title_size = 28
    yaxis = hist.GetYaxis()
    yaxis.SetLabelFont ( 43)
    yaxis.SetLabelOffset( 0.01)
    yaxis.SetLabelSize ( label_size)
    yaxis.SetNdivisions ( 505)
    yaxis.SetTitleFont ( 43)
    yaxis.SetTitleOffset( 1.8)
    yaxis.SetTitleSize ( title_size)
    xaxis = hist.GetXaxis()
    xaxis.SetLabelFont (43)
    xaxis.SetLabelSize ( label_size)
    xaxis.SetNdivisions ( 505)
    xaxis.SetTitleFont ( 43)
    xaxis.SetTitleSize ( title_size)
    if ratio:
        yaxis.CenterTitle()
        xaxis.SetLabelOffset( 0.025)
        xaxis.SetTitleOffset(4)
       

def setStyle():
  style = ROOT.gStyle
  style.SetPalette(ROOT.kDarkRainBow)
  style.SetOptStat(0)
  style.cd()

def setTDRStyle():
  tdrStyle =  ROOT.TStyle("tdrStyle","Style for P-TDR")

   #for the canvas:
  tdrStyle.SetCanvasBorderMode(0)
  tdrStyle.SetCanvasColor(ROOT.kWhite)
  tdrStyle.SetCanvasDefH(600) #Height of canvas
  tdrStyle.SetCanvasDefW(600) #Width of canvas
  tdrStyle.SetCanvasDefX(0)   #POsition on screen
  tdrStyle.SetCanvasDefY(0)


  tdrStyle.SetPadBorderMode(0)
  #tdrStyle.SetPadBorderSize(Width_t size = 1)
  tdrStyle.SetPadColor(ROOT.kWhite)
  tdrStyle.SetPadGridX(False)
  tdrStyle.SetPadGridY(False)
  tdrStyle.SetGridColor(0)
  tdrStyle.SetGridStyle(3)
  tdrStyle.SetGridWidth(1)

#For the frame:
  tdrStyle.SetFrameBorderMode(0)
  tdrStyle.SetFrameBorderSize(1)
  tdrStyle.SetFrameFillColor(0)
  tdrStyle.SetFrameFillStyle(0)
  tdrStyle.SetFrameLineColor(1)
  tdrStyle.SetFrameLineStyle(1)
  tdrStyle.SetFrameLineWidth(1)
  
#For the histo:
  #tdrStyle.SetHistFillColor(1)
  #tdrStyle.SetHistFillStyle(0)
  tdrStyle.SetHistLineColor(1)
  tdrStyle.SetHistLineStyle(0)
  tdrStyle.SetHistLineWidth(1)
  #tdrStyle.SetLegoInnerR(Float_t rad = 0.5)
  #tdrStyle.SetNumberContours(Int_t number = 20)

  tdrStyle.SetEndErrorSize(2)
  #tdrStyle.SetErrorMarker(20)
  #tdrStyle.SetErrorX(0.)
  
  tdrStyle.SetMarkerStyle(20)
  
#For the fit/function:
  tdrStyle.SetOptFit(1)
  tdrStyle.SetFitFormat("5.4g")
  tdrStyle.SetFuncColor(2)
  tdrStyle.SetFuncStyle(1)
  tdrStyle.SetFuncWidth(1)

#For the date:
  tdrStyle.SetOptDate(0)
  # tdrStyle.SetDateX(Float_t x = 0.01)
  # tdrStyle.SetDateY(Float_t y = 0.01)

# For the statistics box:
  tdrStyle.SetOptFile(0)
  tdrStyle.SetOptStat(0) # To display the mean and RMS:   SetOptStat("mr")
  tdrStyle.SetStatColor(ROOT.kWhite)
  tdrStyle.SetStatFont(42)
  tdrStyle.SetStatFontSize(0.025)
  tdrStyle.SetStatTextColor(1)
  tdrStyle.SetStatFormat("6.4g")
  tdrStyle.SetStatBorderSize(1)
  tdrStyle.SetStatH(0.1)
  tdrStyle.SetStatW(0.15)
  # tdrStyle.SetStatStyle(Style_t style = 1001)
  # tdrStyle.SetStatX(Float_t x = 0)
  # tdrStyle.SetStatY(Float_t y = 0)

# Margins:
  tdrStyle.SetPadTopMargin(0.05)
  tdrStyle.SetPadBottomMargin(0.13)
  tdrStyle.SetPadLeftMargin(0.22)
  tdrStyle.SetPadRightMargin(0.12)

# For the Global title:

  tdrStyle.SetOptTitle(0)
  tdrStyle.SetTitleFont(42)
  tdrStyle.SetTitleColor(1)
  tdrStyle.SetTitleTextColor(1)
  tdrStyle.SetTitleFillColor(10)
  tdrStyle.SetTitleFontSize(0.03)
  # tdrStyle.SetTitleH(0) # Set the height of the title box
  # tdrStyle.SetTitleW(0) # Set the width of the title box
  # tdrStyle.SetTitleX(0) # Set the position of the title box
  # tdrStyle.SetTitleY(0.985) # Set the position of the title box
  # tdrStyle.SetTitleStyle(Style_t style = 1001)
  # tdrStyle.SetTitleBorderSize(2)

# For the axis titles:

  tdrStyle.SetTitleColor(1, "XYZ")
  tdrStyle.SetTitleFont(42, "XYZ")
  tdrStyle.SetTitleSize(0.04, "XYZ")
  # tdrStyle.SetTitleXSize(Float_t size = 0.02) # Another way to set the size?
  # tdrStyle.SetTitleYSize(Float_t size = 0.02)
  tdrStyle.SetTitleXOffset(0.75)
  tdrStyle.SetTitleYOffset(1.10)
  # tdrStyle.SetTitleOffset(1.1, "Y") # Another way to set the Offset

# For the axis labels:

  tdrStyle.SetLabelColor(1, "XYZ")
  tdrStyle.SetLabelFont(42, "XYZ")
  tdrStyle.SetLabelOffset(0.007, "XYZ")
  tdrStyle.SetLabelSize(0.03, "XYZ")

# For the axis:

  tdrStyle.SetAxisColor(1, "XYZ")
  tdrStyle.SetStripDecimals(True)
  tdrStyle.SetTickLength(0.03, "XYZ")
  tdrStyle.SetNdivisions(510, "XYZ")
  tdrStyle.SetPadTickX(1)  # To get tick marks on the opposite side of the frame
  tdrStyle.SetPadTickY(1)

# Change for log plots:
  tdrStyle.SetOptLogx(0)
  tdrStyle.SetOptLogy(0)
  tdrStyle.SetOptLogz(0)

# Postscript options:
  tdrStyle.SetPaperSize(20.,20.)
  # tdrStyle.SetLineScalePS(Float_t scale = 3)
  # tdrStyle.SetLineStyleString(Int_t i, const char* text)
  # tdrStyle.SetHeaderPS(const char* header)
  # tdrStyle.SetTitlePS(const char* pstitle)

  # tdrStyle.SetBarOffset(Float_t baroff = 0.5)
  # tdrStyle.SetBarWidth(Float_t barwidth = 0.5)
  # tdrStyle.SetPaintTextFormat(const char* format = "g")
  # tdrStyle.SetPalette(Int_t ncolors = 0, Int_t* colors = 0)
  # tdrStyle.SetTimeOffset(Double_t toffset)
  # tdrStyle.SetHistMinimumZero(kTRUE)

  tdrStyle.SetHatchesLineWidth(5)
  tdrStyle.SetHatchesSpacing(0.05)

  tdrStyle.cd()
