#!/bin/python
import sys
sys.argv.append( '-b' ) # batch mode for root
import ROOT as ROOT 
ROOT.TH1.SetDefaultSumw2()
import argparse
import plotter 
import yaml
import os
import shutil
import random
from collections import defaultdict
import load_card


def setStyle(): 
    style = ROOT.gStyle
    style.SetPalette(ROOT.kDarkRainBow)
    style.SetOptStat(0)
    style.SetTitleFont(43)
    style.SetTitleSize(35)
    style.SetTitleFont(43, "XYZ")
    style.SetTitleSize(30, "XYZ")
    #style.SetTitleOffset(1, "Y")
    #style.SetTitleOffset(1, "X")
    style.SetLabelFont(43, "XY")
    style.SetLabelSize(27, "XY")
    style.SetPadBottomMargin(0.4)
    style.SetPadLeftMargin(0.125)
    style.SetPadRightMargin(0.03)
    style.cd()

def getArgParser():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--card', type=str, required=True, help="Plots card") 
    parser.add_argument('-t', '--tag', type=str, required=True,  help="Plots tag")  
    parser.add_argument('-nl', '--nolog', action="store_true")
    parser.add_argument('-b', '--batch', action="store_true" ) 
    return parser


def do_plot(plot_info, xs, cuts, variables, trees, xs_weight):
    #Set root style
    setStyle()

    print("###############################################")
    print("Plots: ", plot_info["name"])
    print("###############################################")

    #print(trees)
    #print("Cuts: ", cuts)

    # Check if we have to save the histos in one file
    if plot_info["output_root_file"] != "None":
        output_rootfile = ROOT.TFile(plot_info["output_root_file"], "RECREATE")
    else:
        output_rootfile = None

    # Loop on every variable
    for var_name, var, output_path in variables: 
        print("@---> Plotting " +var["name"])
        
        # Check defaults
        if "is_sum" not in var:
            var["is_sum"] = False
        if "weight" not in var:
            var["weight"] = 1
        if "log" not in var:
            var["log"] = False
        if "legpos" not in var:
            var["legpos"] = "tr"
        if "var" not in var:
            var["var"] = var["name"]

        hists = {}
        
        c1 = plotter.getCanvas()        
        # Stack for histos
        stack = ROOT.THStack(f"stack_{var['name']}", "")
        #legend
        legend = plotter.getLegends(var["legpos"], 1,3)
        # Calculate xs correction factor
        nvars = len(var["var_list"]) if var["is_sum"] else 1
        xs_factor =  var["weight"] * 1 / nvars
        # Create the histograms
        minima = []
        maxima = []
        for sample_name, trs in trees.items():
            nselect = 0
            hist_id = "hist_{}_{}".format(var["name"], sample_name)
            h = ROOT.TH1D(hist_id, var["title"],var["nbin"], var["xmin"], var["xmax"])
            
            #cycling of every tree of this sample
            for tree, xs_label, cuts_label, fname in trs:
                print(">> Plotting {} | Weight: {:.6E}".format(fname, xs_weight[xs_label]*xs_factor))
                # Custring is made by (cuts)*xsweight*xs_factor
                cutstring = "("+cuts[cuts_label]+")*" + str(xs_weight[xs_label]*xs_factor)
                if not var["is_sum"]:
                    nselect += tree.Draw("{}>>+{}".format(var["var"], hist_id), cutstring, "goff")
                else:
                    for v in var["var_list"]:
                        nselect += tree.Draw("{}>>+{}".format(v, hist_id), cutstring,  "goff")
            # Fix overflow bin
            plotter.fixOverflowBins(h)
            minimum = h.GetMinimum()
            if minimum == 0:  minimum=0.000001
            minima.append(minimum)
            maxima.append(h.GetMaximum())
            hists[sample_name] = h
            # Print XS 
            print("XS: {} | {} | {:.3f} pb | N events {}".format(var["name"],
                    sample_name, h.Integral(), nselect))

        #Set histo drawing style and add to stack in order of samples_name
        for sample in plot_info["sample_names"]:
            hist = hists[sample]
            #hist.SetMaximum(max(maximums) * 1.1)
            hist.SetLineWidth(2)
            hist.SetFillStyle(random.choice([3003, 3004, 3005]))
            legend.AddEntry(hist, sample, "F") 
            # Add to stack
            stack.Add(hist, "hist")
    
            #herr = hist.DrawCopy("E2 same")
            #herr.SetFillStyle(3013)
            #herr.SetFillColor(13)
            
        # Set log scale
        if (not plot_info["nolog"]) and var["log"]:
            c1.SetLogy()   

        # Draw on the canvas
        background_frame = c1.DrawFrame(var["xmin"], 0.4*min(minima),
                                        var["xmax"] , 50*max(maxima))
        background_frame.SetTitle(var["title"])
        background_frame.GetXaxis().SetTitle(var["xlabel"])
        background_frame.GetYaxis().SetTitle( "XS (pb)")
        background_frame.Draw()
            
        stack.Draw("same PLC PFC")
        c1.RedrawAxis()

        legend.Draw("same")
        
        
        # Print canvas on file
        c1.SaveAs(output_path+ "/" + plot_info["name"] + "_" + var["name"]+".png")

        #Save the histos on file if required
        if output_rootfile != None:
            for hist in hists.values():
                hist.Write()
        
    # Save the output Root file if required
    if output_rootfile != None:
        output_rootfile.Close()


if __name__ == "__main__":
    args = getArgParser().parse_args()
    options = {"nolog":args.nolog}

    for card in load_card.load_card(args.card, args.tag, options):
        do_plot(card.plot_info, card.xs, card.cuts, card.vars, card.trees, card.xs_weight)

        # Save the card in the main folder
        shutil.copyfile(args.card, card.plot_info["output_dir"]+"/plots.card.yaml")


    
   

   
