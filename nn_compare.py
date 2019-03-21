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
    parser.add_argument('-c', '--card', type=str, required=True, help="Plots card") 
    parser.add_argument('-t', '--tag', type=str, required=True,  help="Plots tag")  
    parser.add_argument('-sb', '--score-branch', type=str, required=False, help="Score variable")
    parser.add_argument('-sv', '--score-value', type=float, required=False, help="Score value")
    parser.add_argument('-v', '--variables', nargs='+', type=str, required=False, 
                        help="List of variables to plot")
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

        draw_ratio =  plot_info["ratio"]
        # plotter.setTDRStyle()
        if(draw_ratio):
            c1, pad1,pad2 = plotter.createCanvasPads()
        else:
            c1 = plotter.getCanvas()
        legend = plotter.getLegends(var["legpos"], 1,2)
        legend.SetTextSize(0.03)

        # Calculate xs correction factor
        nvars = len(var["var_list"]) if var["is_sum"] else 1
        xs_factor =  var["weight"] * 1 / nvars
        # Create the histograms
        maximums = []
        minima = []

        events_total = {}

        for sample_name, trs in trees.items():
            nselect = 0
            hist_id = "hist_{}_{}".format(var["name"], sample_name)
            h = ROOT.TH1D(hist_id, var["title"],var["nbin"], var["xmin"], var["xmax"])

            #cycling of every tree of this sample
            for tree, xs_label, cuts_label, fname in trs:
                print(">> Plotting {} | Weight: {:.6E}".format(fname, xs_weight[xs_label]*xs_factor))
                cutstring = "("+cuts[cuts_label]+")*" + str(xs_weight[xs_label]*xs_factor)
                if not var["is_sum"]:
                    nselect += tree.Draw("{}>>+{}".format(var["var"], hist_id), cutstring, "goff")
                else:
                    for v in var["var_list"]:
                        nselect += tree.Draw("{}>>+{}".format(v, hist_id), cutstring,  "goff")

            # Fix overflow bin
            plotter.fixOverflowBins(h)
            # Print XS 
            print("XS: {} | {} | N events {}".format(var["name"], sample_name, nselect))
            #Save the selected events
            events_total[sample_name] = nselect
            #Rescale to 1
            h.Scale(1/h.Integral())
            legend.AddEntry(h, sample_name, "L")
            #Save the histo    
            hists[sample_name] = h
            maximums.append(h.GetMaximum())
            minimum = h.GetMinimum()
            if minimum == 0:  minimum=0.0001
            minima.append(minimum)
            minima.append(minimum)
    
        # Now draw the cuts
        for sample_name, trs in trees.items():
            score_cut = plot_info["nn_cut"]
            nselect = 0
            hist_id = "hist_{}_{}_score".format(var["name"], sample_name)
            h = ROOT.TH1D(hist_id, var["title"],var["nbin"], var["xmin"], var["xmax"])

            #cycling of every tree of this sample
            for tree, xs_label, cuts_label, fname in trs:
                print(">> Plotting {} | Weight: {:.6E}".format(fname, xs[xs_label]["xsnorm"]*xs_factor))
                cutstring = "("+cuts[cuts_label]+"&& " + score_cut +")*" + str(xs[xs_label]["xsnorm"]*xs_factor)
                if not var["is_sum"]:
                    nselect += tree.Draw("{}>>+{}".format(var["var"], hist_id), cutstring, "goff")
                else:
                    for v in var["var_list"]:
                        nselect += tree.Draw("{}>>+{}".format(v, hist_id), cutstring,  "goff")

            # Fix overflow bin
            plotter.fixOverflowBins(h)
            # Calculate efficiency
            eff = nselect / events_total[sample_name]
            print("{} | Eff: {:.2f} | Sample: {} | N events {} | ".format(var["name"],eff, sample_name, nselect))
            #Rescale to 1
            h.Scale(eff/h.Integral())
            legend.AddEntry(h, sample_name +" cut", "L")
            # Save score histo
            hists[sample_name+"_score"] = h
            maximums.append(h.GetMaximum())
            minimum = h.GetMinimum()
            if minimum == 0:  minimum=0.0001
            minima.append(minimum)
            minima.append(minimum)
    
        # Set log scale
        if (not plot_info["nolog"]) and var["log"]:
            c1.SetLogy() 
            maximum = max(maximums)*10
        else:
            maximum = max(maximums)*1.1

         # Canvas background
        background_frame = c1.DrawFrame(var["xmin"], 0.5*min(minima),
                                        var["xmax"] , maximum)
        background_frame.SetTitle(var["title"])
        background_frame.GetXaxis().SetTitle(var["xlabel"])
        background_frame.GetYaxis().SetTitle( "shape")
        background_frame.Draw()

        if draw_ratio : pad1.cd()
        # Draw the histograms
        for hist_name, hist in hists.items():
            if "EWK" in hist_name:
                hist.SetLineWidth(3)
            else:
                hist.SetLineWidth(2)
            if "score" in hist_name:
                hist.SetFillStyle(3003)
            else:
                hist.SetFillStyle(0)
        
            #Draw the histo
            hist.Draw("hist same PLC PFC")
            herr = hist.DrawCopy("E2 same")
            herr.SetFillStyle(3013)
            herr.SetFillColor(13)
        
        # Draw the legend
        legend.Draw("same")

        # Adjust canvas
        c1.RedrawAxis()
        
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
    options = {"nolog": args.nolog, "ratio": False}
    # We can overwrite the defaults in the card
    if args.score_branch != None:
        options["score_branch"] = args.score_branch
    if args.score_value != None: 
        options["score_value"] = args.score_value

    for card in load_card.load_card(args.card, args.tag, options):
        vars_toplot = card.vars
        if args.variables != None:
            #filter vars
            vars_toplot = filter(lambda v: v[0] in args.variables, card.vars)
        do_plot(card.plot_info, card.xs, card.cuts, card.vars, card.trees, card.xs_weight)

        # Save the card in the main folder
        shutil.copyfile(args.card, card.plot_info["output_dir"]+"/plots.card.yaml")


    
