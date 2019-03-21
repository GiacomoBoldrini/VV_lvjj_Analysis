#!/bin/python
import sys
sys.argv.append( '-b' ) # batch mode for root
import ROOT as ROOT 
import plotter 
ROOT.TH1.SetDefaultSumw2()
import argparse
import yaml
import os
import shutil
from collections import defaultdict
from card_loader import load_card , unload_card

plotter.setStyle()

def getArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--card', type=str, required=True, help="Plots card") 
    parser.add_argument('-t', '--tag', type=str, required=True,  help="Plots tag")  
    parser.add_argument('-v', '--variables', nargs='+', type=str, required=False, 
                        help="List of variables to plot")
    parser.add_argument('-o', '--output-type', type=str, required=False, default="png",
                      help="Output type (png|pdf|svg)")    
    parser.add_argument('-r', '--ratio', nargs='+', type=str, required=False, 
                      help="Draw ratio of histos") 
    parser.add_argument('-rl', '--ratio-label', type=str, required=False, default="ratio",
                      help="Ratio plot label" )
    parser.add_argument('-nl', '--nolog', action="store_true")
    parser.add_argument('-ne', '--noerr', action="store_true", help="Do not print error bars")
    parser.add_argument('-b', '--batch', action="store_true" ) 
    return parser


def do_plot(plot_info, variables, trees):
    print("###############################################")
    print("Plots: ", plot_info["name"], " to ", plot_info["output_dir"])
    print("###############################################")

    # Check if we have to save the histos in one file
    if plot_info["output_root_file"] != "None":
        output_rootfile = ROOT.TFile(plot_info["output_root_file"], "RECREATE")
    else:
        output_rootfile = None

    # The cut string is the same for all samples and variables. 
    cuts = plot_info["cuts"]

    # Loop on every variable
    for var_name, var, output_path in variables:        
        print("@---> Plotting " +var["name"])

        # Setting default values
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

        draw_ratio =  plot_info["ratio"] != None 

        if draw_ratio:
            c1, pad1, pad2 = plotter.createCanvasPads()
        else:
            c1 = plotter.getCanvas()
        legend = plotter.getLegends(var["legpos"], 1,2, xmin=0.75)

        # Calculate xs correction factor for specific variable
        nvars = len(var["var_list"]) if var["is_sum"] else 1
        xs_correction =  var["weight"] / nvars

        # Create the histograms
        maximums = []
        minima = []

        for sample_name, trs in trees.items():
            nselect = 0
            hist_id = "hist_{}_{}".format(var["name"], sample_name)
            h = ROOT.TH1D(hist_id, var["title"],var["nbin"], var["xmin"], var["xmax"])

            #cycling of every tree of this sample
            for tree, fname, weight in trs:
                print(">> Plotting {} | Tree weight: {}".format(fname, weight))
                # Final cutstring  with cuts*weight*xs_correction
                cutstring = "({})*({})*({})".format(cuts, weight, xs_correction)
                #print(cutstring)

                if not var["is_sum"]:
                    nselect += tree.Draw("{}>>+{}".format(var["var"], hist_id), cutstring, "goff")
                else:
                    for v in var["var_list"]:
                        nselect += tree.Draw("{}>>+{}".format(v, hist_id), cutstring,  "goff")

            # Fix overflow bin
            plotter.fixOverflowBins(h)
            # Print XS 
            print("XS: {} | {} | {:.3f} pb | N events {}".format(var["name"],
                    sample_name, h.Integral(), nselect))
            if plot_info["normalize"]:
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
            
        if draw_ratio: pad1.cd()

        #### MAIN PAD         
        # Set log scale
        if (not plot_info["nolog"]) and var["log"]:
            ROOT.gPad.SetLogy() 
            maximum = max(maximums)*10
        else:
            maximum = max(maximums)*1.1

        # Canvas background
        background_frame = ROOT.gPad.DrawFrame(var["xmin"], 0.5*min(minima),
                                        var["xmax"] , maximum)
        background_frame.SetTitle(var["title"])
        background_frame.GetXaxis().SetTitle(var["xlabel"])
        background_frame.GetYaxis().SetTitle(plot_info["ylabel"])
        #Fix the style of the axis
        plotter.setHistoStyle(background_frame, draw_ratio)
        background_frame.Draw()

        #Draw the histograms
        for hist_name,  hist in hists.items():
            if "EWK" in hist_name:
                hist.SetLineWidth(4)
            else:
                hist.SetLineWidth(3)
            hist.Draw("hist same PLC")
            if not plot_info["noerr"]:
                herr = hist.DrawCopy("E2 same")
                herr.SetFillStyle(3013)
                herr.SetFillColor(13)
        
        # Draw the legend
        legend.Draw("same")
        # Adjust canvas
        ROOT.gPad.RedrawAxis()

        ## RATIO PAD
        if draw_ratio:
            pad2.cd()
            h1 = hists[plot_info["ratio"][0]]
            h2 = hists[plot_info["ratio"][1]]
            hratio = plotter.createRatio(h1, h2, label=plot_info["ratio_label"])
            plotter.setHistoStyle(hratio, draw_ratio)
            hratio.GetXaxis().SetTitle(var["xlabel"])
            hratio.Draw("hist same PLC")
            herr_ratio = hratio.DrawCopy("E2 same")
            herr_ratio.SetFillStyle(3013)
            herr_ratio.SetFillColor(13)
            herr_ratio.SetMarkerStyle(1)
            # Adjust canvas
            pad2.RedrawAxis()
            
        # Print canvas on file
        c1.SaveAs(output_path+ "/" + plot_info["name"] + "_" + var["name"]+"."+ plot_info["output_type"])
        
        #Save the histos on file if required
        if output_rootfile != None:
            for hist in hists.values():
                hist.Write()
        
    # Save the output Root file if required
    if output_rootfile != None:
        output_rootfile.Close()


if __name__ == "__main__":
    args = getArgParser().parse_args()
    options = {"nolog":args.nolog, "output_type": args.output_type,
              "ratio": args.ratio, "ratio_label": args.ratio_label,
              "noerr": args.noerr}

    for card in load_card(args.card, args.tag, options):
        vars_toplot = card.vars
        if args.variables != None:
            #filter vars
            vars_toplot = filter(lambda v: v[0] in args.variables, card.vars)
        do_plot(card.plot_info, vars_toplot, card.trees)
        # Save the card in the main folder
        shutil.copyfile(args.card, card.plot_info["output_dir"]+"/plots.card.yaml")
        unload_card(card)

    
   
