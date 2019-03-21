import yaml
import os
from collections import namedtuple, defaultdict
import ROOT as R

Card = namedtuple("Card", ["plot_info", "vars", "trees", "files"])

def load_card(card, tag, options={}):
    ''' This function reads the plots card and create conf objects
    for the drawing scripts'''
    # Load the card
    card = yaml.load(open(card))
    cuts = card["cuts"]
    variables_groups = card.get("variables_groups", None)

    results = []
    for plot_info in card["plots"]:
        if plot_info["tag"] != tag:
            continue
        # Create the output dir
        if not os.path.exists(plot_info["output_dir"]):
            os.makedirs(plot_info["output_dir"])
        # Add options to plot info
        plot_info.update(options)
        # Elaborate variables
        var_file = yaml.load(open(plot_info["variables_file"]))
        vars_dict = {}
        for v in var_file:
            vars_dict[v["name"]] = v
        #Elaborating the output
        vars_output = []
        # Now look if there are groups of variables
        if variables_groups not in [None, "None"]:
            # Create dir for the group
            for group, vs in variables_groups.items():
                group_path = plot_info["output_dir"]+"/"+ group +"/"
                if not os.path.exists(group_path):
                    os.makedirs(group_path)
                for v in vs:
                    vars_output.append((v, vars_dict[v], plot_info["output_dir"]+"/"+ group +"/"))
        else:
            for v in var_file:
                vars_output.append((v["name"], v, plot_info["output_dir"]+"/"))
        # Now filter out the variables if necessary
        if "variables_only" in plot_info and plot_info["variables_only"] != None:
            vars_output = list(filter(lambda k: k[0] in plot_info["variables_only"], vars_output))

        # Extract cuts string from the settings:
        # The cut category is global for the tag. 
        plot_info["cuts"] = cuts[plot_info["cuts"]]
        base_weight = plot_info["base_weight"]

        # Load trees
        trees = defaultdict(list)
        files = []
        # Every file item contains filename;sample_name;treename;weight(optional)
        for f in plot_info["trees"]:
            file = R.TFile(f["file"],"READ")
            files.append(file)
            tree = file.Get(f["treename"])
            # A tuple (tree, filename) is saved for every 
            # tree in the samples dictionary
            filename = f["file"].split("/")[-1]
            # Add the specific weight if present
            tree_weight = base_weight
            if "weight" in f:
                tree_weight += "*({})".format(f["weight"])
            # Every tree has the weight to be used for every event
            trees[f["sample"]].append((tree, filename, tree_weight))
        #save the card
        results.append(Card(plot_info, vars_output, trees, files))

    return results


def unload_card(card):
    for f in card.files:
        #print("Closing: ", f)
        f.Close()
