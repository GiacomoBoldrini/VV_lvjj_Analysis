import argparse
import os

'''
This script is used to evaluate a NN model at different steps (or epoch)
creating different branches and using only one template model configuration.
'''

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True,
                    help="Root file")
parser.add_argument('-t', '--tree', type=str, required=False,
                    default="mw_mjj", help="Tree name")
parser.add_argument('-m', '--model-config', type=str, required=True,
                    help="Model configuration file")  
parser.add_argument('-b', '--branchname', type=str, required=False, 
                    default="score", help="Output branch name")
parser.add_argument('-n', '--n-epochs', nargs='+', type=str, required=True, 
                        help="List of epochs")
parser.add_argument('-bs', '--batch-size', type=int, required=False, default=4096,
                    help="Batch size") 
conf =parser.parse_args()

for n in conf.n_epochs:
    with open(conf.model_config) as fin:
        template = fin.read()
        template = template.replace("{EPOCH}", n)
    with open("model.txt", "w") as fout:
        fout.write(template)

    command = "python evaluateNNModel.py -f {0} -t {1} -m {2} -b {3} -bs {4}".format(
        conf.file, conf.tree, "model.txt", conf.branchname +"_epoch{}".format(n), conf.batch_size
    ) 
    print(command)
    os.system(command) 
