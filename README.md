# How to run the analysis.

## Event production

First of all you have to produce the samples... 
Look to the **EventProduction** and **ProductionCard** repositories for instructions 
and configurations for Phantom/Madgraph+Pythia chain. 

The output of the production chain is a ROOT Tree containing all the 
4-vectors describing each event, including some information at LHE level (partons 4-vectors).

## Parton association
First of all you have to associate the partons to the jets in the event to
save useful information. 

```
python associate_jets_lhe.py -i input_file.root  -r jet_radius
```

## Variables extraction
This is the central part of the analysis. Extract variables from the raw events and 
save them in a tree with the appropriate cuts and selection strategies. 
At this stage you can also extract btagging efficiency. 

All this process is handled by the **extractVariables.py** script.  The **VBSAnalysis/OutputTree** module
contains the definition of the extracted variables, whereas the **VBSAnalysis/JetTagging** module
contains the strategies used to select the VBS jets and the Vjets. 

```
python extractVariables     -i      input_file_raw_events.root                        
                            -o      ouput_file.root                                   
                            -ot     output_tree_name           
                            -xs     sample_cross_section (pb)                         
                          [ -it     input_tree_name (default=tree)                ]
                          [ -bt     save bveto weights for L,M,T working points   ]
                          [ -n      nevents_to_extract                            ]
```

## Plots production


## DNN Training

### Prepare datasets
First of all you have to prepare numpy arrays from data trees using the script **DeepLearning/scripts/prepareData.py**.
You have to specify the branches to be extracted and the cuts inside the script. Only events that pass all the cuts
are saved in the numpy array in order to have a dataset ready for training. 

```
python prepareData.py    -i  input_root_file.root 
                         -o  output_numpy_array.npy 
                        [-n  number of events      ]
```
N.B.  Extract different samples for validation and for training. 

### Binary classification
Now you are ready to train a model: let's start with binary classification of one background versus signal. 
The script **DeepLearning/scripts/trainDNN.py** has a lot of option to train a DNN model on the previously prepared datasets. 
The *Keras* models are defined inside **DeepLearning/scripts/models.py**. 
The signal and background samples are automatically balanced to have 50% fraction of the events. 

```
python trainDNN.py    -s    signal_dataset_path
                      -sv   signal validation dataset_path
                      -b    background_dataset_path
                      -bv   background validation dataset_path
                      -o    output_dir
                      -ms   model_schema_id  (one of the models in models.py module)
                    [ -e    number of epochs  (default=10)                                      ]
                    [ -i    initial_epoch   useful to continue a previous training              ]
                    [ -bs   batch_size        (default=50)                                      ] 
                    [ -lr   learning_rate     (default=1e-3)                                    ]
                    [ -dr   learning_rate decay factor   (default=0)                            ]
                    [ -vs   validation number of samples (default=1e5)                          ]
                    [ -p    patience for early stopping,  delta_val:epochs  (default="0.001:5") ]
                    [ -m    model_file to continue a training                                   ]
                    [ -e    True/False only evaluate results                                    ]
                    [ -sst  True/False save results at all steps for incremental studies        ]
```
The output of the training is saved in the specified output directory. A config file is saved with all the parameters
used. The model is saved as a *Keras* output file, and also the roc_curve is saved in a file. 

The data is automatically standardize with mean 0 and standard deviation 1 before the training. The Scaler object is saved in a file to be used later for the application of the discriminator. 


## DNN Evaluation

### ROC curves
First of all we want to extract ROC curves of the different trained NN on indipendent samples of events. 
The datasets are prepared like the training data with **DeepLearning/scripts/prepareData.py** script, as described before. 

Once a model have been trained we have to define a model configuration file which incapsulate informations like the model file path, 
the scaler to be used, the variables and the cuts used for the training.  The model conf file is a yaml file with this structure:

```yaml
variables: ["deltaeta_mu_vbs_high",  "deltaphi_mu_vbs_high", "deltaeta_mu_vjet_high", "deltaphi_mu_vjet_high", 
              "deltaR_mu_vbs","deltaR_mu_vjet", "L_p", "L_pw", "Mww" ,"w_lep_pt", "C_ww", "C_vbs", "A_ww",
              "Zmu", "met", "mu_eta",  "mu_pt", "mjj_vjet", "deltaR_vjet", "vjet_pt_high", "A_vjet", 
              "Rvjets_high", "Zvjets_high", "mjj_vbs", "deltaeta_vbs", "deltaR_vbs", "A_vbs", "vbs_pt_high",
              "N_jets_forward", "N_jets_central", "N_jets", "Ht", "R_mw", "R_ww" ]
              
model_path: DeepLearning/models/run5/w+jets/v2/current_model

scaler: DeepLearning/models/run5/w+jets/v2/scaler.pkl

cuts:  >-
         (mu_eta< 2.1) &&  (mjj_vjet > 65) && (mjj_vjet < 105) &&
         (mjj_vbs>200) && (deltaeta_vbs > 2.5) && (met > 20) && (mu_pt > 20)

```
Once you have model conf file for each of your model you can extract ROC curves using the **DeepLearning/script/ROC_curve.py** script. 

```
python ROC_curve.py     -s    signal_dataset
                        -b    background_dataset
                        -m    model configuration file
                        -o    output directory
                        -f    output filename
                      [ -ss   Size of the score (default=1), use it in case of models with multiple outputs   ]
```
Now, to plot the ROC_curves use the script **DeepLearning/script/plotROCs.py**.
The expected structure of roc curve data files inside
the input directory is:  `$nn/roc_model_$nn_ewk_$bkg_sample` with `$nn` representing the model under test and `$bkg_sample` the background sample evaluated against the signal.  

```
python plotROCs.py    -i input_directory -o output_directory
```
A plot is produced for every `$bkg_sample` to compare the performance of each of the model. 


### Save the scores

The scores of the neural network classifiers must be saved inside the event TTree. The script **evaluateNNModel.py** applies the Keras
model to each entry of the TTree, filling a new branch with the output score of the neural network. 

```
python evaluateNNModel.py       -f    input Root file with events
                                -t    Events tree name
                                -m    model configuration file
                                -b    branch name for the output
                              [ -bs   bunch size for the evaluation of the model (default=4096)                       ]
                              [ -nt   not overwrite but create a new cloned TTree                                     ]
                              [ -ss   Size of the score (default=1), use it in case of models with multiple outputs   ]
```


## JES uncertainties
JES uncertainties are evalutated varying up and down the energy of every jet at the 
beginning of the analysis (on raw events) before the extraction of the variables. 
The uncertainty factors are provided by CMS in pt and eta bins. 
The code that reads and elaborated JES is already provided by CMSSW, so you can use the 
macro **VBSAnalysis/JES/JES.cc** inside a CMSSW release to elaborate raw events and get a TTree with
scaled jets. 

The macro requires as input the ROOT file containing the TTree with raw events and it create a separate
file with the new events. It asks also a txt file with the CMS JES parameters and the JES correction 
direction (up or down) as last argument.

To run it:

```
root -b -x -q "JES.cc('input_file.root', 'ouputfile.root', 'JES_parameters_file.txt', '+1/-1')
```

The output file is a standard raw event TTree, ready for all the steps of the analysis. 


# Significance extraction

## Cut & Count analysis

First of all we have to extract the yields of the event following a grid of cuts on the scores of out NN discriminators. 
To do that, use the **Yields/extractYields.py** script.

```
python extractYields.py    -c  configuration_card.yaml
```

The configuration card contains all the necessary information. 

```yaml
cut: >-
    (abs(mu_eta) < 2.1) && (mjj_vjet > 65) && (mjj_vjet < 105) &&
    (mjj_vbs>200) && (deltaeta_vbs > 2.5) && (met > 20) && (mu_pt > 20)

weights: xs_weight*bveto_weights[1]

lumi: 100000

# Which branch has to be used as a score?
scores:
    score_t0 : score_t0_v5
    score_t1 : score_t1
    score_t2 : score_t2
    score_wjets : score_wjets_v5

trees: 
    ewk: 
        mw_mjj : ../data/output_swan/ewk_btag_val.root
        jes_up : ../data/output_swan/ewk_btag_jes.root
        jes_down: ../data/output_swan/ewk_btag_jes.root
    qcd_t0: 
        ...(the same as ewk)
    qcd_t1: 
        ...
    qcd_t2: 
       ...
    wjets: 
       ...
    
output_file: yields_grid.root

# Grid search on the scores
ranges:  
    t0: "0.65:0.79:0.05"
    t1: "0.7:0.71:0.1"
    t2: "0.8:0.81:0.1"
    wjets: "0.980:0.987:0.002"
```
The script will create a TTree containing for each *(score_t0, score_t1, score_t2, score_wjets)* the number of events extract for every sample, along with the number of events for jes up/down data.  Also the MC statistical error is saved in a branch.  Moreover, the script saves also **s/sqrt(b)** (without systematics) to do quick explorations. 