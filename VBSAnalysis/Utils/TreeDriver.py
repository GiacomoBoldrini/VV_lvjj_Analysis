from array import array
from . import Converter
from collections import namedtuple

class BaseTreeDriver():
    ''' Structure for memory efficient TTree reading'''
    def __init__(self, tree):
        self.evId = 0
        self.tree = tree
        self.n_jets = array("i", [0])
        #jets
        self.E_jets = array("d", [0.]*25)
        self.px_jets = array("d", [0.]*25)
        self.py_jets = array("d", [0.]*25)
        self.pz_jets = array("d", [0.]*25)
        self.tree.SetBranchAddress("njets", self.n_jets)
        self.tree.SetBranchAddress("E_jets", self.E_jets)
        self.tree.SetBranchAddress("px_jets", self.px_jets)
        self.tree.SetBranchAddress("py_jets", self.py_jets)
        self.tree.SetBranchAddress("pz_jets", self.pz_jets)
        #muon and met
        self.p_mu = array("d", [0.]*4)
        self.p_nu = array("d", [0.]*4)
        self.p_met = array("d", [0.]*4)
        self.p_met_uncl = array("d", [0.]*4)
        self.tree.SetBranchAddress("p_mu", self.p_mu)
        self.tree.SetBranchAddress("p_nu", self.p_nu)
        self.tree.SetBranchAddress("met", self.p_met)
        self.tree.SetBranchAddress("unclustered_met", self.p_met_uncl)
        # Populate data from first entry
        self.tree.GetEntry(0)
        # Cache for jets four-vectors
        self.jets_vectors = []
        self.all_jets = []
        self.jets_loaded = False
    
    @property
    def entries(self):
        return self.tree.GetEntries()

    @property
    def njets(self):
        return self.n_jets[0]
        
    @property
    def jets(self):
        if not self.jets_loaded:
            # Caching the vector of jets
            self.jets_vectors = Converter.convertToLorentzVector(
                self.njets, self.px_jets, self.py_jets, 
                self.pz_jets, self.E_jets)
            # save all jets as not cut
            self.all_jets = self.jets_vectors
            self.jets_loaded = True
        return self.jets_vectors

    @jets.setter
    def jets(self, js):
        self.jets_vectors = js
        self.n_jets[0] = len(js)

    @property
    def muon(self): 
        return Converter.convertToLorentzVector_single(
            self.p_mu[0], self.p_mu[1],
            self.p_mu[2], self.p_mu[3] )

    @property
    def neutrino(self): 
        return Converter.convertToLorentzVector_single(
            self.p_nu[0], self.p_nu[1],
            self.p_nu[2], self.p_nu[3] )

    @property
    def P_met(self):
        return Converter.convertToLorentzVector_single(
            self.p_met[0], self.p_met[1],
            self.p_met[2], self.p_met[3] )
    
    @property
    def P_uncl_met(self):
        return Converter.convertToLorentzVector_single(
            self.p_met_uncl[0], self.p_met_uncl[1],
            self.p_met_uncl[2], self.p_met_uncl[3] )
    
    def get_jets(self, li):
        ''' Get a sublist of jets'''
        return [self.jets[l] for l in li ]

    # Iterator protocol
    #-----------------------------------------------------------
    # The getEntry method will call the subclass one
    # and every subclass getEntry will call the super one. 
    # At the end the getEntry of this baseclass will move the tree. 
    def all(self):
        ''' Generator to iterator on all the entries'''
        for i in range(self.tree.GetEntries()):
            self.getEntry(i)
            yield self

    def getEntry(self, i):
        # Mark jets to be cleared
        self.jets_loaded = False
        self.evId = i  
        self.tree.GetEntry(i)

##################################################
# Tree Driver also with partons information

class PartonsTreeDriver(BaseTreeDriver):

    def __init__(self, tree):
        BaseTreeDriver.__init__(self, tree)
        self.n_partons = array("i", [0])
        self.E_parton = array("d", [0.]*4)
        self.px_parton = array("d", [0.]*4)
        self.py_parton = array("d", [0.]*4)
        self.pz_parton = array("d", [0.]*4)
        self.partons_flavour = array("i", [0]*4)
        self.tree.SetBranchAddress("npartons", self.n_partons)
        self.tree.SetBranchAddress("E_parton", self.E_parton)
        self.tree.SetBranchAddress("px_parton", self.px_parton)
        self.tree.SetBranchAddress("py_parton", self.py_parton)
        self.tree.SetBranchAddress("pz_parton", self.pz_parton)
        self.tree.SetBranchAddress("partons_flavour", self.partons_flavour)
        self.p_mu_lhe = array("d", [0.]*4)
        self.tree.SetBranchAddress("p_mu_lhe", self.p_mu_lhe)
        self.tree.GetEntry(0)
        self.partons_vectors = []
        self.partons_loaded = False

    @property
    def npartons(self):
        return self.n_partons[0]

    @property
    def partons(self):
        if not self.partons_loaded:
            # Caching the vector of jets
            self.partons_vectors = Converter.convertToLorentzVector(
                4, self.px_parton, self.py_parton, 
                self.pz_parton, self.E_parton )
            self.partons_loaded = True
        return self.partons_vectors

    @property
    def muon_lhe(self):
        return Converter.convertToLorentzVector_single(
            self.p_mu_lhe[0], self.p_mu_lhe[1],
            self.p_mu_lhe[2], self.p_mu_lhe[3] )

    
    def get_partons(self, li):
        ''' Get a sublist of partons'''
        return [self.partons[l] for l in li]
    

    def getEntry(self,i):
        # Only clear data, the tree is iterated by the base class
        self.partons_loaded = False
        # Call the super() get entry as last operatioin
        super().getEntry(i)


#######################################################################
# Tree Driver for pairing information

# Basic object for Jet/Parton pair
JetPair = namedtuple("JetPair", ["parton", "jet", "flavour"])

class PairingTreeDriver(PartonsTreeDriver):

    def __init__(self, tree, tree_pairs):
        PartonsTreeDriver.__init__(self, tree)
        self.tree_pairs = tree_pairs
        #Parton-Jet loaded pairs
        self.pjpairs = []
        self.pairs_loaded = False

    def __getattr__(self, name):
        ''' The attributes not present in PartonsTreeDriver
        are proxied to tree_pairs object'''
        return getattr(self.tree_pairs, name)

    def getEntry(self,i):
        # Only clear data, the base tree is iterated by the base class
        self.pjpairs.clear()
        self.pairs_loaded = False
        # Get Entry on tree_pairs tree
        self.tree_pairs.GetEntry(i)
        super().getEntry(i)

    @property
    def paired_jets(self):
        ''' This function returns all the pairs partons-jets
        without checking if the jet has been cut in the event'''
        #Caching a vector of JetPairs 
        if not self.pairs_loaded:
            for i, p in enumerate(self.tree_pairs.pairs):
                # Uncut jets are used to preserve indexes meaning
                self.pjpairs.append(JetPair(self.partons[i],
                            self.all_jets[p], abs(self.partons_flavour[i])))
            self.pairs_loaded = True
        return self.pjpairs

    @property
    def paired_jets_not_cut(self):
        ''' This function returns only the pairs where the corresponding
        jet is not cut in the event
        '''
        return [j for j in self.paired_jets if j.jet in self.jets]

    def paired_jet(self, parton_index):
        ''' This function returns the jet corresponding to 
        requested parton ( using parton index position in the list)'''
        return self.pjpairs[parton_index].jet

    def paired_parton(self, jet):
        ''' This function returns the parton associated with the requested
        jet, if it exists. 
        '''
        for pair in self.paired_jets:
            # Let's use the IsEqual function to check 
            # the equality of TLorentzVectors
            if pair.jet.IsEqual(jet):
                return pair.parton
        # if not found
        return None

    @property
    def pass_jets_cuts(self):
        ''' The function checks if the paired jets 
        are not cut in the event.  It can be used to filter out events
        where the paired jets don't pass the cuts'''
        # in operator it works because the object is the same
        for p in self.paired_jets:
            if p.jet not in self.jets:
                return False
        return True
        

    
