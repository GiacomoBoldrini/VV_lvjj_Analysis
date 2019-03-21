import array 
import numpy as np
import ROOT as R 


class TreeWriter:

    def __init__(self, treename):
        self._tree = R.TTree(treename, treename)
        self._branches = {}
        self._dim = {}

    def define_branches(self, branches):
        for dim, _ in branches.items(): 
            if dim>1:
                rootdim = "[{}]".format(dim)
            else: 
                rootdim = ""
                
            for typ, names in _.items():
                if typ == float:
                    rootype = "D"
                elif typ == int:
                    rootype = "I"

                for br in names:
                    data = np.zeros(dim, dtype=typ)
                    self._branches[br] = data
                    self._dim[br] = dim
                    br_root = "{0}{1}/{2}".format(br, rootdim, rootype)
                    #print("Adding branch: ", br_root)
                    self._tree.Branch(br, data, br_root)
        # Add the flag attribute
        self._ready = True

    def _put_branch_value(self, branch, value):
        if self._dim[branch] == 1:
            self._branches[branch][0] = value 
        else:
            for i, v in enumerate(value):
                self._branches[branch][i] = v

    def fill(self):
        self._tree.Fill()

    def write(self):
        self._tree.Write("", R.TObject.kOverwrite)

    def __setattr__(self, name, value):
        if "_ready" in self.__dict__ :
            self._put_branch_value(name, value)
        else:
            self.__dict__[name] = value