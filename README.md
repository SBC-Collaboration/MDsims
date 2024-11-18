# MDsims
HOOMD MD sims of bubble nucleation

# Recommended Install
After a fresh install of miniconda, you can do the following:

```
conda update conda
conda config --add channels conda-forge
conda create -n md hoomd
conda activate md
conda install matplotlib scipy ipython fresnel jupyter h5py gsd pandas ipympl
```

# Description
This repository contains python modules and jupyter notebooks for the simulation of bubble nucleation in superheated liquids in hoomd (https://glotzerlab.engin.umich.edu/hoomd-blue/).  This includes tools to prepare (and save) the superheated state, inject heat into that state, evolve the state, and evaluate whether that evolution results in bubble growth.  There are also tools for common plot/visualization commands, etc.

# Contents
'sbcmd' contains the following importable libraries:
  'thermocalcs.py' --- functions to evaluate thermophysical properties of simulated fluids via HOOMD simulation
  
