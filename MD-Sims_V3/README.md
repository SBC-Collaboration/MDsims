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

# Recommended implementation on Fermilab flexible computing facility
When on Fermilab network (or on Fermilab VPN), access the flexible computing facility at
https://analytics-hub.fnal.gov/hub/home

Creating a server from that link launches a VM and jupyterlab server.

For first-time use, clone this repository in your user directory on the VM (this directory persists across VM sessions), via a terminal tab in jupyterlab.
It's recommended to set up ssh-key authentication for github, per the instructions here:
 -- generating key on VM:  https://docs.github.com/en/enterprise-server@3.12/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
 -- adding key to github account:  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

For first-time use, also create the md conda environt as above, and define the python kernel you'll want to use.  You probably need these commands:
> conda init
> 
> source ~/.bashrc
> 
> (conda environment commands above)
> 
> python -m ipykernel install --user --name <your_kernel_name>

If the conda md enviroment is created already, you can also source the launch_hoomd_kernel.sh script in this repository

Finally, refresh your jupyterlab window, and link your notebook to the kernel you've created

# Description
This repository contains python modules and jupyter notebooks for the simulation of bubble nucleation in superheated liquids in hoomd (https://glotzerlab.engin.umich.edu/hoomd-blue/).  This includes tools to prepare (and save) the superheated state, inject heat into that state, evolve the state, and evaluate whether that evolution results in bubble growth.  There are also tools for common plot/visualization commands, etc.

# Contents
'sbcmd' contains the following importable libraries:
  'thermocalcs.py' --- functions to evaluate thermophysical properties of simulated fluids via HOOMD simulation
  
