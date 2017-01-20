# GeneralisedDDT

The standard DDT method de-correlates a jet substructure variables (e.g. `tau21`) by plotting the substructure profile vs. the rho, or rhoDDT, of the jet, defined as `rho = log(m^2/pT^2)` and `rhoDDT = log(m^2/pT/1GeV)`, resp., and performing a linear correction to remove bias with m and pT. This has the limitations that (1) it requires the substructure variable to be linear with rho, or a similar variable, and (2) it might require restricting the kinematic phase-space to regions where this linearity holds.

The idea is to generalise the DDT to (1) not require linearity, or any other explicit functional form, and (2) impose no restrictions on kinematic phase space. One way to do this is to plot the substructure profile in 2D, as a function of `m` and `pT`, and perform a non-parametric fit to this profile in order to remove bias. The same method can be used to fit the RMS profile. This can be used to rescale the substructure variable in question, leading to an overall de-correlation method which applies a (`m`, `pT`)-dependent affine transformation to the substructure variable which leaves the distribution with unit mean and fixed with across kinematic phase space.


## Workflow
The generalised DDT method described above is implemented in the following three python scripts:

1. [hyperparameterOptimisation.py](hyperparameterOptimisation.py): Read in and pre-process data; perform grid search with K-fold cross validation to find optimal hyperparameters for kernel regression of substructure mean- and RMS profiles;
2. [ensemblefitting.py](ensemblefitting.py): Using optimal hyperparameters, use bagging to fit an ensemble of kernel regression estimators;
3. [decorrelation.py](decorrelation.py): Using the fitted ensemble of estimators, compute the de-correlated substructure variable and perform performance studies.

The [common.py](common.py) file includes various utility functions which are used by some or all of the three scripts above.


## Dependencies

The various scripts in this repository depend on `numpy`, `root_numpy`, `scipy`, `scikit-learn`, and `matplotlib` for data handling, fitting and plotting. These either need to be installed manually using a package manager such as pip or, if working on the CERN lxplus cluster, they can be used by specifying the correct environment variables e.g. using the [pythonenv.sh](github.com/asogaard/scripts/pythonenv.sh) script in the [scripts](github.com/asogaard/scripts) repository.