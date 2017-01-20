#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Script for performing ensemble fitting for GDDT with optimal hyperparameters. '''

# Basic.
import sys
import os
import itertools
import pickle
from multiprocessing import Pool
import time

# Local utilitiy files.
from common import *

# ROOT.
from ROOT import *

# Try to import necessary external packages.
# Require correct environment setup on lxplus.
try:
    import numpy as np

    from sklearn.kernel_ridge import KernelRidge
    from sklearn import preprocessing
    from sklearn.base import clone

    import matplotlib as ml
    from matplotlib import rc
    import matplotlib.pyplot as plt
except:
    print "ERROR: Scientific python packages were not set up properly."
    print " $ source ~/pythonenv.sh"  
    print "or see e.g. [http://rootpy.github.io/root_numpy/start.html]."
    pass

# Utility function(s).

# --------------------------------------------------------------------
# Set configuration of any object based on dict.
def setConfig(obj, config):
    for attr, val in config.iteritems():
        setattr(obj, attr, val)
        pass
    return obj


# Main function.
def main ():

    # ==========================================================================
    # Setup.
    # --------------------------------------------------------------------------
    print "Setup."

    # ..
    summary = ""

    # Initialise output directory, and create if necessary.
    output_dir = './output/ensemblefitting/'

    checkMakeDirectory(output_dir)

    # Validate arguments.
    validateArguments(sys.argv)

    # Get input data.
    data, Njets = getData(sys.argv)
    
    # Initialise substructure variables.
    substructurevars = ['D2'] # ['tau21', 'D2', 'logD2']
    types = ['mean', 'std']
    
    # Initialise axis variable names.
    varx, vary = ('logm', 'logpt')

    xlabel = displayNameUnit(varx, latex = True)
    ylabel = displayNameUnit(vary, latex = True)

    # Check that the hyperparameter optimisation has been performed for the given choice of variables.
    print "-- Check hyperparameter optimisation."
    hyperopt_dir = "./output/hyperparameteroptimisation/"
    hyperopt_README = hyperopt_dir + "README.txt"
    hyperopt_xscaler = hyperopt_dir + "scaler_%s.pkl" % varx
    hyperopt_yscaler = hyperopt_dir + "scaler_%s.pkl" % vary
    hyperopt_bins    = hyperopt_dir + "bins.pkl"
     
    hyperopt_files = [hyperopt_README, hyperopt_xscaler, hyperopt_yscaler, hyperopt_bins]
    if False in map(os.path.isfile, hyperopt_files):
        msg = "The following file(s):\n"
        for f in hyperopt_files:
            if not os.path.isfile(f):
                msg += "  %s\n" % f
                pass
            pass
        msg += "were not found. This indicates that a hyperparameter optimisation was not performed for axis variables %s and %s, or that is was not performed correctly.\n" % (varx, vary)
        raise IOError(msg)
   
    # Load scalers from hyperparameter optimisation
    xscaler = pickle.load( open(hyperopt_xscaler, 'rb') )
    yscaler = pickle.load( open(hyperopt_yscaler, 'rb') )

    # Scale to unit range
    print "-- Scale axis variables to unit range."
    varxscaled = '%s_scaled' % varx
    varyscaled = '%s_scaled' % vary
    
    data[varxscaled] = xscaler.transform(data[varx][:, np.newaxis])
    data[varyscaled] = yscaler.transform(data[vary][:, np.newaxis])
    
    varx, vary = varxscaled, varyscaled
    
    xlabel += " (scaled)"
    ylabel += " (scaled)"
    
    # Initialise bin and axis variables.
    print "-- Initialise bin and axis variables."
    bins = pickle.load( open(hyperopt_bins, 'rb') )
    midpoints = (bins[1:] + bins[:-1]) * 0.5
    meshx, meshy = np.meshgrid(midpoints, midpoints)
    meshX = np.column_stack((meshx.ravel(), meshy.ravel()))
    
    x = data[varx]
    y = data[vary]
    w = data['weight']

    for var, t in itertools.product(substructurevars, reversed(types)):
        
        print "\nFitting %s of %s." % (t, var)
        
        # ==================================================================
        # Fitting ensemble of regressors.
        # ------------------------------------------------------------------

        # @TODO: - Save summary file, listing used parameters etc.
        
        # Define easy-to-access data array for the current substructure variable.
        z = data[var]

        # Load optimal config.
        hyperopt_config = hyperopt_dir + "optimal_hyperparameters__%s__%s__vs__%s__%s.pkl" % (t, var, varx, vary)
        if not os.path.isfile(hyperopt_config):
            print "WARNING: No optimal configuration for substructure variable %s vs. %s and %s was found." % (var, varx, vary)
            print "         Looking here: '%s'" % hyperopt_config
            print "         Continuing."
            continue

        config = pickle.load( open(hyperopt_config, 'rb') )

        # Initialise base regressor.
        clf = KernelRidge(kernel = 'rbf')
        setConfig(clf, config)

        # Initialise ensemble settings.
        Nensemble = 10
        ensemble = list()
        seed     = 21
        np.random.seed(seed)
        frac     = 1./np.sqrt(Nensemble)

        # Initialise penalty variables.
        min_unweighted_entries = 3
        err_penalty = 9999.

        print "Ensembling %d KernelRidge regressors with optimal parameters:" % Nensemble
        print " ",config
        print "  fitting each with to a random sample of %.01f%% of the available data," % (frac * 100.)
        print "  using numpy random seed:", seed

        # Loop ensemble.
        for i in range(Nensemble):
            print "-- Classifier %d/%d." % (i + 1, Nensemble)
            
            # Clone best classifier.
            ensemble.append(clone(clf))
            
            # Generate new sample.
            sample = np.random.choice(Njets, int( frac * Njets ), True) # True: bagging
            
            # Compute the mean profile for this subsample
            sample_profile, _, sample_weight = project(x[sample], y[sample], z[sample], w[sample], bins, bins, t)

            # Fit the current ensemble regressor.
            ensemble[-1].fit(meshX, sample_profile.ravel(), sample_weight.ravel())

            # Save classifier to file.
            clf_name = output_dir + "estimator_%s_%s_%02d.pkl" % (t, var, i)
            pickle.dump( ensemble[-1], open(clf_name, 'wb') )

            pass

        print "Done."

    # README ...
    
    return        
    

# Main function call.
if __name__ == '__main__':
    main()
    pass
