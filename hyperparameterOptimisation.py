#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Script for computing generalised designed de-correlated taggers (GDDT).

...

'''

# Basic.
import sys
import os
import itertools
import pickle
import getpass
from time import gmtime, strftime

#from array import array
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
    from sklearn.cross_validation import train_test_split, KFold
    from sklearn import preprocessing
    from sklearn.base import clone

    from scipy import ndimage

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
# Run CV fold, for parameter grid search.
def runCVfold (clf, X, z, w, binsx, binsy, t, train, test):

    if not t in ['mean', 'std']:
        raise ValueError("runCVfold: 't' not in ['mean', 'std'].")

    # Setup coordinate mesh.
    midpointsx = (binsx[1:] + binsx[:-1]) * 0.5
    midpointsy = (binsy[1:] + binsy[:-1]) * 0.5
    meshx, meshy = np.meshgrid(midpointsx, midpointsy)
    meshX = np.column_stack((meshx.ravel(), meshy.ravel()))

    # Generate 'train' and 'test' splits.
    X_train, X_test = X[train,:], X[test,:]
    z_train, z_test = z[train], z[test]
    w_train, w_test = w[train], w[test]

    # Fill the 'train' profile.
    train_mean, train_err, train_weight = project(X_train[:,0], X_train[:,1], z_train, w_train, binsx, binsy, t)
    
    # Fit the 'train' profile.
    clf.fit(meshX, train_mean.ravel(), sample_weight = train_weight.ravel())
    
    # Fill the 'test' profile.
    test_mean, test_err, test_weight = project(X_test[:,0], X_test[:,1], z_test, w_test, binsx, binsy, t)

    # Get 'test' prediction for the coordinate mesh.
    z_pred = clf.predict(meshX).reshape(test_mean.shape)

    # Get (non-zero- indices for which to compute score.
    msk = np.where(test_err < 9999.)

    # Compute the CV score: mean squared errors.
    z_test = np.array(test_mean)
    s_test = np.array(test_err)
    
    score = np.mean(np.power((z_pred[msk] - z_test[msk]) / s_test[msk], 2.))

    # Return CV score.
    return score


# --------------------------------------------------------------------
# Product of dict entries.
def dict_product(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))


# Main function.
def main ():

    # ==========================================================================
    # Setup.
    # --------------------------------------------------------------------------
    print "Setup."

    # Initialise summary string.
    summary  = "hyperparameterOptimisation.py was run by %s on %s from here.\n" % (getpass.getuser(), strftime("%a, %d %b %Y %H:%M:%S", gmtime()))
    summary += "  %s\n" % os.getcwd()

    # Initialise output directory, and create if necessary.
    output_dir = './output/hyperparameterOptimisation/'

    checkMakeDirectory(output_dir)

    # Validate input arguments
    validateArguments(sys.argv)

    # Get data, as prescribed in common.py
    data, Njets = getData(sys.argv)
    
    summary += "\nRead following %d files:\n" % (len(sys.argv) - 1)
    for path in sys.argv[1:]:
        summary += "  %s\n" % path
        pass

    # Initalise substructure variables to use
    substructure_variables = ['D2'] # ['tau21', 'D2', 'logD2']
    types = ['mean', 'std']
    
    summary += "\nOptimising hyperparameters for substructure variables:\n"
    for subvar in substructure_variables:
        summary += "  %s\n" % subvar
        pass

    # Initialise axis variable names.
    print "-- Initialise axis variable names."
    varx,  vary  = ('logm', 'logpt')

    summary += "\nProfiling substructure variables with as functions of:\n"
    summary += "  %s\n  %s\n" % (varx, vary)
    
    xlabel = displayNameUnit(varx, latex = True)
    ylabel = displayNameUnit(vary, latex = True)
    
    # Scale axis variables to unit range.
    print "-- Scale axis variables to unit range."
    varxscaled = '%s_scaled' % varx
    varyscaled = '%s_scaled' % vary
    
    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()

    data[varxscaled] = xscaler.fit_transform(data[varx][:, np.newaxis])
    data[varyscaled] = yscaler.fit_transform(data[vary][:, np.newaxis])

    # Save scalers to file.
    pickle.dump( xscaler, open( output_dir + "scaler_%s.pkl" % varx, "wb" ) )
    pickle.dump( yscaler, open( output_dir + "scaler_%s.pkl" % vary, "wb" ) )

    summary += "\nSaving preprocessing scalers as:\n"
    summary += "  %s\n  %s\n" % (output_dir + "scaler_%s.pkl" % varx, output_dir + "scaler_%s.pkl" % vary)

    # Rename axis variables and -labels.
    varx, vary = varxscaled, varyscaled
    
    xlabel += " (scaled)"
    ylabel += " (scaled)"
        
    # Initialise bin and axis variables.
    print "-- Initialise bin and axis variables."
    nbins     = 50
    bins = np.linspace(0, 1, nbins + 1, True)
    midpoints = (bins[1:] + bins[:-1]) * 0.5
    meshx, meshy = np.meshgrid(midpoints, midpoints)
    meshX = np.column_stack((meshx.ravel(), meshy.ravel()))
    
    summary += "\nUsing the following %d bins along the (scaled) x- and y-axes:\n" % nbins
    summary += "  [%s]\n" % (', '.join("%.2f" % point for point in bins))

    pickle.dump( bins, open( output_dir + 'bins.pkl', "wb" ) )        

    # Initialise ease-to-access data variables.
    x = data[varx]
    y = data[vary]
    w = data['weight']
    X = np.column_stack((x,y))


    # ==========================================================================
    # Perform hyperparameter optimisation.
    # --------------------------------------------------------------------------

    # Initialise base regressor.
    base_clf = KernelRidge(kernel = 'rbf')
    summary += "\nUsing KernelRidge regressors with 'rbf' kernel for the non-parametric fitting.\n"

    # Initialising parameter grid(s) for scan.
    print "Initialise parameter grid(s) for scan."
    parameters = {
        'mean' : {
            'alpha' : [1.0E-9], # np.logspace( -1,  2, 5 * 1 + 1, True),
            'gamma' : np.logspace(  -1,  4, 5 * 10 + 1, True),
            },
        'std' : {
            'alpha' : [1.0E-9], # np.logspace(-10,  0, 5 * 1 + 1, True),
            'gamma' : np.logspace(  -1,  4, 5 * 10 + 1, True),
            }
        }
    
    summary += "\nScanning the following parameter grid:\n"
    for t in types:
        summary += "  %s\n" % t
        for key, values in parameters[t].iteritems():
            summary += "    %s: [%s]\n" % (key, ', '.join(['%.03e' % val for val in values]))
            pass
        pass

    # Initialise CV setting.
    cv_folds = 10
    summary += "\nRunning %d CV folds for each parameter configuration.\n" % cv_folds

    # Initialise multiprocessing pool.
    pool = Pool()
    timeout = 999999999999

    for var, t in itertools.product(substructure_variables, types):
        
        print "\nPerforming hyperparameter optimisation for %s of %s." % (t, var)

        # Initialise list to hold results for optimisation wrt. current variable.
        cv_results = list()
        
        # Initialise easy-to-access value array for the current substructure variable.
        z = data[var]

        # @TEMP
        project(x, y, z, w, bins, bins, t)

        # Compute the number of jobs to be run in total.
        num_jobs = np.prod([len(l) for l in parameters[t].itervalues()]) * cv_folds
        job_digits = int(np.log10(num_jobs)) + 1

        # Loop parameter configurations in grid.
        for i_config, config in enumerate(dict_product(parameters[t]), start = 1):

            print "-- Running %d CV folds for config: %s" % (cv_folds, ', '.join('%s = %.02e' % (key, val) for key, val in config.iteritems()))
            clf = clone(base_clf)
            setConfig(clf, config)

            # Run 'cv_fold' folds in parallel.
            start = time.time()
            args = [clf, X, z, w, bins, bins, t]
            results = list()
            cv = KFold(z.shape[0], n_folds = cv_folds, shuffle = True)
            for train, test in cv:
                results.append( pool.apply_async(runCVfold, args + [train, test]) )
                pass
            cv_scores = [result.get(timeout = timeout) for result in results]
            end = time.time()

            # Append current configuration and CV scores to list of results.
            cv_results.append((config, cv_scores))
            
            # Print progress.
            cv_score_mean = np.mean(cv_scores)
            cv_score_std  = np.std (cv_scores)
            print "---> [%*d/%*d] Mean CV score: %6.03f +/- %5.03f (%4.1fs)" % (job_digits, i_config * cv_folds, job_digits, num_jobs, cv_score_mean, cv_score_std, end - start)

            pass


        # Plotting the CV curves for the parameter scans
        xpar = 'gamma'
        xvals = dict()
        yvals = dict()
        yerrs = dict()
        keys  = list()
        for config, cv_scores in cv_results:
            key = ', '.join(['%s: %s' % (displayName(key, latex = True), sci_notation(val, 2)) for key, val in config.items() if key != xpar])
            if key not in yvals:
                xvals[key] = np.array([])
                yvals[key] = np.array([])
                yerrs[key] = np.array([])
                keys.append(key)
                pass
            
            xvals[key] = np.append(xvals[key], config[xpar])
            yvals[key] = np.append(yvals[key], np.mean(cv_scores))
            yerrs[key] = np.append(yerrs[key], np.std (cv_scores))
            pass
        
        title = 'Hyperparameter optimisation for '
        if t == 'mean':
            title += "mean(%s)" % displayName(var, latex = True)
        else:
            title += "RMS(%s)" % displayName(var, latex = True)
            pass
        
        # Re-scaling errors from RMS to error in mean
        for key in keys:
            yerrs[key] /= np.sqrt(cv_folds)
            pass

        fig = plt.figure()
        fig.suptitle(title, fontsize = 18)
        for ikey, key in enumerate(keys):
            plt.plot(xvals[key], yvals[key], color = colours[ikey % len(colours)], linewidth = 2.0, label = key)
            plt.fill_between(xvals[key], yvals[key] + yerrs[key], yvals[key] - yerrs[key], color = colours[ikey % len(colours)], linewidth = 2.0, alpha = 0.3)
            pass
        
        plt.grid()
        plt.xlabel(displayName(xpar, latex = True))
        plt.ylabel("Cross-validation scores (mean squared error)")
        plt.xscale('log')
        #plt.yscale('log')
        plt.ylim([0., 10.])
        #if len(keys) > 1:
        plt.legend(prop = {'size':16})
        #    pass
        plt.savefig(output_dir + 'plot_cv_curves__%s_%s__vs__%s__%s.pdf' % (t, var, varx, vary))
        plt.show()
        
        # Get optimal config.
        best_config_index = cv_results.index( min(cv_results, key = lambda pair : np.mean(pair[1])) )
        best_config = cv_results[best_config_index][0]

        print best_config
        print "Score for best config: %6.03f +/- %5.03f" % (np.mean(cv_results[best_config_index][1]), np.std(cv_results[best_config_index][1]))

        output_best_config = 'optimal_hyperparameters__%s__%s__vs__%s__%s.pkl' % (t, var, varx, vary)
        summary += "\nOptimal configuration found for %s of %s was: %s\n" % (t, var, str(best_config))
        summary += "Score for optimal configuration: %6.03f +/- %5.03f\n" % (np.mean(cv_results[best_config_index][1]), np.std(cv_results[best_config_index][1]))
        summary += "Saving optimal hyperparameters to: '%s'\n" % (output_dir + output_best_config)

        # Save optimal hyperparameters.
        pickle.dump( best_config, open( output_dir + output_best_config, "wb" ) )        

        pass # end: loop substructure variables.

    # Saving README file for current run.
    README = open(output_dir + "README.txt", "w")
    README.write(summary)
    README.close()
    
    return


# Main function call.
if __name__ == '__main__':
    main()
    pass
