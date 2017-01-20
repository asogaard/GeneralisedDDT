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
from AFunctions import *
from AStyle import *
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
# Use classifier tp produce predictions (wrapper function)
def predict (clf, data):
    return clf.predict(data)

# --------------------------------------------------------------------
# Produce prediction using asynchronous processes.
def asyncPredict (clf, data, num_processes = 10, batch_size = 10000):

    # Create multiprocessing pool.
    pool = Pool()
    timeout = 9999999999

    # Get number of examples for which to produce predictions.
    num_examples = data.shape[0]

    # Compute suitable batch size.
    num_rounds = int(num_examples/float(num_processes * batch_size)) + 1

    # Loop batches.
    print "Total number of examples: %d (%d is maximal index)" % (num_examples, num_examples - 1)
    predictions = list()
    for iround, round_indices in enumerate(batch(np.arange(num_examples), num_processes * batch_size), start = 1):
        results = list()
        print "asyncPredict: Round %d of %d." % (iround, num_rounds)
        for indices in batch(round_indices, batch_size):

            # Submit prediction as asynchronous process.
            args = [clf, data[indices,:]]
            results.append( pool.apply_async(predict, args) )            
            pass

        # Collect predictions.
        predictions += [result.get(timeout = timeout) for result in results]
        pass

    # Return predictions.
    return np.hstack(predictions)


# Main function.
def main ():

    # ==========================================================================
    # Setup.
    # --------------------------------------------------------------------------
    print "Setup."

    # ..
    summary = ""

    # Initialise output directory, and create if necessary.
    output_dir = './output/decorrelation/'

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
    hyperopt_README  = hyperopt_dir + "README.txt"
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
    X = np.column_stack((x,y))

    for var, t in itertools.product(substructurevars, reversed(types)):

        # Define easy-to-access data array for the current substructure variable.
        z = data[var]

        # Set axis min- and maximum
        vmin = 0.00
        if t == 'mean':
            if   var == 'tau21': vmax = 0.65
            elif var == 'D2':    vmax = 7.00
            else:                vmax = 3.00
        else:
            if   var == 'tau21': vmax = 0.20
            elif var == 'D2':    vmax = 4.00
            else:                vmax = 1.00
            pass
        
        if t == 'mean':
            zlabel = r'$\langle%s\rangle$' % displayNameUnit(var, latex = True).replace('$', '')
        else:
            zlabel = r'RMS(%s)' % displayNameUnit(var, latex = True)
            pass

        # Draw profile for (m, pT).
        bins_m  = np.linspace(  0,  300, 50 + 1, True)
        bins_pt = np.linspace(150, 1500, 50 + 1, True)
        midpoints_m  = (bins_m [1:] + bins_m [:-1]) * 0.5
        midpoints_pt = (bins_pt[1:] + bins_pt[:-1]) * 0.5
        mesh_m, mesh_pt = np.meshgrid(midpoints_m, midpoints_pt)
        profile_m_pt, _, _ = project(data['m'], data['pt'], z, w, bins_m, bins_pt, t)

        plt.pcolormesh(mesh_m, mesh_pt, profile_m_pt, vmin = vmin, vmax = vmax)
        plt.xlim([bins_m [0], bins_m [-1]])
        plt.ylim([bins_pt[0], bins_pt[-1]])
        plt.xlabel(displayNameUnit('m',  latex = True))
        plt.ylabel(displayNameUnit('pt', latex = True))
        cb = plt.colorbar()
        cb.set_label(zlabel, labelpad=20)
        plt.savefig(output_dir + 'profile_%s_m_pt.pdf' % var)
        plt.show()

        # Draw profile for (logm, logpT).
        bins_logm  = np.linspace(np.log(np.min(data['m'])),  np.log(np.max(data['m'])),  50 + 1, True)
        bins_logpt = np.linspace(np.log(np.min(data['pt'])), np.log(np.max(data['pt'])), 50 + 1, True)
        midpoints_logm  = (bins_logm [1:] + bins_logm [:-1]) * 0.5
        midpoints_logpt = (bins_logpt[1:] + bins_logpt[:-1]) * 0.5
        mesh_logm, mesh_logpt = np.meshgrid(midpoints_logm, midpoints_logpt)
        profile_logm_logpt, _, _ = project(data['logm'], data['logpt'], z, w, bins_logm, bins_logpt, t)

        plt.pcolormesh(mesh_logm, mesh_logpt, profile_logm_logpt, vmin = vmin, vmax = vmax)
        plt.xlim([bins_logm [0], bins_logm [-1]])
        plt.ylim([bins_logpt[0], bins_logpt[-1]])
        plt.xlabel(displayNameUnit('logm',  latex = True))
        plt.ylabel(displayNameUnit('logpt', latex = True))
        cb = plt.colorbar()
        cb.set_label(zlabel, labelpad=20)
        plt.savefig(output_dir + 'profile_%s_logm_logpt.pdf' % var)
        plt.show()

        # Draw profile for (logm_scaled, logpT_scaled).
        profile_logmscaled_logptscaled, _, _ = project(x, y, z, w, bins, bins, t)

        plt.pcolormesh(meshx, meshy, profile_logmscaled_logptscaled, vmin = vmin, vmax = vmax)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cb = plt.colorbar()
        cb.set_label(zlabel, labelpad=20)
        plt.savefig(output_dir + 'profile_%s_%s_logmscaled_logptscaled.pdf' % (t, var))
        plt.show()

        # Get mean profile and error.
        print "---- Get mean profile and error."
        mean, mean_err, _ = project(x, y, z, w, bins, bins, t)
        

        # ==================================================================
        # Compute modified variable using asynchronous processes.
        # ------------------------------------------------------------------

        # Initialise ensemble settings.
        Nensemble = 10
        ensemble = list()
        fitting_dir = "./ensemblefitting/"

        # Load regressor ensemble.
        for i in range(Nensemble):
            clf_name = fitting_dir + "estimator_%s_%s_%02d.pkl" % (t, var, i)
            print "Loading estimator '%s'." % clf_name
            if not os.path.isfile(clf_name):
                print "WARNING: Classifier '%s' was not found. Continuing." % clf_name
                continue
            ensemble.append(pickle.load( open(clf_name, 'rb') ))
            #print ensemble[-1].dual_coef_
            pass


        # Initialise asynchronous processing variables.
        num_processes = 10
        mvar = t + '_' + var
        print "\nPredicting modified variable '%s' using %d asynchronous processes." % (mvar, num_processes)

        # Perform ensemble prediction.
        data[mvar] = np.zeros((Njets,))
        for icl, cl in enumerate(ensemble, start = 1):
            print "-- Classifier %d/%d." % (icl, Nensemble)
            start = time.time()
            data[mvar] += asyncPredict(cl, X, num_processes)
            end  = time.time()
            print "---- Elapsed: %.1fs" % (end - start)
            pass
        data[mvar] /= float(len(ensemble))
        print ""

        # Plotting the profiles and densities
        if t == 'mean':
            label = r'$\langle {%s} \rangle$' % displayName(var, latex = True).replace('$', '')
        else:
            label = r'RMS(%s)' % displayName(var, latex = True)
            pass
        fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (11,9))
        fig.suptitle(r'%s profiles' % label, fontsize=21)

        # -- Mesh
        nbins     = 50
        nbinsfine = 50
        meshxfine, meshyfine = np.meshgrid(np.linspace(0, 1, nbinsfine, True),
                                           np.linspace(0, 1, nbinsfine, True))

        # -- (1)
        mean = mean.reshape(meshx.shape)
        ax[0,0].pcolormesh(meshx, meshy, mean, vmin = vmin, vmax = vmax)
        ax[0,0].set_xlim([0,1])
        ax[0,0].set_ylim([0,1])
        ax[0,0].set_ylabel(ylabel)
        ax[0,0].set_title('Profile of %s measurements' % label, fontsize = 16)

        # -- (2)
        zpred = data[mvar]
        zpredprofile, _, _ = project(x, y, zpred, w, bins, bins, 'mean') 
        # we either want to profile the predicted quantity, e.g. 'mean(D2)' or 'RMS(D2)', but in either case we want the _average_ of that value, even if the value itself in a measure of standard deviations

        ax[0,1].pcolormesh(meshx, meshy, zpredprofile, vmin = vmin, vmax = vmax)
        ax[0,1].set_title(r'Ensemble-averaged (%d) estimator' % Nensemble, fontsize = 16)

        print "=" * 70
        print np.mean(mean)
        print np.std (mean)
        print np.mean(zpredprofile)
        print np.std (zpredprofile)
        print "=" * 70

        # -- (3)
        zdiff = mean.ravel() - zpredprofile.ravel()
        mean_err[np.where(mean_err == 0)] = 9999.
        zpull = zdiff / mean_err.ravel()
        ax[1,0].pcolormesh(meshx, meshy, zpull.reshape(meshx.shape), vmin = -2., vmax = 2., cmap = 'RdBu')
        ax[1,0].set_xlabel(xlabel)
        ax[1,0].set_ylabel(ylabel)
        ax[1,0].set_title(r'Residual pulls ($\pm 2 \sigma$)', fontsize = 16)

        # -- (4)
        print "Computing meshzfine."
        meshXfine = np.column_stack((meshxfine.ravel(), meshyfine.ravel()))
        meshzfine = np.zeros(meshxfine.shape)
        for i, cl in enumerate(ensemble):
            print "-- Classifier %d." % (i + 1)
            meshzfine += cl.predict(meshXfine).reshape(meshxfine.shape)
            pass
        meshzfine /= float(len(ensemble))

        im = ax[1,1].pcolormesh(meshxfine, meshyfine, meshzfine, vmin = vmin, vmax = vmax, shading='gouraud')
        ax[1,1].set_xlabel(xlabel)
        ax[1,1].set_title('Ensamble-averaged (%d) estimator,\nfunction' % Nensemble, fontsize = 16)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
        fig.colorbar(im, cax=cbar_ax, label = '%s' % label)

        print "(Non-zero) residual pulls:"
        msk = np.where(zpull != 0)
        #print zpull[msk]
        print "-- Mean: %.03f" % np.mean(zpull[msk])
        print "-- RMS:  %.03f" % np.std (zpull[msk])

        # Save the figure
        fig.savefig(output_dir + "plot_%s_%s_profiles.pdf" % (t, var))
        plt.show()

        pass

    
    # Perform _actual_ decorrelation
    lines = [ "#sqrt{s} = 13 TeV",
              "Inclusive #gamma MC",
              "Trimmed anti-k_{t}^{R=1.0}",
              "Req. 1 #gamma with p_{T} > 155 GeV",
              "Jet p_{T} > 150 GeV",
              "Jet p_{T} > 2 M (boosted)",
              ]

    for rms in [True, False]:
        for var in substructurevars:
            
            mvar = var + '_GDDT'
            
            # Plot correlation with jet mass
            data[mvar] = (data[var].ravel() - data['mean_' + var].ravel()) / data['std_' + var].ravel() * 0.2 + 1.
            
            
            slices = [
                ( 150,  200),
                ( 400,  500),
                (1000, 1500)
                ]
            
            profiles = dict()
            for v in [var, mvar]:
                
                profiles[v] = list()
                for i, sl in enumerate(slices):
                    profiles[v].append( TProfile("profile_%s_vs_m_slice_%d" % (v, i), "", 50, (-2 + i * 2) if rms else 0, (298 + i * 2) if rms else 300, ('S' if rms else '')) )
                    profiles[v][-1].GetXaxis().SetRangeUser(0, 300.)
                    msk = np.where((data['pt'] >= sl[0]) & (data['pt'] < sl[1]))
                    arr = np.column_stack((data['m'].ravel()[msk], data[v].ravel()[msk]))
                    fill_profile(profiles[v][-1], arr, data['weight'].ravel()[msk])
                    pass
                
                names = ['[%d, %d] GeV' % (sl[0], sl[1]) for sl in slices]
                legendOpts = LegendOptions(histograms = profiles[v],
                                           header = 'Jet p_{T} in:',
                                           names = names,
                                           xmin = 0.59,
                                           ymax = 0.835)
                
                textOpts   = TextOptions(lines = lines)
                
                ymax = (2.5 if v.endswith('_GDDT') else (5.0 if v.startswith('D2') else None))
                
                c = makePlot( profiles[v],
                              legendOpts,
                              textOpts,
                              padding = 1.2,
                              ymin = 0.0,
                              ymax = ymax,
                              xtitle = "%s" % (displayNameUnit('m')),
                              ytitle = "#LT%s#GT" % (displayName(v)) + (' #pm RMS' if rms else ''),
                              ylines = (([1.0] + ([0.8, 1.2] if rms else [])) if v == mvar else []))
                
                c.SaveAs(output_dir + 'profile_%s_vs_m_decorrelation%s.pdf' % (v, '_pmRMS' if rms else ''))
                raw_input('...')

                pass            

            pass

        pass


    # Compare jet mass spectra
    for var in substructurevars:

        mvar = var + '_GDDT'

        perc = [1, 10, 20, 40, 60, 80, 99]
        slices = { v : 
                   zip(np.percentile(data[v], perc[:-1]), 
                       np.percentile(data[v], perc[1:] ))
                   for v in [var, mvar]
                   }

        spectra = dict()
        for v, log in itertools.product([var, mvar], [True, False]):
            
            spectra[v] = list()
            for i, sl in enumerate(slices[v]):
                spectra[v].append( TH1F("jetmassspectrum_%s_slice_%d" % (v, i), "", 50, 0, 300) )
                spectra[v][-1].Sumw2()
                msk = np.where((data[v] >= sl[0]) & (data[v] < sl[1]))
                fill_hist(spectra[v][-1], data['m'].ravel()[msk], data['weight'].ravel()[msk])
                
                spectra[v][-1].Scale(1./spectra[v][-1].Integral())
                pass
            
            print "Integrals:"
            for spectrum in spectra[v]:
                print " ", spectrum.Integral()
                pass
            
            names = ['[%.2f, %.2f]' % (sl[0], sl[1]) for sl in slices[v]]
            legendOpts = LegendOptions(histograms = spectra[v],
                                       header = 'Jet %s in:' % displayName(v),
                                       names = names,
                                       xmin = 0.59,
                                       ymax = 0.835,
                                       types = 'L')
            
            textOpts   = TextOptions(lines = lines)
            
            c = makePlot( spectra[v],
                          legendOpts,
                          textOpts,
                          padding = 1.2,
                          xtitle = "%s" % (displayNameUnit('m')),
                          ytitle = "Jets (a.u.)",
                          drawOpts = 'HIST',
                          logy = log,
                          #normalise = True,
                          )
            
            c.SaveAs(output_dir + 'jetmassspectrum_%s_%sy.pdf' % (v, 'log' if log else 'lin'))
            raw_input('...')
            pass

        pass
    
    return        
    

# Main function call.
if __name__ == '__main__':
    main()
    pass
