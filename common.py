#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Common definitions and utility function for generalised DDT. '''

# Basic include(s).
import sys
import os
import itertools
from functools import reduce

# HEP include(s).
from ROOT import *

# Scientific include(s).
try:
    import numpy as np
    from numpy.lib.recfunctions import append_fields
    from root_numpy import tree2array, fill_profile, fill_hist, hist2array

    from scipy import ndimage

    import matplotlib as ml
    import matplotlib.pyplot as plt
except:
    print "ERROR: Scientific python packages were not set up properly."
    print " $ source ~/pythonenv.sh"
    print "or see e.g. [http://rootpy.github.io/root_numpy/start.html]."
    raise 
    pass


# Settings.
ml.rc('font',**{'family':'serif','serif':['Palatino'],'size':16})
ml.rc('text', usetex=True)
ml.rcParams['image.cmap'] = 'inferno'

coloursROOT = [kViolet + 7, kAzure + 7, kTeal, kSpring - 2, kOrange - 3, kPink]
colours     = [(tc.GetRed(), tc.GetGreen(), tc.GetBlue()) for tc in map(gROOT.GetColor, coloursROOT)]

# --------------------------------------------------------------------

def project (x, y, z, w, binsx = None, binsy = None, t = 'mean'):

    # Input checks.
    if t not in ['mean', 'std']:
        msg = "project accepts only 'mean' and 'std' as type options, not ''." % t
        raise ValueError(msg)
    
    if (binsx is None) != (binsy is None):
        msg = "Cannot specify only one set of bins."
        raise ValueError(msg)
    elif binsx is None:
        print "No bins were provided. Assuming 50 bins on [0, 1] for both axes."
        binsx = np.linspace(0, 1, 50 + 1, True)
        binsy = binsx
        pass

    # Define minimum number of MC entries each bin needs to have in order to be considered. Done to remove isolated bins with excessive weights.
    min_unweighted_entries = 5
    err_penalty            = 9999.

    profile_weighted_entries,  _  = computeHistVec(x, y, binsx, binsy, w)
    profile_unweighted_entries, _ = computeHistVec(x, y, binsx, binsy)

    # Fill histogram(s).
    if t == 'mean':
        profile, profile_err           = computeProfileVec(x, y, z, binsx, binsy, w)

    elif t == 'std':
        _, profile                    = computeProfileVec(x, y, z, binsx, binsy, w, option = 'S')

        # Sample 'Nsample' profile RMS'es, and compute the standard deviation directly.
        profiles_sample = list()
        N = x.size
        Nsample = 10
        for _ in xrange(Nsample):
            sample = np.random.choice(N, N/2, True)
            xsample = x.ravel()[sample]
            ysample = y.ravel()[sample]
            zsample = z.ravel()[sample]
            wsample = w.ravel()[sample]
            _, tmp_profile = computeProfileVec(xsample, ysample, zsample, binsx, binsy, wsample, option = 'S')

            # Append to list
            profiles_sample.append(tmp_profile)
            pass

        profile_err = np.std(profiles_sample, axis = 0)
        pass

    # Penalise bins with too few entries
    profile_err[np.where(profile_err == 0)]                                    = err_penalty
    profile_err[np.where(profile_unweighted_entries < min_unweighted_entries)] = err_penalty

    # Compute weight as 1/sigma.
    profile_weight = np.power(profile_err, -1.)

    # @TEMP
    '''
    vmin, vmax = 0, 5.00
    fig, (ax1, ax2) = plt.subplots(1,2, sharex = True, sharey = True, figsize = (11,5))
    ax1.imshow     (profile,        vmin = vmin, vmax = vmax, origin = 'lower', interpolation = 'none')
    im = ax2.imshow(profile_weight,                           origin = 'lower', interpolation = 'none')
    plt.colorbar(im)
    plt.show()
    '''
    # Outlier removal.
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])

    kernel_expectation = ndimage.convolve(profile_weight, kernel) / np.sum(kernel)
    reldev = np.abs(profile_weight - kernel_expectation) / kernel_expectation
    msk_outlier = np.where(reldev > 20.)
    profile_weight[msk_outlier] = 1./9999.
    profile_err   [msk_outlier] = 9999.

    return profile, profile_err, profile_weight


# --------------------------------------------------------------------

def displayName (var, latex = False):
    output = var

    # tau21
    if   var == "tau21":                output = "#tau_{21}"
    elif var == "tau21_ut":             output = "#tau_{21,untrimmed}"
    elif var == "tau21_mod_rhoPrime":   output = "#tilde{#tau}_{21} "
    elif var == "tau21_mod_rhoDDT":     output = "#tau_{21}^{DDT}"
    elif var == "tau21_SDDT":           output = "#tau_{21}^{(S)DDT}"
    elif var == "tau21_GDDT":           output = "#tau_{21}^{(G)DDT}"
    # D2
    elif var == "D2":                   output = "D_{2}"
    elif var == "D2mod":                output = "#tilde{D}_{2}"
    elif var == "D2_SDDT":              output = "D_{2}^{(S)DDT}"
    elif var == "D2_GDDT":              output = "D_{2}^{(G)DDT}"
    # Kinematic variables
    elif var.lower() == "pt":           output = "p_{T} "
    elif var.lower() == "m":            output = "M"
    # rho
    elif var == "rho":                  output = "#rho"
    elif var == "rho_ut":               output = "#rho_{untrimmed}"
    elif var == "rhoPrime":             output = "#rho'"
    elif var == "rhoPrime_ut":          output = "#rho'_{untrimmed}"
    elif var == "rhoDDT":               output = "#rho^{DDT}"
    elif var == "rhoDDT_ut":            output = "#rho^{DDT}_{untrimmed}"
    # log(...)
    elif var.lower().startswith('log'): output = "#log(%s)" % displayName(var[3:])
    # other
    elif var == "gamma":                output = "#gamma"
    elif var == "alpha":                output = "#alpha"

    return r'$%s$' % output.replace('#', '\\') if latex else output

# --------------------------------------------------------------------

def displayUnit (var):
    if   var.lower() == 'pt':    return 'GeV'
    elif var.lower() == 'm':     return 'GeV'
    elif var.lower() == "logm":  return "log(%s)" % displayUnit("m")
    elif var.lower() == "logpt": return "log(%s)" % displayUnit("pt")
    return ''

# --------------------------------------------------------------------

def displayNameUnit (var, latex = False):
    name = displayName(var, latex)
    unit = displayUnit(var)
    return name + (r" [%s]" % unit if unit else unit)

# --------------------------------------------------------------------
# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Taken from here: [http://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting]

    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if not exponent:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if not precision:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

# --------------------------------------------------------------------

def validateArguments(args):
    if len(args) == 1:
        msg  = "Please specify at least one target ROOT file. Run as:\n"
        msg += " $ python %s path/to/file.root" % args[0]
        raise IOError(msg)
    return

# --------------------------------------------------------------------

def checkMakeDirectory(path):
    if path.startswith('/'):
        raise ValueError("'checkMakeDirectory' does not accept absolute paths (%s)." % paths)
    if not os.path.isdir(path):
        os.makedirs(path)
        pass
    return

# --------------------------------------------------------------------

def loadXsec (path):
    ''' Load cross section weights from file. '''

    xsec = dict()
    with open(path, 'r') as f:
        for l in f:
            line = l.strip()
            if line == '' or line.startswith('#'):
                continue
            fields = [f.strip() for f in line.split(',')]
            try:
                if int(fields[2]) == 0:
                    continue
                xsec[int(fields[0])] = float(fields[1]) / float(fields[2]) * float(fields[3])
            except:
                # If data.
                continue
            pass
        pass
    return xsec


# --------------------------------------------------------------------

def loadDataFast (paths, treename, branches, prefix = '', xsec = None, ignore = None, keepOnly = None, DSIDvar = 'DSID', isMCvar = 'isMC', Nevents = 29):

    print ""
    print "loadDataFast: Reading data from %d files." % len(paths)

    if len(paths) == 0:
        print "loadDataFast: Exiting."
        return dict()

    ievent = 0

    # Initialise data array.
    data = None

    # Initialise DSID variable.
    DSID = None
    isMC = None

    # Loop paths.
    for ipath, path in enumerate(paths):

        # Print progress.
        print "\rloadDataFast:   [%-*s]" % (len(paths), '-' * (ipath + 1)),
        sys.stdout.flush()

        # Get file.
        f = TFile(path, 'READ')

        # Get DSID.
        outputTree = f.Get(treename.split('/')[0] + '/outputTree')
        for event in outputTree:
            DSID = eval('event.%s' % DSIDvar)
            isMC = eval('event.%s' % isMCvar)
            break

        if not DSID:
            print "\rloadDataFast:   Could not retrieve DSID file output is probably empty. Skipping."
            continue

        # Check whether to explicitly keep or ignore.
        if keepOnly and DSID and not keepOnly(DSID):
            print "\rNot keeping DSID %d." % DSID
            continue
        elif ignore and DSID and ignore(DSID):
            print "\rloadDataFast:   Ignoring DSID %d." % DSID
            continue

        # Get tree.
        t = f.Get(treename)

        if not t:
            print "\rloadDataFast:   Tree '%s' was not found in file '%s'. Skipping." % (treename, path)
            continue

        # Load new data array.
        arr = tree2array(t,
                         branches = [prefix + br for br in branches],
                         include_weight = True,
                         )
        # Add cross sections weights.
        if isMC and xsec and DSID:

            # Ignore of we didn't provide cross section information.
            if DSID not in xsec:
                print "\rloadDataFast:   Skipping DSID %d (no sample info)." % DSID
                continue

            # Scale weight by cross section.
            arr['weight'] *= xsec[DSID]

            # Add DSID array.
            arr = append_fields(arr, 'DSID', np.ones(arr['weight'].shape) * DSID)

            pass

        # Append to existing data arra
        if data is None:
            data = arr
        else:
            data = np.concatenate((data, arr))
            pass

        pass

    print ""

    # Change branch names to remove prefix.
    data.dtype.names = [name.replace(prefix, '') for name in data.dtype.names]

    # Dict-ify.
    values = dict()
    for branch in data.dtype.names:
        values[branch] = data[branch]
        pass

    return values

# --------------------------------------------------------------------

def getData (args):
    '''  '''

    # Load cross sections files.
    print "-- Load cross sections file."
    xsec = loadXsec('../share/sampleInfo.csv')

    # Get list of file paths to plot from commandline arguments.
    print "-- Get list of input paths."
    paths = [arg for arg in args[1:] if not arg.startswith('-')]

    # Specify which variables to get.
    print "-- Specify variables to read."
    treename = 'BoostedJet+ISRgamma/Fatjets/Nominal/BoostedRegime/Postcut'
    prefix   = ''

    substructurevars = ['plot_object_%s' % var for var in ['tau21', 'D2']]
    getvars  = ['m', 'pt'] + substructurevars

    # Load data.
    print "-- Load data."
    values = loadDataFast(paths, treename, getvars, prefix, xsec,
                          keepOnly = (lambda DSID: 361039 <= DSID <= 361062 )
                          )

    # @TOFIX: Temporary fix for inconstistent branch names
    print "-- Rename variables (temporary)."
    for var in substructurevars:
        values[var.replace('plot_object_', '')] = values.pop(var)
        pass
    substructurevars = [var.replace('plot_object_', '') for var in substructurevars]

    # Check output.
    print "-- Check output exists."
    if not values:
        print "WARNING: No values were loaded."
        return

    # Discard unphysical jets.
    print "-- Discard unphysical jets. No other jets should be excluded."
    msk_good = reduce(np.intersect1d, (np.where(values['pt']    > 0),
                                       np.where(values['m']     > 0),
                                       np.where(values['tau21'] > 0),
                                       np.where(values['D2']    > 0)))

    for var, arr in values.items():
        values[var] = arr[msk_good]
        pass

    # Initialise total number of jets.
    print "-- Initialise number of (good) jets."
    num_examples = len(values[getvars[0]])

    # Compute new variables.
    print "-- Compute new variables."
    values['rho']   = np.log(np.power(values['m'], 2) / np.power(values['pt'], 2))

    values['logm']     = np.log(values['m'])
    values['logpt']    = np.log(values['pt'])
    values['logD2']    = np.log(values['D2'])
    values['logtau21'] = np.log(values['tau21'])

    return values, num_examples

# --------------------------------------------------------------------
# Return 'iterable' in batches of (at most) 'n'.
def batch (iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        pass
    pass


# --------------------------------------------------------------------
def setConfig(obj, config):
    ''' Set properties of instance using (string, value) pairs '''
    for attr, val in config.iteritems():
        setattr(obj, attr, val)
        pass
    return obj

# --------------------------------------------------------------------
# ...
def computeProfileVec (vecx, vecy, vecz, binsx, binsy, weights = None, option = '', cls = TProfile2D):
    matrix = np.column_stack((vecx, vecy, vecz))
    return computeProfileMat(matrix, binsx, binsy, weights, option, cls)

# --------------------------------------------------------------------
# ...
def computeHistVec (vecx, vecy, binsx, binsy, weights = None, option = '', cls = TH2F):
    matrix = np.column_stack((vecx, vecy))
    return computeProfileMat(matrix, binsx, binsy, weights, option, cls)

# --------------------------------------------------------------------
# ...
def computeProfileMat (matrix, binsx, binsy, weights = None, option = '', cls = TProfile2D):

    if weights is None:
        weights = np.ones((matrix.shape[0],))
    elif len(weights) != matrix.shape[0]:
        ValueError("Number of samples (%d) and weights (%d) do not agree." % (matrix.shape[0], len(weights)))
        pass

    nx = len(binsx) - 1
    ny = len(binsy) - 1

    if cls == TProfile2D:
        profile = cls('profile', "", nx, binsx, ny, binsy, option)
        fill_profile(profile, matrix, weights = weights)
    else:
        profile = cls('hist',    "", nx, binsx, ny, binsy)
        fill_hist(profile, matrix, weights = weights)
        pass

    means  = np.zeros((nx,ny))
    errors = np.zeros((nx,ny))
    for (i,j) in itertools.product(xrange(nx), xrange(ny)):
        means [i,j] = profile.GetBinContent(j + 1, i + 1)
        errors[i,j] = profile.GetBinError  (j + 1, i + 1)
        pass

    return means, errors
