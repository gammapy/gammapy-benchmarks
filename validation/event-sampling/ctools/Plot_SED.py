#!/usr/bin/env python
# name: ACDC_plot_SED.py
# author: mario <mario@piffio.org>
# version: v2
# date: 2018-02-26
# description:
#   read the output of ctbutterfly, csspec, and generate a plot
#   of the spectrum to be saved into a figure
# changelog:
#  v0 - entry version from show_butterfly.py by Michael Mayer
#  v1 - added filefunction support
#  v2 - spec, butterfly, and model are made optional
# todo:
#  - manage the case when either the butterfly or the spec files are missing
__version__ = '2018-02-16'
# these modules come with the standard Python
import argparse
import xml.etree.ElementTree as ET
import sys, os
try:
  import numpy as np
  import astropy as ap
  import astropy.units as u
  import astropy.io.fits as pf
  from astropy.modeling import models
  import scipy.stats as st
  import matplotlib as mpl
  mpl.use('Agg')
  import matplotlib.pyplot as plt
except:
  print("These Python modules are needed:")
  print("    astropy,    matplotlib")
  print("Please verify that they are correctly installed")
  print("Check also your environment (eg. $PYTHONPATH).")
  print("Cannot load the required Python modules. Exit.")
  sys.exit(1)

# reopen stdout to turn off the buffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')

# plot params
Emin = u.Quantity('0.3 TeV')
Emax = u.Quantity('100 TeV')
x_u = 'TeV' # it must be an energy unit
x_label = 'Energy [TeV]'
x_unit = u.Unit(x_u)
y_u = 'erg cm-2 s-1' # it must be an energy flux unit (F = E**2 dN/dE)
y_label = r'$E^{2}\times\frac{dN}{dE}$ [erg $cm^{-2}$ $s^{-1}$]'
y_unit = u.Unit(y_u)
min_y_ratio = 36. # this ensures at least one major tick
y_grace = 1.2 # further grace in the y axis

def main():
  # parse the command line options
  try:
    options = parse_command_line()
  except IOError as e:
    print("I/O error(%d): %s. Exit." % (e.errno, e.strerror))
    sys.exit(2)
  except Exception: # what other errors can be raised?
    print("Cannot parse the command-line options. Exit.")
    print("Try:", __file__, "-h")
    sys.exit(3)

  spec, butterfly, model, image = validate_options(options)

  # try to read the butterfly contents as Quantities
  E_bf, F_bf, Fmin_bf, Fmax_bf = read_butterfly(butterfly)

  # try to read the spec contents as Quantities
  E_s, dE_d_s, dE_u_s, F_det, Ferr_s, F_ul, TS = read_spec(spec)
  if E_s is not None:
    # select what indexes are considered detections
    ul_mask, F_s = select_detections(TS, F_det, F_ul, options.threshold)

  # try to read the model in the XML file
  E_m = None
  if model is not None:
    # isolate the section with the target
    target = get_target(model, options.name)
    if target is not None:
      # try to parse the section into an SED as Quantities
      E_m, dNdE_m = get_dNdE(target)

  if E_bf is None and E_s is None and E_m is None:
    print('Nothing to plot. Exit')
    return

  plt.loglog()
  plt.grid()
  plt.xlabel(x_label, fontsize=16)
  plt.ylabel(y_label, fontsize=16)

  # if available, plot the estimated flux from the butterfly file
  if E_bf is not None:
    # plot the best-fit model from the butterfly plot
    plt.plot(E_bf.to(x_u)/x_unit, F_bf.to(y_u)/y_unit, \
        color='blue', ls='-', zorder=2, label='Best-fit')
    # plot the shaded area from the butterfly plot
    plt.fill_between(x=E_bf.to(x_u)/x_unit, \
        y1=Fmin_bf.to(y_u)/y_unit, y2=Fmax_bf.to(y_u)/y_unit, \
        color='blue', alpha=0.5, zorder=1) # Bottom

  # if available plot the binned spectrum from csspec
  if E_s is not None:
    plt.errorbar(E_s.to(x_u)/x_unit, F_s.to(y_u)/y_unit, \
        xerr=[dE_d_s.to(x_u)/x_unit, dE_u_s.to(x_u)/x_unit], \
        yerr=Ferr_s.to(y_u)/y_unit, \
        fmt='ro', uplims=ul_mask, zorder=10, label='') # Top

  # if available, plot the expected flux from the XML model
  if E_m is not None:
    F_m = dNdE_m * E_m**2
    plt.plot(E_m.to(x_u)/x_unit, F_m.to(y_u)/y_unit, \
        color='black', zorder=3, label='Simulated model')

  # fix axes ranges and tick marks and labels
  plt.xlim(Emin.to(x_u)/x_unit, Emax.to(x_u)/x_unit)
  y_min, y_max = plt.ylim()
  y_gmean = np.sqrt(y_min * y_max)
  if y_max < min_y_ratio * y_min:
    y_grange = np.sqrt(min_y_ratio)
    y_min = y_gmean / y_grange
    y_max = y_gmean * y_grange
  plt.ylim(y_min / y_grace, y_max * y_grace)
  my_formatter = mpl.ticker.FuncFormatter(my_clean_format)
  plt.gca().xaxis.set_major_formatter(my_formatter)

  plt.legend()
  plt.tight_layout()
  print('Save plot into file %s' % image)
  plt.savefig(image, format='png', dpi=200)
  plt.close()
  
def read_butterfly(butterfly_file):
  if not os.path.isfile(butterfly_file):
    print('Cannot open the butterfly file. Skipping.')
    return None, None, None, None
  dNdE_bottom = 1e-30
  # F = E^2 dN/dE 
  E_MeV, dNdE_MeV, dNdEmin_MeV, dNdEmax_MeV = np.loadtxt(butterfly_file).transpose()
  E = E_MeV * u.Unit('MeV')
  # non-detection may return crazy numbers in ctbutterfly at higher energies
  # this is to prevent these numbers from screwing up the plot 
  F = np.clip(dNdE_MeV, dNdE_bottom, None) * u.Unit('MeV-1 cm-2 s-1') * E**2
  Fmin = np.clip(dNdEmin_MeV, dNdE_bottom, None) * u.Unit('MeV-1 cm-2 s-1') * E**2
  Fmax = np.clip(dNdEmax_MeV, dNdE_bottom, None) * u.Unit('MeV-1 cm-2 s-1') * E**2
  return E, F, Fmin, Fmax

def read_spec(spec_file):
  if not os.path.isfile(spec_file):
    print('Cannot open the binned spectrum file. Skipping.')
    return None, None, None, None, None, None, None
  spec_data = pf.open(spec_file)[1].data
  #E = spec_data['Energy'] * u.Unit(spec_data.columns['Energy'].unit)
  E = spec_data['Energy'] * u.Unit('TeV')
  dE_d = spec_data['ed_Energy'] * u.Unit('TeV')
  dE_u = spec_data['eu_Energy'] * u.Unit('TeV')
  #F_det = spec_data['Flux'] * u.Unit(spec_data.columns['Flux'].unit)
  F_det = spec_data['Flux'] * u.Unit('erg cm-2 s-1')
  Ferr = spec_data['e_Flux'] * u.Unit('erg cm-2 s-1')
  F_ul = spec_data['UpperLimit'] * u.Unit('erg cm-2 s-1')
  TS = spec_data['TS']
  return E, dE_d, dE_u, F_det, Ferr, F_ul, TS

def select_detections(TS, F_det, F_ul, Z_min=3.5):
  TS_min = st.chi2.isf(2 * st.norm.sf(Z_min), 2)
  F_unit = F_det.unit
  # where seems to get rid of the units...
  F = np.where(TS < TS_min, F_ul.to(F_unit), F_det) #* F_unit
  ul_mask = np.where(TS < TS_min, 1, 0)
  return ul_mask, F

# parse the XML file with the sky model
def get_target(sky_model, target_name):
  if not os.path.isfile(sky_model):
    print('Cannot open model file. Skipping.')
    return None
  tree = ET.parse(sky_model)
  src_lib = tree.getroot()
  for src in src_lib.iter('source'):
    if src.attrib.get('name') == target_name:
      return src
  print('Cannot find source in model file. Skipping.')
  return None

def get_dNdE(src):
  XML_spectrum = src[0] # TODO not really...
  # parse the spectral section
  spec_model = str(XML_spectrum.attrib.get('type'))
  print('Found a %s spectral model' % spec_model)
  if spec_model == 'FileFunction':
    return get_dNdE_from_file(str(XML_spectrum.attrib.get('file')))
  if spec_model == 'PowerLaw':
    dNdE = get_PowerLaw_model(XML_spectrum)
  elif spec_model == 'PowerLaw2':
    dNdE = get_PowerLaw2_model(XML_spectrum)
  elif spec_model == 'ExponentialCutoffPowerLaw':
    dNdE = get_ExpCutoff_model(XML_spectrum)
  elif spec_model == 'LogParabola':
    dNdE = get_LogParabola_model(XML_spectrum)
  elif spec_model == 'BrokenPowerLaw':
    dNdE = get_BrokenPowerLaw_model(XML_spectrum)
  else:
    print('Spectral model %s not supported - skipping' % spec_model)
    return None, None
  E_TeV = np.linspace(Emin.to('TeV')/u.Unit('TeV'), Emax.to('TeV')/u.Unit('TeV'), 100)
  return E_TeV * u.Unit('TeV'), dNdE(E_TeV) * u.Unit('cm-2 s-1 TeV-1')

def get_dNdE_from_file(dNdE_file):
  print('Read spectrum from file %s' % (dNdE_file))
  if not os.path.isfile(dNdE_file):
    print('Cannot open FileFunction file. Skipping.')
    return None, None
  E_f_MeV, dNdE_f_MeV = np.loadtxt(dNdE_file).transpose()
  E_f = E_f_MeV * u.Unit('MeV')
  dNdE_f = dNdE_f_MeV * u.Unit('cm-2 s-1 MeV-1')
  # narrow the energy range
  ok = np.where((E_f>Emin) * (E_f<Emax))
  return E_f[ok].to('TeV'), dNdE_f[ok].to('cm-2 s-1 TeV-1')



def get_PointSource_model(XML_spectrum):
    # CTools definition:
    # dN/dE = Prefactor * (E/Scale)^Index
    # [Prefactor] = ph * cm^-2 * s^-1 * MeV^-1
    # [Scale] = MeV
    # Astropy definition:
    # f(x) = amplitude * (x/x_0)^(-1. * alpha)
    # [amplitude] = ph * cm^-2 * s^-1 * TeV^-1
    # [x_0] = TeV
    par_array = [0.] * 2
    for idx, item in enumerate(XML_spectrum[:]):
        par_name = item.attrib.get('name')
        if par_name == 'RA':
            par_array[0] = item.attrib.get('value')
        elif par_name == 'DEC' :
            par_array[1] = item.attrib.get('value')
    print(f"Parameters: RA={par_array[0]}, DEC={par_array[1]}")
    return par_array


def get_PointSource_model_error(XML_spectrum):
    # CTools definition:
    # dN/dE = Prefactor * (E/Scale)^Index
    # [Prefactor] = ph * cm^-2 * s^-1 * MeV^-1
    # [Scale] = MeV
    # Astropy definition:
    # f(x) = amplitude * (x/x_0)^(-1. * alpha)
    # [amplitude] = ph * cm^-2 * s^-1 * TeV^-1
    # [x_0] = TeV
    par_array = [0.] * 4
    for idx, item in enumerate(XML_spectrum[:]):
        par_name = item.attrib.get('name')
        if par_name == 'RA':
            par_array[0] = item.attrib.get('value')
            par_array[2] = item.attrib.get('error')
        elif par_name == 'DEC' :
            par_array[1] = item.attrib.get('value')
            par_array[3] = item.attrib.get('error')
    print(f"Parameters: RA={par_array[0]}, DEC={par_array[1]}")
    return par_array

# Library of functions:
# get_dNdE = get_MODELNAME_model(XML_spectrum_section)
# where MODEL is one of:
#   PowerLaw
#   PowerLaw2
#   ExpCutoff
#   LogParabola
#   BrokenPowerLaw
# return a pointer to a function (callable) that describes
# the spectral shape of a source in an XML file.
# Mind the unit definitions in the CTools XML interface
# and in the astropy.modeling library as the energy is
# passed in TeV units:
# http://cta.irap.omp.eu/ctools/user_manual/getting_started/models.html
# http://docs.astropy.org/en/stable/modeling/index.html

def get_PowerLaw_model(XML_spectrum):
  # CTools definition:
  # dN/dE = Prefactor * (E/Scale)^Index
  # [Prefactor] = ph * cm^-2 * s^-1 * MeV^-1
  # [Scale] = MeV
  # Astropy definition:
  # f(x) = amplitude * (x/x_0)^(-1. * alpha)
  # [amplitude] = ph * cm^-2 * s^-1 * TeV^-1
  # [x_0] = TeV
  par_array = [0.] * 3
  for idx, item in enumerate(XML_spectrum[:]):
    XML_value = float(item.attrib.get('value')) * float(item.attrib.get('scale'))
    par_name = item.attrib.get('name')
    if par_name == 'Prefactor':
      par_array[0] = XML_value * 1.e6
    elif par_name == 'Scale' or par_name == 'PivotEnergy':
      par_array[1] = XML_value / 1.e6
    elif par_name == 'Index':
      par_array[2] = -1. * XML_value
  print('Parameters: Prefactor=%g, Scale=%e, Index=%f' % \
      (par_array[0], par_array[1], par_array[2]))
  spec_model_function = getattr(models, 'PowerLaw1D')
  return spec_model_function(par_array[0], par_array[1], par_array[2])

def get_PowerLaw_model_error(XML_spectrum):
    # CTools definition:
    # dN/dE = Prefactor * (E/Scale)^Index
    # [Prefactor] = ph * cm^-2 * s^-1 * MeV^-1
    # [Scale] = MeV
    # Astropy definition:
    # f(x) = amplitude * (x/x_0)^(-1. * alpha)
    # [amplitude] = ph * cm^-2 * s^-1 * TeV^-1
    # [x_0] = TeV
    par_array = [0.] * 4
    for idx, item in enumerate(XML_spectrum[:]):
        XML_value = float(item.attrib.get('value')) * float(item.attrib.get('scale'))
#        XML_value_error = float(item.attrib.get('error')) * float(item.attrib.get('scale'))
        par_name = item.attrib.get('name')
        if par_name == 'Prefactor':
            par_array[0] = XML_value * 1.e6
            par_array[2] = float(item.attrib.get('error')) * float(item.attrib.get('scale')) * 1.e6
        elif par_name == 'Index':
            par_array[1] = -1. * XML_value
            par_array[3] = -1. * float(item.attrib.get('error')) * float(item.attrib.get('scale'))
    return par_array

def get_PowerLaw2_model(XML_spectrum):
  # CTools definition:
  # dN/dE = Integral * (Index+1) * E^Index / (UpperLimit^(Index+1) - LowerLimit^(Index+1))
  # [Integral] = ph * cm^-2 * s^-1
  # [UpperLimit] = MeV
  # [LowerLimit] = MeV
  # Astropy definition:
  # f(x) = amplitude * (x/x_0)^(-1. * alpha)
  # [amplitude] = ph * cm^-2 * s^-1 * TeV^-1
  # [x_0] = TeV
  for idx, item in enumerate(XML_spectrum[:]):
    XML_value = float(item.attrib.get('value')) * float(item.attrib.get('scale'))
    par_name = item.attrib.get('name')
    if par_name == 'Integral':
      integral = XML_value
    elif par_name == 'UpperLimit':
      X_max = XML_value / 1.e6
    elif par_name == 'LowerLimit':
      X_min = XML_value / 1.e6
    elif par_name == 'Index':
      alpha = XML_value
  if alpha == -1:
    return None
  amplitude = integral * (1.+alpha) / (X_max**(1.+alpha) - X_min**(1.+alpha))
  par_array = [amplitude, 1., -1. * alpha]
  print('Parameters: Prefactor=%g, Scale=%e, Index=%f' % \
      (par_array[0], par_array[1], par_array[2]))
  spec_model_function = getattr(models, 'PowerLaw1D')
  return spec_model_function(par_array[0], par_array[1], par_array[2])

def get_ExpCutoff_model(XML_spectrum):
  # CTools definition:
  # dN/dE = Prefactor * (E/Scale)^Index * exp(-E/Cutoff)
  # [Prefactor] = ph * cm^-2 * s^-1 * MeV^-1
  # [Scale] = MeV
  # [Cutoff] = MeV
  # Astropy definition:
  # f(x) = amplitude * (x/x_0)^(-1. * alpha) * exp(-x/x_cutoff)
  # [amplitude] = ph * cm^-2 * s^-1 * TeV^-1
  # [x_0] = TeV
  # [x_cutoff] = TeV
  par_array = [0.] * 4
  for idx, item in enumerate(XML_spectrum[:]):
    XML_value = float(item.attrib.get('value')) * float(item.attrib.get('scale'))
    par_name = item.attrib.get('name')
    if par_name == 'Prefactor':
      par_array[0] = XML_value * 1.e6
    elif par_name == 'PivotEnergy' or par_name == 'PivotEnergy':
      par_array[1] = XML_value / 1.e6
    elif par_name == 'Index':
      par_array[2] = -1. * XML_value
    elif par_name == 'CutoffEnergy':
      par_array[3] = XML_value / 1.e6
  print('Parameters: Prefactor=%g, PivotEnergy=%e, Index=%f, CutoffEnergy=%e' % \
      (par_array[0], par_array[1], par_array[2], par_array[3]))
  spec_model_function = getattr(models, 'ExponentialCutoffPowerLaw1D')
  return spec_model_function(par_array[0], par_array[1], par_array[2], par_array[3])

def get_LogParabola_model(XML_spectrum):
  # CTools definition:
  # dN/dE = Prefactor * (E/Scale)^(Index + Curvature * ln(E/Scale))
  # [Prefactor] = ph * cm^-2 * s^-1 * MeV^-1
  # [Scale] = MeV
  # Astropy definition:
  # f(x) = amplitude * (x/x_0)^(-1. * alpha - beta * ln(x/x_0))
  # [amplitude] = ph * cm^-2 * s^-1 * TeV^-1
  # [x_0] = TeV
  par_array = [0.] * 4
  for idx, item in enumerate(XML_spectrum[:]):
    XML_value = float(item.attrib.get('value')) * float(item.attrib.get('scale'))
    par_name = item.attrib.get('name')
    if par_name == 'Prefactor':
      par_array[0] = XML_value * 1.e6
    elif par_name == 'Scale' or par_name == 'PivotEnergy':
      par_array[1] = XML_value / 1.e6
    elif par_name == 'Index':
      par_array[2] = -1. * XML_value
    elif par_name == 'Curvature':
      par_array[3] = -1. * XML_value
  print('Parameters: Prefactor=%g, Scale=%e, Index=%f, Beta=%f' %  \
      (par_array[0], par_array[1], par_array[2], par_array[3]))
  spec_model_function = getattr(models, 'LogParabola1D')
  return spec_model_function(par_array[0], par_array[1], par_array[2], par_array[3])

def get_BrokenPowerLaw_model(XML_spectrum):
  # CTools definition:
  # dN/dE = Prefactor * (E/BreakValue)^Index1  (if E<BreakValue)
  # dN/dE = Prefactor * (E/BreakValue)^Index2  (if E>=BreakValue)
  # [Prefactor] = ph * cm^-2 * s^-1 * MeV^-1
  # [BreakValue] = MeV
  # Astropy definition:
  # f(x) = amplitude * (x/x_break)^(-1. * alpha1)  (if x<x_break)
  # f(x) = amplitude * (x/x_break)^(-1. * alpha2)  (if x>=x_break)
  # [amplitude] = ph * cm^-2 * s^-1 * TeV^-1
  # [x_0] = TeV
  par_array = [0.] * 4
  for idx, item in enumerate(XML_spectrum[:]):
    XML_value = float(item.attrib.get('value')) * float(item.attrib.get('scale'))
    par_name = item.attrib.get('name')
    if par_name == 'Prefactor':
      par_array[0] = XML_value * 1.e6
    elif par_name == 'BreakValue' or par_name == 'Scale':
      par_array[1] = XML_value / 1.e6
    elif par_name == 'Index1':
      par_array[2] = -1. * XML_value
    elif par_name == 'Index2':
      par_array[3] = -1. * XML_value
  print('Parameters: Prefactor=%g, BreakValue=%e, Index1=%f, Index2=%f ' % \
      (par_array[0], par_array[1], par_array[2], par_array[3]))
  spec_model_function = getattr(models, 'BrokenPowerLaw1D')
  return spec_model_function(par_array[0], par_array[1], par_array[2], par_array[3])

def my_clean_format(x, p):
  l10_x = int(np.floor(np.log10(x)))
  if l10_x >= 0.:
    return '%d' % x
  fmt = '%%.%df' % (-1 * l10_x)
  return fmt % x
  
#TODO add checks to the input parameters
def validate_options(options):
  if options.outfile:
    image = options.outfile
  else:
    image='SED_%s_v1.png' % options.name
  return options.spec, options.butterfly, options.model, image

def parse_command_line():
  """parse_command_line()"""
  """Read the arguments passed on the command line"""
  """to the script. """
  Parser = argparse.ArgumentParser(
      description='Plot the spectrum of a source from an ASTRI obs')
  Parser.add_argument('--butterfly', '-b', \
      default='butterfly.txt', help='ASCII butterfly file (output of ctbutterfly)')
  Parser.add_argument('--spec', '-s', \
      default='spec.fits', help='FITS binned spectrum (output of csspec)')
  Parser.add_argument('--threshold', '-t', type=float, \
      default=3.5, help='detection threshold in the FITS binned spectrum (default 3.5 sigma)')
  Parser.add_argument('--model', '-m', \
      default='model.xml', help='XML model file')
  Parser.add_argument('--name', '-n', \
      default='source', help='source name in the XML model file')
  Parser.add_argument('--outfile', '-o', \
      default=None, help='output PNG image file name (default SED_$NAME$_v1.png)')
  return Parser.parse_args()

if __name__ == '__main__':
  main()
