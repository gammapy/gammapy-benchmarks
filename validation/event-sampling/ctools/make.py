from pathlib import Path
import logging
import warnings
import click
import multiprocessing
from itertools import repeat
import xml.etree.ElementTree as ET

import Plot_SED
import gammalib, ctools, cscripts
from astropy.coordinates import SkyCoord
from astropy.table import Table
import sys, os
import numpy as np
import astropy.units as u
from astropy.modeling import models
import matplotlib.pyplot as plt

###
# export CALDB="$GAMMAPY_DATA/cta-1dc/caldb"
#

log = logging.getLogger(__name__)

# path config
BASE_PATH = Path("../make.py").parent

AVAILABLE_MODELS = ["point-pwl", "point-ecpl", "point-log-parabola",
                    "point-pwl2", "point-ecpl-3fgl", "point-ecpl-4fgl",
                    "point-template", "diffuse-cube",
                    "disk-pwl", "gauss-pwl",
                    "gauss-pwlsimple", "point-pwlsimple", "disk-pwlsimple",
                    "point-pwltest", "point-pwl-time"]

DPI=600
#########
# Set the configuration parameters

E_MIN = 0.3 #TeV
E_MAX = 100 #TeV
N_BIN = 50
N_PIX = 50
BINSZ = 0.02 #deg
CALDB = '1dc'
IRF = 'South_z20_50h'
COORDSYS = 'CEL'
PROJ = 'CAR'

#######

#log.info("Starting...")


@click.group()
@click.option(
              "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"])
              )
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    
    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("all", help="Run all steps")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS)))
@click.option(
              "--obs_ids", default=1, nargs=1, help="Select a single observation", type=int
              )
def all_cmd(model, obs_ids):
    xmlfile = BASE_PATH / f"ctools/models/{model}.xml"
    inmodels = gammalib.GModels(str(xmlfile))
    log.info(f"Reading {xmlfile}")

    mod = inmodels[0]
    name=mod.name()

    ra = mod.spatial().region().ra()
    dec = mod.spatial().region().dec()

    evt = BASE_PATH / f"data/models/{model}/"

    for obsid in np.arange(obs_ids):

        event_file = evt / f"events_1h_{obsid:04d}.fits.gz"
        log.info(f"Reading {event_file}")
        outmodel = BASE_PATH / f"ctools/results/models/{model}/results_{obsid:04d}.xml"
        log.info(f"Reading {outmodel}")

        log.info(f"Analysing data...")
        stackedPipeline(name=name, obsfile=str(event_file),
                        l=ra, b=dec,
                        emin=E_MIN, emax=E_MAX, enumbins=N_BIN,
                        nxpix=N_PIX, nypix=N_PIX, binsz=BINSZ,
                        coordsys=COORDSYS, proj=PROJ, caldb=CALDB, irf=IRF,debug=False,
                        inmodel=xmlfile, outmodel=outmodel)

        log.info(f"Plot results...")
        os.system(f"python Plot_SED.py -m bkg_cube.xml -n {model} -s {str(outmodel.parent)}/{str(outmodel.stem)}.fits -b {str(outmodel.parent)}/{str(outmodel.stem)}.txt -o {str(outmodel.parent)}/{str(outmodel.stem)}.png")



@cli.command("Plot_distrib", help="Make pull-distributions")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS)))
@click.option(
              "--obs_ids", default=1, nargs=1, help="Select a single observation", type=int
              )
def plot_pull_distrib(model,obs_ids):
    
    sim_mod = BASE_PATH / f"ctools/models/{model}.xml"

    source = Plot_SED.get_target(sim_mod, model)
    spec_model = source[0].attrib.get('type')
    spat_model = source[1].attrib.get('type')
    if spec_model == 'PowerLaw':
        spec_param = Plot_SED.get_PowerLaw_model(source[0])
        tab_spec = Table()
        tab_spec["amplitude"] = [spec_param.amplitude.value]
        tab_spec["index"] = [spec_param.alpha.value]
        ampl = []
        e_ampl = []
        index = []
        e_index = []
    if spat_model == 'PointSource':
        spat_param = Plot_SED.get_PointSource_model(source[1])
        tab_spat = Table()
        tab_spat['ra'] = [float(spat_param[0])]
        tab_spat['dec'] = [float(spat_param[1])]
        ra = []
        e_ra = []
        dec = []
        e_dec = []

    for obsid in np.arange(obs_ids):
        
        fit_mod = BASE_PATH / f"ctools/results/models/{model}/results_{obsid:04d}.xml"
        source_fit = Plot_SED.get_target(fit_mod, model)
        spec_fit_model = source_fit[0].attrib.get('type')
        spat_fit_model = source_fit[1].attrib.get('type')

        if spec_fit_model == 'PowerLaw':
            spec_fit_param = Plot_SED.get_PowerLaw_model_error(source_fit[0])
            ampl.append(float(spec_fit_param[0]))
            index.append(float(spec_fit_param[1]))
            e_ampl.append(float(spec_fit_param[2]))
            e_index.append(float(spec_fit_param[3]))

        if spat_fit_model == 'PointSource':
            spat_fit_param = Plot_SED.get_PointSource_model_error(source_fit[1])
            ra.append(float(spat_fit_param[0]))
            dec.append(float(spat_fit_param[1]))
            e_ra.append(float(spat_fit_param[2]))
            e_dec.append(float(spat_fit_param[3]))


    if spec_fit_model == 'PowerLaw':
        tab = Table()
        tab['amplitude'] = ampl
        tab['e_amplitude'] = e_ampl
        tab['index'] = index
        tab['e_index'] = e_index
        names = [name for name in tab.colnames if "e_" not in name]
        for name in names:
            pull = (tab[name] - tab_spec[name]) / tab["e_"+name]
            plt.hist(pull, bins=21, normed=True, range=(-5,5))
            plt.xlim(-5, 5)
            plt.xlabel("(value - value_true) / error")
            plt.ylabel("PDF")
            plt.title(f"Pull distribution for {model}: {name} ")
            filename = f"ctools/results/models/{model}/pull-distribution-{name}.png"
            save_figure(filename)

    if spat_fit_model == 'PointSource':
        tab = Table()
        tab['ra'] = ra
        tab['e_ra'] = e_ra
        tab['dec'] = dec
        tab['e_dec'] = e_dec
        names = [name for name in tab.colnames if "e_" not in name]
        for name in names:
            pull = (tab[name] - tab_spat[name]) / tab["e_"+name]
            plt.hist(pull, bins=21, normed=True, range=(-5,5))
            plt.xlim(-5, 5)
            plt.xlabel("(value - value_true) / error")
            plt.ylabel("PDF")
            plt.title(f"Pull distribution for {model}: amplitude ")
            filename = f"ctools/results/models/{model}/pull-distribution-{name}.png"
            save_figure(filename)


def stackedPipeline(name='Crab', obsfile='index.xml', l=0.01, b=0.01, emin=0.1, emax=100.0,
                    enumbins=20, nxpix=200, nypix=200, binsz=0.02,
                    coordsys='CEL', proj='CAR', caldb='prod2', irf='acdc1a',debug=False,inmodel='Crab', outmodel='results'):
    """
    Simulation and stacked analysis pipeline
    
    Parameters
    ----------
    obs : `~gammalib.GObservations`
    Observation container
    ra : float, optional
    Right Ascension of counts cube centre (deg)
    dec : float, optional
    Declination of Region of counts cube centre (deg)
    emin : float, optional
    Minimum energy (TeV)
    emax : float, optional
    Maximum energy (TeV)
    enumbins : int, optional
    Number of energy bins
    nxpix : int, optional
    Number of pixels in X axis
    nypix : int, optional
    Number of pixels in Y axis
    binsz : float, optional
    Pixel size (deg)
    coordsys : str, optional
    Coordinate system
    proj : str, optional
    Coordinate projection
    debug : bool, optional
    Debug function
    """
    
    # Bin events into counts map
    bin = ctools.ctbin()
    bin['inobs']= obsfile
    bin['ebinalg']  = 'LOG'
    bin['emin']     = emin
    bin['emax']     = emax
    bin['enumbins'] = enumbins
    bin['nxpix']    = nxpix
    bin['nypix']    = nypix
    bin['binsz']    = binsz
    bin['coordsys'] = coordsys
    bin['proj']     = proj
    bin['xref']     = l
    bin['yref']     = b
    bin['debug']    = debug
    bin['outobs'] = 'cntcube.fits'
    bin.execute()
    print('Datacube : done!')
    
    
    # Create exposure cube
    expcube = ctools.ctexpcube()
    #expcube['incube']=bin.obs()
    expcube['inobs']    = obsfile
    expcube['incube']   = 'NONE'
    expcube['ebinalg']  = 'LOG'
    expcube['caldb']    = caldb
    expcube['irf']      = irf
    expcube['emin']     = emin
    expcube['emax']     = emax
    expcube['enumbins'] = enumbins
    expcube['nxpix']    = nxpix
    expcube['nypix']    = nypix
    expcube['binsz']    = binsz
    expcube['coordsys'] = coordsys
    expcube['proj']     = proj
    expcube['xref']     = l
    expcube['yref']     = b
    expcube['debug']    = debug
    expcube['outcube']  = 'cube_exp.fits'
    expcube.execute()
    print('Expcube : done!')
    
    # Create PSF cube
    psfcube = ctools.ctpsfcube()
    psfcube['inobs']    = obsfile
    psfcube['incube']   = 'NONE'
    psfcube['ebinalg']  = 'LOG'
    psfcube['caldb']    = caldb
    psfcube['irf']      = irf
    psfcube['emin']     = emin
    psfcube['emax']     = emax
    psfcube['enumbins'] = enumbins
    psfcube['nxpix']    = 10
    psfcube['nypix']    = 10
    psfcube['binsz']    = 1.0
    psfcube['coordsys'] = coordsys
    psfcube['proj']     = proj
    psfcube['xref']     = l
    psfcube['yref']     = b
    psfcube['debug']    = debug
    psfcube['outcube']  = 'psf_cube.fits'
    psfcube.execute()
    print('Psfcube : done!')
    
    edispcube = ctools.ctedispcube()
    edispcube['inobs']    = obsfile
    edispcube['ebinalg']  = 'LOG'
    edispcube['incube']   = 'NONE'
    edispcube['caldb']    = caldb
    edispcube['irf']      = irf
    edispcube['xref']     = l
    edispcube['yref']     = b
    edispcube['proj']     = proj
    edispcube['coordsys'] = coordsys
    edispcube['binsz']    = 1.0
    edispcube['nxpix']    = 10
    edispcube['nypix']    = 10
    edispcube['emin']     = emin
    edispcube['emax']     = emax
    edispcube['enumbins'] = enumbins
    edispcube['outcube']  = 'edisp_cube.fits'
    edispcube['debug']    = debug
    edispcube.execute()
    print('Edispcube : done!')
    
    # Create background cube
    bkgcube = ctools.ctbkgcube()
    bkgcube['inobs']= obsfile
    bkgcube['incube']   = 'cntcube.fits'
    bkgcube['caldb']    = caldb
    bkgcube['irf']      = irf
    bkgcube['debug']    = debug
    bkgcube['inmodel']  = str(inmodel)
    bkgcube['outcube']  = 'bkg_cube.fits'
    bkgcube['outmodel'] = 'bkg_cube.xml'
    bkgcube.execute()
    print('Bkgcube : done!')
    
    #Fix the instrumental background parameters
    bkgcube.models()['BackgroundModel']['Prefactor'].fix()
    bkgcube.models()['BackgroundModel']['Index'].fix()
#
#    # Attach background model to observation container
    bin.obs().models(bkgcube.models())
#
#    
#    # Set Exposure and Psf cube for first CTA observation
#    # (ctbin will create an observation with a single container)
    bin.obs()[0].response(expcube.expcube(), psfcube.psfcube(), edispcube.edispcube(), bkgcube.bkgcube())

    # Perform maximum likelihood fitting
    like = ctools.ctlike(bin.obs())
#    like['inmodel']='bkg_cube.xml'
    like['edisp'] = True
#    like['edispcube'] = 'edisp_cube.fits'
#    like['expcube'] = 'cube_exp.fits'
#    like['psfcube'] = 'psf_cube.fits'
#    like['bkgcube'] = 'bkg_cube.fits'
    like['outmodel'] = str(outmodel)
    #    like['outcovmat']=inmodel+'_covmat.txt'
    like['debug'] = debug # Switch this always on for results in console
    #like['statistic']='CSTAT'
    like.execute()
    print('Likelihood : done!')
    
    # Set the best-fit models (from ctlike) for the counts cube
    bin.obs().models(like.obs().models())
    
    # Obtain the best-fit butterfly
    try:
        butterfly = ctools.ctbutterfly(bin.obs())
        butterfly['srcname'] = name
        butterfly['inmodel'] = str(outmodel)
        butterfly['edisp']   = True
        butterfly['emin']    = emin
        butterfly['emax']    = emax
        butterfly['outfile'] = str(outmodel.parent) +"/"+ str(outmodel.stem)+".txt"
        butterfly['debug']   = debug # Switch this always on for results in console
        #like['statistic']='CSTAT'
        butterfly.execute()
        print('Butterfly : done!')
    except:
        print("I COULDN'T CALCULATE THE BUTTERFLY....")
    
    # Extract the spectrum
    try:
        csspec = cscripts.csspec(bin.obs())
        csspec['srcname']  = name
        csspec['inmodel']  = str(outmodel)
        csspec['method']   = 'AUTO'
        csspec['ebinalg']  = 'LOG'
        csspec['emin']     = emin
        csspec['emax']     = emax
        csspec['enumbins'] = 10
        csspec['edisp']    = True
        csspec['outfile']  = str(outmodel.parent) +"/"+ str(outmodel.stem)+".fits"
        csspec['debug']    = debug # Switch this always on for results in console
        csspec.execute()
        print('Csspec : done!')
    except:
        print("I COULDN'T CALCULATE THE SPECTRUM....")
    
    # Return
    return


def save_figure(filename):
    path = BASE_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing {path}")
    plt.savefig(path, dpi=DPI)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    cli()



##pippolo='_test'
#pippolo=''
#
##for j in np.arange(3,20):
#for j in np.arange(1,19):
#    elem=str('{:02d}'.format(j))
#    
#    os.system('mkdir ' + elem)
#    #xmlfile='models.xml'
#    xmlfile=ACDC_MODEL_DIR+'/'+elem+'/mod_acdc1.1_'+elem+pippolo+'.xml'
#    
#    os.system('cp '+xmlfile+' ./'+elem)
#    
#    #outmodels=inmodels.copy()
#    
#    db = Table.read('db_par.txt', format='ascii')
#    db2 = dict(zip(db['spMod'],db['pars']))
#    
#    mod = open('./'+elem+'/mod_acdc1.1_'+elem+pippolo+'.xml', 'w')
#    xml=open(xmlfile)
#    qq=xml.readlines()
#    for q in qq:
#        q=q.replace('free="1"' , 'free="0"')
#            q=q.replace('name=' , 'tscalc="1" name=')
#                mod.write(q)

#mod.close()
#
#inmodels = gammalib.GModels('./'+elem+'/mod_acdc1.1_'+elem+pippolo+'.xml')
##    inmodels_double = gammalib.GModels('./'+elem+'/mod_acdc1.1_'+elem+'.xml')
#
#for mod in inmodels:
##      print(mod)
#name=mod.name()
##      if mod.spectral().type() == 'FileFunction':
##            continue
#
#if name == 'GCDiff':
#continue
#if (name != 'CTABackgroundModel') and (name != 'IEM') and (name != 'Fermi bubbles'):
#
#print
#print 'Facciamo', name
#
#if not mod.spatial().has_par('RA'):
#try:
#p=SkyCoord.from_name(name)
#print 'e che sta in ', p.galactic.data
#except:
##                print("Non ho trovato le coordinate...")
#print("Prendo le coordinate dalla mappa...")
#ra = mod.spatial().region().ra()
#dec = mod.spatial().region().dec()
##                continue
#else:
#p=SkyCoord( ra=mod.spatial()['RA'].value()  ,dec=mod.spatial()['DEC'].value() , unit='deg', frame='icrs' )
#print 'e che sta in ', p.galactic.data
#
##        mod.ts(1)
#
#
##xmlout='results_'+name+'.xml'
##outmodel=gammalib.GModels(xmlout)
##outmodel=gammalib.GModels(xmlfile)
##outmodels[name]=outmodel[name]
#
#
#spmodel = mod.spectral().type()
#print 'cha ha come modello spettrale una ',spmodel
#pars= db2[spmodel].split(',')
#print 'i cui parametri sono :',pars
#
#ctoolsfordummy.model_calculator(mod, e_pivot)  # ATTENZIONE, MODIFICA IL PREFACTOR MA NON IL PIVOT, CHE VIENE MODIFICATO NELLE RIGHE SUCCESSIVE
#for par in pars:
#mod.spectral()[par].free()
#if mod.spectral().has_par('Scale'):
##            mod.spectral()['Scale'].value(1e6)
##            ref_e = mod.spectral().pivot()
#mod.spectral()['Scale'].value(e_pivot*1e6)
#mod.spectral()['Scale'].fix()
#if mod.spectral().has_par('PivotEnergy'):
##              mod.spectral()['PivotEnergy'].value(1e6)
##              ref_e = mod.spectral().pivot()
#mod.spectral()['PivotEnergy'].value(e_pivot*1e6)
#mod.spectral()['PivotEnergy'].fix()
##          print(mod.spectral()[par])
##        flux_sim = mod.spectral().flux(ref_e *emin, ref_e*emax)
#
#inmodels.save('./'+elem+'/temp_'+str(name)+pippolo+'.xml')
##
##        print(inmodels)
#for par in pars:
#mod.spectral()[par].fix()
#
#
#ctoolsfordummy.stackedPipeline(name=name,obsfile=OBS_DIR+'/obslist_acdc1.1_'+elem+'.xml',
#l=p.galactic.l.value, b=p.galactic.b.value,
#emin=emin, emax=emax, enumbins=nbin,
#nxpix=npix, nypix=npix, binsz=binsz,
#coordsys=coordsys, proj=proj, caldb=caldb, irf=irf,debug=False,
#inmodel='./'+elem+'/temp_'+str(name)+pippolo)
#
#
#temp_inmodel='./'+elem+'/temp_'+str(name)+pippolo+'.xml'
#in_models = gammalib.GModels(temp_inmodel)
#in_root = in_models[name]
#in_spec = in_root.spectral()
#in_spat = in_root.spatial()
#
#ctlike_results='./'+elem+'/temp_'+str(name)+pippolo+'_results.xml'
#res_models = gammalib.GModels(ctlike_results)
#root = res_models[name]
#orig = root.spectral()
#orig_sp = root.spatial()
#
#cts_sim = ctoolsfordummy.count_sim_calculator('/Users/fabio/LAVORO/CTA/ACDC_ASTRI/ACDC1.1/ANALYSIS_PIVOT_3TeV/list_src.txt', name, '/Users/fabio/LAVORO/CTA/ACDC_ASTRI/ACDC1.1/ANALYSIS_PIVOT_3TeV/'+str(elem)+'/obs-index.fits.gz', elem, emin, emax)
#
#if orig.type() == 'LogParabola':
#function_sim = models.LogParabola(str(in_spec.prefactor() * 1e6) + 'cm-2 s-1 TeV-1', str(in_spec.pivot()), in_spec.index()*-1., in_spec.curvature()*-1.)
#function_fit = models.LogParabola(str(orig.prefactor()*1e6) + 'cm-2 s-1 TeV-1', str(orig.pivot()), orig.index()*-1., orig.curvature()*-1.)
#
#if orig.type() == 'PowerLaw':
#function_sim = models.PowerLaw(in_spec.index()*-1., str(in_spec.prefactor()*1e6) + ' cm-2 s-1 TeV-1', str(in_spec.pivot()))
#function_fit = models.PowerLaw(orig.index()*-1., str(orig.prefactor()*1e6) + ' cm-2 s-1 TeV-1', str(orig.pivot()))
#
#if (orig.type() == 'ExpCutoff') | (orig.type() == 'ExponentialCutoffPowerLaw'):
#function_sim = models.ExponentialCutoffPowerLaw(in_spec.index()*-1., str(in_spec.prefactor()*1e6) + ' cm-2 s-1 TeV-1', str(in_spec.pivot()), str(1./in_spec.cutoff().TeV()) + 'TeV-1')
#function_fit = models.ExponentialCutoffPowerLaw(orig.index()*-1., str(orig.prefactor()*1e6) + ' cm-2 s-1 TeV-1', str(orig.pivot()), str(1./orig.cutoff().TeV()) + 'TeV-1')
#
#
#if (orig.type() != 'FileFunction') and (orig.type() != 'ConstantValue'):
#flux_sim = ctoolsfordummy.flux_calculation(emin, emax, function_sim)
#flux_fit = ctoolsfordummy.flux_calculation(emin, emax, function_fit)
#flux_error = ctoolsfordummy.random_flux_calculation(root, emin, emax, function_fit, flux_fit)
#
#in_models = gammalib.GModels(temp_inmodel)
#in_root = in_models[name]
#in_spec = in_root.spectral()
#in_spat = in_root.spatial()
#res_models = gammalib.GModels(ctlike_results)
#root = res_models[name]
#orig = root.spectral()
#orig_sp = root.spatial()
#
#fd = open('./SRCs_best-fit.txt','a')
#scrivi = str("field_"+ elem) + " " + str(name.replace(' ','_')) + " " + str(orig.type()) + " " + str(in_spec[1].value()) + " " + str(orig[1].value()) + " " + str(orig[1].error()) + " " + str(in_spec[0].value()*1e6) + " " + str(orig[0].value()*1e6) + " " + str(orig[0].error()*1e6) + " " + str(in_spec.pivot()).replace(' ','_') + " " + str(orig.pivot()).replace(' ','_') + " " + str(in_spec[2].value()) + " " + str(orig[2].value()) + " " + str(orig[2].error()) + " " + str(root.ts()) + " " + str(flux_sim) + " " + str(flux_fit) + " " + str(flux_error) + " " + str(root.type()) + " " + str(orig_sp.type()) + " " + str(orig_sp.region().radius()) + " " + str(cts_sim)
##            scrivi = str("field_"+ elem) + " " + str(name.replace(' ','_')) + " " + str(orig.type()) + " " + str(mod.spectral().index()) + " " + str(orig.index()) + " " + str(orig[1].error()) + " " + str(mod.spectral()[0].value()*1e6) + " " + str(orig[0].value()*1e6) + " " + str(orig[0].error()*1e6) + " " + str(mod.spectral().pivot()) + " " + str(orig.pivot()) + " " + str(mod.spectral()[2].value()) + " " + str(orig[2].value()) + " " + str(orig[2].error()) + " " + str(root.ts()) + " " + str(flux_sim) + " " + str(flux_fit) + " " + str(flux_error) + " " + str(root.type()) + " " + str(orig_sp.type()) + " " + str(orig_sp.region().radius()) + " " + str(cts_sim)
#fd.write(scrivi +"\n")
#fd.close()
#
#else:
#fd = open('./SRCs_best-fit.txt','a')
#scrivi = str("field_"+ elem) + " " + str(name.replace(' ','_')) + " " + str(orig.type()) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(0.0) + " " + str(root.ts()) + " " + str(1.0) + " " + str(orig[0].value()) + " " + str(orig[0].error()) + " " + str(root.type()) + " " + str(orig_sp.type()) + " " + str(orig_sp.region().radius()) + " " + str(cts_sim)
#fd.write(scrivi +"\n")
#fd.close()
#
#os.system('mv ./'+str(elem)+'/temp_'+str(name.replace(' ','\ '))+'.xml ./'+str(elem)+'/temp_'+str(name.replace(' ',''))+'.xml')
#os.system('mv ./'+str(elem)+'/temp_'+str(name.replace(' ','\ '))+'_spec.fits ./'+str(elem)+'/temp_'+str(name.replace(' ',''))+'_spec.fits')
#os.system('mv ./'+str(elem)+'/temp_'+str(name.replace(' ','\ '))+'_results.xml ./'+str(elem)+'/temp_'+str(name.replace(' ',''))+'_results.xml')
#os.system('mv ./'+str(elem)+'/temp_'+str(name.replace(' ','\ '))+'_butterfly.txt ./'+str(elem)+'/temp_'+str(name.replace(' ',''))+'_butterfly.txt')
##        os.system('mv ./'+str(elem)+'/temp_'+str(name.replace(' ','\ '))+'_covmat.txt ./'+str(elem)+'/temp_'+str(name.replace(' ',''))+'_covmat.txt')
#
#ctoolsfordummy.plotter_for_paper(elem,name)
#
#



