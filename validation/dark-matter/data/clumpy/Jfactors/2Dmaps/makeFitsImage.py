"""@package docstring
\file makeFitsImage.py \brief Convert 2D skymaps in the HEALPix format to FITS images

More details (documentation to be implemented).
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################
#    Authors:
#    This file belongs to the Clumpy code, http://lpsc.in2p3.fr/clumpy
#    For any questions, please contact moritz.huetten@desy.de
###############################################################################

# Minimal example Python script for transforming FITS files created by Clumpy 
# with the -o2 option) into FITS images according to the WCS standard.
#
# For a more complete introduction into healpy and its plotting features, please
# have a look at the nice documentation at http://healpy.readthedocs.org
#
# You need to have installed 
#   - matplotlib
#   - numpy 
#   - healpy 
#   - astropy.io
# to run this script.

# example for usage:
# 0.) ./bin/clumpy -g7 -D
# 1.) ./bin/clumpy -g7 -i clumpy_params_g7.txt
# 2.) ./bin/clumpy -o2 -i output/annihil_gal2D_LOS180,0_FOV4x4_rse1_alphaint0.10deg_nside1024.fits 1 1
# 3.) python python_helperscripts/makeFitsImage.py --infile output/annihil_gal2D_LOS180,0_FOV4x4_rse1_alphaint0.10deg_nside1024-JFACTOR-Jtot.fits
# if you hace healpy version >= 1.9.0, directly do
# 2.) python python_helperscripts/makeFitsImage.py --infile output/annihil_gal2D_LOS180,0_FOV4x4_rse1_alphaint0.10deg_nside1024.fits --extension 1 --column 1

###############################################################################
#    import modules:
import sys
import warnings
import healpy as hp # warning: due to a bug in healpy, importing it before pylab can cause a segmentation fault in some circumstances.
hp_version = float(hp.__version__[:3])
if (hp_version > 1.2 and hp_version < 1.9):
        warnings.warn('WARNING: Your healpy version is older than 1.9.0.\n Reading part-sky map in/or explicit indexing will not work.\n')
import numpy as np
import astropy
astropy_version = float(astropy.__version__[0])
from astropy.io import fits
import reproject as rp

###############################################################################
##    main part: 
###############################################################################

help_message = \
' Minimal example Python script for transforming FITS files created by Clumpy\n\
 with the -o2 option) into FITS images according to the WCS standard.\n\
 For a more complete introduction into healpy and its plotting features, please\n\
 have a look at the nice documentation at http://healpy.readthedocs.org\n\n\
 You need to have installed\n\
   - matplotlib\n\
   - numpy\n\
   - healpy\n\
   - astropy.io\n\
 to run this script.\n\n\
 example for usage:\n\
 0.) ./bin/clumpy -g7 -D  \n\
 1.) ./bin/clumpy -g7 --infile clumpy_params_g7.txt \n\
 2.) ./bin/clumpy -o2 --infile output/annihil_gal2D_LOS180,0_FOV4x4_rse1_alphaint0.10deg_nside1024.fits 1 1 \n\
 3.) python python_helperscripts/makeFitsImage.py --infile output/annihil_gal2D_LOS180,0_FOV4x4_rse1_alphaint0.10deg_nside1024-JFACTOR-Jtot.fits \n\
 if you hace healpy version >= 1.9.0, directly do \n\
 2.) python python_helperscripts/makeFitsImage.py --infile output/annihil_gal2D_LOS180,0_FOV4x4_rse1_alphaint0.10deg_nside1024.fits --extension 1 --column 1 \n\n\
 With the optional input --coordsys or -s you can fix the output coordinate system G,C, or E (galactic, celestial, or ecliptic).\n'

def main(argv):
    
    ###########################################################################
    #    import modules:
    import os
    import getopt
    
    ###########################################################################
    #  read input variables:
    
    infile = 'empty'
    i_ext  = -1
    i_col  = -1
    coordsys_out = '-1'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:e:c:s:',['infile=','extension=','column=','coordsys='])
    except getopt.GetoptError:
        print('Wrong input. The input options are:')
        print('-i or --infile for reading the input FITS file')
        print('-e or --extension for specifying the FITS extension')
        print('-c or --column for specifying the FITS column')
        print('-s or --coordsys for specifying the target (output) coordinate system')
        print('-h for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # help option
            print(help_message)
            sys.exit()
        elif opt in ('-i', '--infile'):
            infile = str(arg)
        elif opt in ('-e', '--extension'):
            i_ext = int(arg)
        elif opt in ('-c', '--column'):
            i_col = int(arg)
        elif opt in ('-s', '--coordsys'):
            coordsys_out = arg
    
    if infile == 'empty':
        print(' Please parse an input FITS file with -i file.fits or --infile file.fits\n\
 Use the -h option for further help.')
        sys.exit()
    
    convert(infile, i_ext, i_col, coordsys_out)
    
    ###########################################################################

def convert(infile='empty', i_ext=-1, i_col=-1, coordsys_out='-1'):
    """Main function of makeFitsImage.py
    
    Execute makeFitsImage.py --help for more details (it is not yet been fully implemented to be loaded within Python, also it should work already).
    """
    
    warnings.warn('Attention: Conversion from HEALPix format to FITS image causes degradation.\n All information in the FITS header corresponds to the original HEALPix data.')
    
    #  read header:
    hdulist_in = fits.open(infile)
    type = hdulist_in[1].header['TTYPE1']
    
    # detect type of input file
    if type[:5] == 'PIXEL':
        print('  ... detected Healpix part-sky map...')
        fullsky = False
        if (hp_version > 1.2 and hp_version < 1.9):
            raise IOError('Handling of part-sky FITS files not supported by healpy versions < 1.9.0.')
    else:
        try:
            scheme = hdulist_in[1].header['INDXSCHM']
            if scheme[:8] == 'IMPLICIT':
                print('  ... detected Healpix full-sky map, e.g. created by Clumpy with the -o2 option.')
                fullsky = True
            elif scheme[:8] == 'EXPLICIT':
                print('  ... detected Healpix part-sky map, e.g. created by Clumpy with the -o2 option.')
                fullsky = False
            else:
                raise IOError('FITS keyword "INDXSCHM" must be either "IMPLICIT" or "EXPLICIT".')
        except:  
            warnings.warn('Possibly wrong input file format. Must contain Healpix sky map.\n')
            fullsky = True
            i_ext = 1
            i_col = 1
    
    if i_ext == -1:
        warnings.warn('No input extension specified with --extension or -e. Set to first extension.')
        i_ext = 1
    if i_col == -1:
        warnings.warn('No input column specified with --column or -c. Set to first column.')
        i_col = 1
    
    input_header = hdulist_in[i_ext].header
    ext_name = input_header['EXTNAME']
    columns = hdulist_in[i_ext].columns
    if fullsky == False:
        column  = columns[i_col] # zeroth column is PIXEL
    else:
        column  = columns[i_col -1 ]
    
    try:
        map_center_psi_deg      = input_header['PSI_0']
    except:
        raise IOError('Necessary input information "PSI_0" missing.\n Note: This script is tailored to read Healpix maps created by Clumpy > 2015.06_corr3.')
    try:
        map_center_theta_deg    = input_header['THETA_0']
    except:
        raise IOError('Necessary input information "THETA_0" missing.\n Note: This script is tailored to read Healpix maps created by Clumpy > 2015.06_corr3.')
    try:
        map_diam_theta_deg      = input_header['SIZE_Y']
    except:
        raise IOError('Necessary input information "SIZE_Y" missing.\n Note: This script is tailored to read Healpix maps created by Clumpy > 2015.06_corr3.')
    try:
        map_diam_theta_orth_deg = input_header['SIZE_X']
    except:
        map_diam_theta_orth_deg = map_diam_theta_deg
    try:
        coordsys_in             = input_header['COORDSYS']
    except:
        raise IOError('Necessary input information "COORDSYS" missing.\n Note: This script is tailored to read Healpix maps created by Clumpy > 2015.06_corr3.')
    try:
        units               = input_header['TUNIT'+str(i_col + 1)]
    except:
        units               = 'not specified'
    try:
        fsky                = input_header['F_SKY']
    except:
        fsky                = 'not specified'
    try:
        map_mean            = input_header['MEAN']
    except:
        map_mean            = 'not specified'
    try:
        nside               = input_header['NSIDE']
    except:
        raise IOError('Necessary input information "NSIDE" missing.')
    try:
        dtype_hpx           = input_header['HPX_TYPE']
    except:
        raise IOError('Necessary input information "HPX_TYPE" missing.')
    
    # special treatment for clumpy halo mode files:
    map_center_psi_deg_out = map_center_psi_deg
    map_center_theta_deg_out = map_center_theta_deg
    try:
        simumode            = input_header['SIMUMODE']
        if (simumode[0] == 'h'):
            map_center_psi_deg = 0.
            map_center_theta_deg = 0.
    except:
        warnings.warn('Could not read "SIMUMODE" keyword. Possibly wrong sky area exported.')
        simumode = "-1"
    
    tbdata  = hdulist_in[i_ext].data
    npix_fov = len(tbdata[column.name])
    
    map_sum = tbdata[column.name].sum(dtype=np.float64)
    if map_mean == 'not specified':
        print('Calculate mean...')
        map_mean = tbdata[column.name].mean(dtype=np.float64)
    hdulist_in.close()
    
    #  read map:
    dtype_map = np.float64
    if dtype_hpx == 'FLOAT32': 
        dtype_map = np.float32
    elif dtype_hpx == 'FLOAT64': 
        dtype_map = np.float64
    else:
        raise IOError('Healpix datatype (keyword "HPX_TYPE" in input file header) must be FLOAT32 or FLOAT64.')

    print('Read input map...')
    if fullsky == True:
        map_in = hp.read_map(infile)
    else:
        map_in = hp.read_map(infile, partial=True, field=i_col-1, hdu=i_ext, dtype=dtype_map)
    
    if fsky == 'not specified':
        npix   = len(map_in)
        fsky   = float(npix_fov) / float(npix)
   
    if map_diam_theta_orth_deg < 342.85:
        map_diam_theta_orth_deg *= 1.05
    if map_diam_theta_deg < 171.42:
        map_diam_theta_deg *= 1.05
    
    pixelnr = hp.nside2npix(nside)
    resol_deg = hp.nside2resol(nside) * 180./np.pi
    # make sure that npix_x, npix_y are odd numbers:
    npix_x_check = int(map_diam_theta_orth_deg / resol_deg) * 2 + 1
    npix_y_check = int(map_diam_theta_deg / resol_deg) * 2 + 1
    if npix_x_check > 1999:
        npix_x_check = 1997
        warnings.warn('Maximum number of 1997 pixels in x-dimension is reached.')
    if npix_y_check > 1999:
        npix_y_check = 1997
        warnings.warn('Maximum number of 1997 pixels in y-dimension is reached.')

    
    flip='astro'
    ## flip map for odd behavior at the gal. anticenter:  
    #if abs(map_center_psi_deg) == 180.0 and map_center_theta_deg == 0:
    #    flip='geo'
    
    # transform to galactic coordinates:
    if coordsys_in[:5] == 'G':
        print('Detected galactic coordinate system')
        if coordsys_out == '-1' or coordsys_out == 'G':
            coord = 'G'
            coordsys_out = coord
        else:
            coord = ['G', coordsys_out]
        
    elif coordsys_in[:5] == 'C':
        print('Detected equatorial coordinate system')
        coord = ['C','G']
        if coordsys_out == '-1' or coordsys_out == 'C':
            coord = 'C'
            coordsys_out = coord
        else:
            coord = ['C', coordsys_out]
    elif coordsys_in[:5] == 'E':
        print('Detected ecliptic coordinate system')
        coord = ['E','G']
        if coordsys_out == '-1' or coordsys_out == 'E':
            coord = 'E'
            coordsys_out = coord
        else:
            coord = ['E', coordsys_out]
    
    else:
        raise IOError('Input coordinate system not recognized. Must be either G, C, or E.')
    
    if coordsys_out not in ['-1', 'G', 'C', 'E']:
        raise IOError('Output coordinate system not recognized. Must be either G, C, or E.')
    
    
    if len(coord) > 1:
        map_center_theta_deg, map_center_psi_deg = hp.Rotator(coord=coord, deg=False)(np.pi/2 - map_center_theta_deg / 180.*np.pi, map_center_psi_deg / 180.*np.pi)
        map_center_psi_deg *= 180./np.pi
        map_center_theta_deg = 90. - map_center_theta_deg*180./np.pi


   
        
    #  get projected map
    image_array = hp.cartview(map_in,
                              rot = [map_center_psi_deg,map_center_theta_deg], \
                              lonra = [- map_diam_theta_orth_deg/2, map_diam_theta_orth_deg/2], \
                              latra = [- map_diam_theta_deg/2, map_diam_theta_deg/2], \
                              xsize=npix_x_check, \
                              flip=flip, \
                              coord=coord, \
                              return_projected_map = True)


    

    
    npix_x = image_array.shape[1]
    npix_y = image_array.shape[0]
    if npix_x_check != npix_x:
        warnings.warn('Attention!  npix_x_check ='+str(npix_x_check)+'!= npix_x='+str(npix_x))
    if npix_y_check != npix_y:
        warnings.warn('Attention!  npix_y_check ='+str(npix_y_check)+'!= npix_y='+str(npix_y))
    
    x_center = (npix_x + 1) / 2
    y_center = (npix_y + 1) / 2
    
    delta_x = map_diam_theta_orth_deg / npix_x
    delta_y = map_diam_theta_deg / npix_y
    print('Resolution of map ( delta_x , delta_y ): ('+str(delta_x)+','+str(delta_y)+')')    
    
    # rotate map for odd behavior at the gal. poles:
    #if abs(map_center_theta_deg) == 90.0:
        #map_center_psi_deg = map_center_psi_deg + 180
    
    if map_center_psi_deg < 0:
            map_center_psi_deg += 360.
    
    map_out = np.zeros((npix_y,npix_x), dtype=dtype_map)
    map_out[:][:] = image_array[:][:]
    
    # replace 1e-40 values by HEALPIX blind value
    #map_out[map_out < 4e-40] = -1.6375e30
    map_out[map_out < 4e-40] = 0
    #map_out[map_out < 0] = 0
    # write out total J-Factor/flux in map:
    if 'sr^-1' in units:
        totalflux = map_mean * 4.*np.pi * fsky
    else:
        totalflux = map_sum
    
    # create fits header:
    if coordsys_out == 'G':
        coordlon = 'GLON-CAR'
        coordlat = 'GLAT-CAR'
    elif coordsys_out == 'C':
        coordlon = 'RA---CAR'
        coordlat = 'DEC--CAR'
    elif coordsys_out == 'E':
        coordlon = 'ELON-CAR'
        coordlat = 'ELAT-CAR' 

    #radecsys = 'J2000'
    #coordlon = 'RA---CAR'
    #coordlat = 'DEC--CAR'
    #map_center_psi_deg_out=0
    #map_center_theta_deg_out=0
    
    reference_header = input_header[41:]
    reference_header.update({'EXTNAME' : ('Skymap'), \
                             #'RADECSYS': (radecsys, 'reference coordinate system'), \
                             'CDELT1'  : (-delta_x, 'pixel size (approx.) in longitude-dir. [deg]'), \
                             'CDELT2'  : (delta_y, 'pixel size (approx.) in latitude-dir. [deg]'), \
                             'CRPIX1'  : (x_center, 'central pixel in longitude direction'), \
                             'CRPIX2'  : (y_center, 'central pixel in latitude direction'), \
                             'CRVAL1'  : (map_center_psi_deg_out, 'longitude coordinate of map center [deg]'), \
                             'CRVAL2'  : (map_center_theta_deg_out, 'latitude coordinate of map center [deg]'), \
                             'CTYPE1'  : (coordlon, 'longitude coord. system (cartesian projection)'), \
                             'CTYPE2'  : (coordlat, 'latitude coord. system (cartesian projection)'), \
                             'CUNIT1'  : ('deg', 'longitude axis unit'), \
                             'CUNIT2'  : ('deg', 'latitude axis unit'), \
                             'NAXIS'   : 2, \
                             'NAXIS1'  : (npix_x, 'number of pixels in longitude direction'), \
                             'NAXIS2'  : (npix_y, 'number of pixels in latitude direction'), \
                             'BUNIT'   : (units, 'pixel value unit'),\
                             'F_SKY'   : (fsky, 'fraction of sky covered by FOV'), \
                             'MEAN'    : (map_mean, 'mean value in FOV (same unit as BUNIT)'), \
                             'FLUX_TOT': (totalflux, 'total integrated J-Factor (flux) in FOV'), \
                             'NSIDE'   : (nside, 'HEALPix resolution of original image'), \
                             'AUTHOR'  : 'file created by Clumpy, makeFitsImage.py'})
    
    # write fits file:
    if len(columns) > 1:
        outfile = infile[:-5] + '-' + ext_name + '-' + column.name + '-image.fits'
    else:
        outfile = infile[:-5] + '-image.fits'
        
    if astropy_version < 2:
        fits.writeto(outfile, map_out, reference_header, clobber=True)
    else:
        fits.writeto(outfile, map_out, reference_header, overwrite=True)
    print( 'Output file written to: '+outfile)
    
if __name__ == '__main__':
    
    main(sys.argv[1:])

##    end of file    ##########################################################
############################################################################### 
