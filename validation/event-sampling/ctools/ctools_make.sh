export CALDB="$GAMMAPY_DATA/cta-1dc/caldb"

model=$1

hr=1

caldb=1dc
irf=South_z20_50h

emin=0.3
emax=100

ra=266.404996
dec=-28.936172

bin=10

for i in {0..99};
do

    i=$(printf "%04d" $i)

    ctbin inobs=../data/models/$model/events_"$hr"h_"$i".fits.gz coordsys=CEL proj=CAR xref="$ra" yref="$dec" binsz=0.02 nxpix=50 nypix=50 ebinalg=LOG emin="$emin" emax="$emax" enumbins=20 outobs=ctscube.fits
    echo "Ctbin done!"

    ctexpcube inobs=../data/models/$model/events_"$hr"h_"$i".fits.gz caldb="$caldb" irf="$irf" incube=NONE outcube=cube.fits coordsys=CEL proj=CAR xref="$ra" yref="$dec"  binsz=0.02 nxpix=50 nypix=50 ebinalg=LOG emin="$emin" emax="$emax" enumbins=20
    echo "Ctexpcube done!"

    ctpsfcube inobs=../data/models/$model/events_"$hr"h_"$i".fits.gz incube=NONE caldb="$caldb" irf="$irf" coordsys=CEL proj=CAR xref="$ra" yref="$dec" binsz=1.0 nxpix=10 nypix=10 ebinalg=LOG emin="$emin" emax="$emax" enumbins=20 outcube=psf_cube.fits
    echo "Ctpsfcube done!"

    ctedispcube inobs=../data/models/$model/events_"$hr"h_"$i".fits.gz incube=NONE caldb="$caldb" irf="$irf" coordsys=CEL proj=CAR xref="$ra" yref="$dec" binsz=1.0 nxpix=10 nypix=10 ebinalg=LOG emin="$emin" emax="$emax" enumbins=20 outcube=edisp_cube.fits
    echo "Ctedispcube done!"

    ctbkgcube inobs=../data/models/$model/events_"$hr"h_"$i".fits.gz caldb="$caldb" irf="$irf" incube=ctscube.fits inmodel=./models/$model.xml outcube=bkg_cube.fits outmodel=bkg_cube_"$hr"hr.xml
    echo "Ctebkgcube done!"

    ctlike edisp=True inobs=ctscube.fits expcube=cube.fits psfcube=psf_cube.fits bkgcube=bkg_cube.fits edispcube=edisp_cube.fits inmodel=bkg_cube_fix.xml outmodel=./results/models/$model/results_"$i".xml
    echo "Ctlike done!"

    csspec edisp=True inmodel=./results/models/$model/results_"$i".xml outfile=./results/models/$model/spectrum_"$i".fits inobs=ctscube.fits expcube=cube.fits psfcube=psf_cube.fits bkgcube=bkg_cube.fits edispcube=edisp_cube.fits srcname=$model method=AUTO ebinalg=LOG emin="$emin" emax="$emax" enumbins=$bin
    echo "Csspec done!"

    ctbutterfly edisp=True inmodel=./results/models/$model/results_"$i".xml outfile=./results/models/$model/spectrum_"$i".txt inobs=ctscube.fits expcube=cube.fits psfcube=psf_cube.fits bkgcube=bkg_cube.fits edispcube=edisp_cube.fits srcname=$model emin="$emin" emax="$emax"
    echo "Ctbutterfly done!"

    python Plot_SED.py -m "bkg_cube_fix.xml" -b "./results/models/$model/spectrum_"$i".txt" -n "$model" -s "./results/models/$model/spectrum_"$i".fits" -o "./results/models/$model/spectrum_"$i".png"

done;
