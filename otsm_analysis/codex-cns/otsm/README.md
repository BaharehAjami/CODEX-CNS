The `otsm_<>.ipyb` contain the files to perform segmentation. The output masks will then need to be passed onto [https://github.com/adyprat/quantification](https://github.com/adyprat/quantification) to obtain average intensity values, using:

`python  python CommandSingleCellExtraction.py --masks 3026GliaMask.tif --channel_names 3026Markers.csv --image 3026CtlCortex.qptiff --output ../`
