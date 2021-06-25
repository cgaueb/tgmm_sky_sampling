# Analytic Sampling of Sky Models
# Authors: [removed for review purposes]
# This file contains the command-line interface for the fitting code.

import argparse
from gmm_fitting import GMM

if __name__ == '__main__':

    # Example instructions
    # python main.py -g ..\..\dataset\hosek\hosek_sky ..\..\dataset\hosek\hosek_sky_luminance
    # python main.py -v "fit\model\params\model.csv" 4 to visualize the first 4 GMMs in the model
    # Change GMM.skip() to skip any needed files during fitting
    # python main.py -f to fit
    # python main.py -f -p to fit and plot each setting
    # python main.py -b 1 4 to fit and plot GMMs from 1-4

    parser = argparse.ArgumentParser()
    parser.add_argument('--generateDataset', '-g', metavar=('input directory', 'output_directory'), type=str, nargs=2, help="Generates the dataset for the fitting process.")
    parser.add_argument('--best_num_gaussians', '-b', metavar=('min_value', 'max_value'), type=int, nargs=2, help="Performs fitting and stores the best GMM count. Expects min max arguments.")
    parser.add_argument('--fit', '-f', action='store_true', help="Fits the .tiff skymap files in the skymap dir")
    parser.add_argument('--plot', '-p', action='store_true', help="Plots the result during fitting")
    parser.add_argument('--visualize_model', '-v', action='store', nargs=2, type=str, metavar=('model name', 'max_GMMs_to_show'), help="Visualises up to 'max_GMMs' from the model")
    parser.add_argument('--skymapdir', '-s', action='store', metavar=('skymap directory'), nargs=1, type=str, help="Sky Model PDF directory. Default is ../../dataset/hosek/hosek_sky_luminance")
    parser.add_argument('--outdir', '-o', action='store', metavar=('output directory'), nargs=1, type=str, help="Output directory. Default is fit")
    args = parser.parse_args()
    argsdict = vars(args)

    # custom run
    # args.visualize_model = ['fit\model\params\model.csv', '4'] # to enable GMM visualisation
    # args.best_num_gaussians = [1,3] # to enable comparative calculation of GMMs
    # args.fit = True # to enable fit
    # args.plot = True # to enable plotting after fit
    #args.generatePDF = ['..\..\dataset\hosek\hosek_sky', '..\..\dataset\hosek\hosek_sky_pdf2']
    # end custom run

    print(args)
    if args.fit is False and args.visualize_model is None and args.best_num_gaussians is None and args.generateDataset is None:
        parser.error("One of --generateDataset, --fit, --visualize_model, --best_num_gaussians need to be given")

    if args.outdir is not None:
        GMM.output_directory = args.outdir[0]

    if args.skymapdir is not None:
        GMM.skymap_directory = args.skymapdir[0]

    GMM.plot = args.plot

    if args.visualize_model is not None:
        GMM.visualize_model = True
        GMM.visualize_model_name = args.visualize_model[0]
        GMM.visualize_model_lim = int(args.visualize_model[1])
        GMM.visualize_gaussians()

    if args.best_num_gaussians is not None:
        GMM.compute_num_gaussians = True
        GMM.compute_num_gaussians_min = args.best_num_gaussians[0]
        GMM.compute_num_gaussians_max = args.best_num_gaussians[1]
        GMM.bestGMM()

    if args.fit:
        GMM.createGMMModel()

    if args.generateDataset:
        GMM.generateDataset(args.generateDataset[0], args.generateDataset[1])