# Analytic Sampling of Sky Models
# Authors: [removed for review purposes]
# This file contains the core fitting implementation

# Argument handling
import sys

# Handle File System
import os.path

# Numpy
import numpy as np

# Curve fitting with scipy
from scipy.optimize import least_squares

# Plotting library
import matplotlib.pyplot as plt

# Image manipulation
from PIL import Image

# Results Output
import csv

import utils


class GMM:

    # visualise GMMs based on an existing model
    @staticmethod
    def visualize_gaussians():
        row_count = max(0, sum(1 for line in open(GMM.visualize_model_name)) - 1)
        count = row_count / GMM.MODEL_COMPONENT_COUNT

        if count > GMM.visualize_model_lim:
            count = GMM.visualize_model_lim
        count = int(np.ceil(np.sqrt(count)))
        GMM.num_rows = count
        GMM.num_columns = count
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('GMMs with samples\nModel: %s' % (GMM.visualize_model_name), fontsize=16)

        limit = 1
        model_file = open(GMM.visualize_model_name, newline='')
        model = csv.reader(model_file, delimiter=',', quotechar='|')
        next(model)
        gaussians = []
        iter = 0

        for gaussian in model:
            if limit > GMM.visualize_model_lim:
                break

            if iter == 0:
                turbidity = int(gaussian[0])
                elevation = int(gaussian[2])
                print('Visualizing GMM with Turbidity:%d, Elevation:%d' % (turbidity, elevation))

            gaussians.append(gaussian)
            iter += 1
            if iter % GMM.MODEL_COMPONENT_COUNT == 0:
                GMM.visualize(gaussians)
                limit += 1
                gaussians.clear()
                iter = 0
                continue
        model_file.close()

        fig.tight_layout()
        plt.show()

    # visualise a GMM based on an existing model
    def visualize(model_gaussians):
        gaussians = []
        weights = np.array([])

        turbidity = int(model_gaussians[0][0])
        elevation = int(model_gaussians[0][2])
        for gauss in model_gaussians:
            mean_x = float(gauss[4])
            mean_y = float(gauss[5])
            sigma_x = float(gauss[6])
            sigma_y = float(gauss[7])
            weight = float(gauss[8])
            weights = np.append(weights, weight)
            gaussians.append(utils.Gaussian2D(mean_x, mean_y, sigma_x, sigma_y, weight))

        normalized_weights = weights / weights.sum()
        boundsx = [0, 2.0 * np.pi]
        boundsy = [0, np.pi / 2.0]
        xmin, xmax, nx = boundsx[0], boundsx[1], 50
        ymin, ymax, ny = boundsy[0], boundsy[1], 50
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)

        skymap_fit = GMM.gmm_eval(X, Y, normalized_weights, gaussians)
        for i in range(len(gaussians)):
            gauss = gaussians[i]
            print(gauss)

        # Plot the 3D figure of the fitted function and the residuals.
        GMM.row_index += 1
        ax = plt.subplot(GMM.num_rows, GMM.num_columns, GMM.row_index, projection='3d')
        ax.set_title('Turbidity:%d, Elevation:%d' % (turbidity, elevation))
        ax.plot_surface(X, Y, skymap_fit, cmap='plasma', antialiased=True, rstride=4, cstride=4, alpha=0.25)
        ax.set_zlim(np.min(skymap_fit), np.max(skymap_fit))
        ax.set_xlim(boundsx[0], boundsx[1])
        ax.set_ylim(boundsy[0], boundsy[1])
        ax.view_init(elev=25., azim=-45.)
        # Turn off tick labels
        # ax.set_zticklabels([])
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])

        # cset = ax.contourf(X, Y, skymap_array-skymap_fit, zdir='z', offset=-4, cmap='plasma')
        # ax.set_zlim(-4,np.max(skymap_fit))

        N = 1500
        x = np.zeros(N)
        y = np.zeros(N)
        z = np.zeros(N)
        comps = np.random.uniform(size=N)

        cdf = []
        cdf.append(0)
        # build CDF
        for i in range(1, GMM.MODEL_COMPONENT_COUNT + 1):
            cdf.append(cdf[i - 1] + normalized_weights[i - 1])

        # select gaussian, sample and evaluate
        for i in range(N):
            comp = 0
            # select component
            for comp_i in range(1, GMM.MODEL_COMPONENT_COUNT + 1):
                if comps[i] < cdf[comp_i]:
                    comp = comp_i - 1
                    break

            selected_gauss = gaussians[comp]
            x[i], y[i] = selected_gauss.sample(1)
            for n in range(len(gaussians)):
                gauss = gaussians[n]
                val = utils.gaussian_truncated(x[i], y[i], gauss.meanx, gauss.meany, gauss.sigmax, gauss.sigmay,
                                               normalized_weights[n], boundsx[0], boundsx[1], boundsy[0], boundsy[1])
                z[i] += val
        ax.scatter(x, y, z, marker='.')


    # folder generation
    @staticmethod
    def makeFolders():
        os.makedirs(GMM.output_directory, exist_ok=True)

    # skip utility to avoid parsing certain configurations in a directory
    @staticmethod
    def skip(turbidity, elevation):
        # if ( turbidity != 4 or elevation != 23 ):
        # if ( turbidity != 4 or elevation != 73 ):
        # if (elevation != 43 and elevation != 3 and elevation != 13 and elevation != 73):
        # if (turbidity != 4):
        #    return True
        return False

    # fitting, calculation and comparison between different number of tGMMs
    @staticmethod
    def bestGMM():
        GMM.makeFolders()

        gaussians_comp_file_name = os.path.join(GMM.output_directory, 'model_comp.csv')
        gaussians_comp_file = open(gaussians_comp_file_name, 'w', newline='')
        gaussians_comp_file_writer = csv.writer(gaussians_comp_file, delimiter=',', quotechar='|',
                                                quoting=csv.QUOTE_MINIMAL)
        gaussians_comp_file_writer.writerow(
            ['Turbidity', 'Azimuth', 'Elevation', 'Num Gaussians', 'Normalization', 'MAE', 'RMSE'])

        fig = plt.figure()
        plt.title('GMMs Plot')
        plt.xlabel('GMMs')
        plt.ylabel('MAE')
        xint = range(GMM.compute_num_gaussians_min, GMM.compute_num_gaussians_max + 1)
        plt.xticks(xint)
        for skymap_file in os.listdir(GMM.skymap_directory):
            if not skymap_file.endswith(".tiff"):
                continue

            elevation = int(skymap_file.split('_')[1])
            turbidity = int(skymap_file.split('_')[2].split('.')[0])

            if GMM.skip(turbidity, elevation):
                continue

            skymap_file_full = os.path.join(GMM.skymap_directory, skymap_file)
            print("Fitting test for: %s. Result will be stored at: %s" % (skymap_file_full, gaussians_comp_file_name))

            skymap_image = Image.open(GMM.skymap_directory + '/' + skymap_file)
            skymap_image = skymap_image.resize((GMM.width, GMM.height), Image.ANTIALIAS)

            x = []
            y = []
            mae_min = sys.float_info.max
            for i in range(GMM.compute_num_gaussians_min, GMM.compute_num_gaussians_max + 1):
                print('Fitting with %d gaussians' % (i))

                local_factor, normalized_weights, skymap_array, skymap_fit, local_gaussians, local_mae, local_rmse = GMM.fit(
                    skymap_image, i)

                from sklearn.metrics import mean_absolute_percentage_error as mape
                mape(skymap_array, skymap_fit)
                gaussians_comp_file_writer.writerow(
                    [turbidity, GMM.AZIMUTH, elevation, i, local_factor, local_mae, local_rmse])
                if local_mae < mae_min:
                    mae_min = local_mae
                    print("New min found with %d gaussians" % (len(local_gaussians)))
                    print("MAE:", local_mae)
                    print("RMSE:", local_rmse)
                    print("Best Factor:", local_factor)
                x.append(i)
                y.append(local_mae)
            plt.plot(x, y, label='Tu%d_El%d' % (turbidity, elevation))
            plt.legend()
        plt.show()
        print('Saved file to:', gaussians_comp_file_name)

    # generic fitting function
    @staticmethod
    def fit(skymap_image, num_components):
        ncols, nrows = skymap_image.size
        # Curve fit
        gaussians, mae, rmse, max_res, factor = utils.fit_gmm_linear(skymap_image, ncomponents=num_components)
        print("RMSE:", rmse)
        print("MSE:", rmse * rmse)
        print("MAE:", mae)
        print("Max Res:", max_res)
        print("Best Factor:", factor)

        # normalize weights
        weights = np.array([])
        for gauss in gaussians:
            weights = np.append(weights, gauss.weight)
        normalized_weights = weights / weights.sum()
        print("Weights : ", weights)
        print("Normalized Weights : ", normalized_weights)

        # refine Weights for tGMMs
        # Image is column major so we transpose it
        skymap_array = np.array(skymap_image.getdata()).reshape((nrows, ncols))

        xmin, xmax, nx = 0.0, np.pi * 2.0, GMM.width
        ymin, ymax, ny = 0.0, np.pi / 2.0, GMM.height
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        ys = skymap_array

        def fun(weights):
            return (GMM.tgmm_eval(X, Y, weights, gaussians) - ys).flatten()

        weights0 = normalized_weights
        refined_weights = least_squares(fun, weights0)

        skymap_fit = GMM.tgmm_eval(X, Y, refined_weights.x, gaussians)

        normalized_weights = refined_weights.x / refined_weights.x.sum()
        print("Refined Normalized Weights:", normalized_weights)

        refined_mse = np.mean((skymap_array - skymap_fit) ** 2)
        refined_rmse = np.sqrt(refined_mse)
        refined_mae = np.mean(np.abs(skymap_array - skymap_fit))

        print('RMSE after refitting', refined_rmse)
        print('MSE after refitting', refined_mse)
        print('MAE after refitting', refined_mae)

        return factor, normalized_weights, skymap_array, skymap_fit, gaussians, refined_mae, refined_rmse

    # Removing axes margings in 3D plot
    # source: https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
    ###patch start###
    from mpl_toolkits.mplot3d.axis3d import Axis
    if not hasattr(Axis, "_get_coord_info_old"):
        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs

        Axis._get_coord_info_old = Axis._get_coord_info
        Axis._get_coord_info = _get_coord_info_new
    ###patch end###

    rows = 1
    cols = 4
    index = 1
    fig = None

    # display a GMM evaluation in hemispherical range and save it as svg
    @staticmethod
    def savefig(X, Y, skymap_array, savename, zlimup):
        if GMM.fig is None:
            font = {'family': 'Myriad Pro', 'size': 8}
            plt.rc('font', **font)
            GMM.fig = plt.figure(figsize=(6.9, 2))
            # GMM.fig = plt.figure(figsize=(12, 9))
            plt.tight_layout()
            GMM.fig.show()
        # skymap_array /= np.max(skymap_array)
        # ax = plt.axes(projection='3d')
        ax = plt.subplot(GMM.rows, GMM.cols, GMM.index, projection='3d')
        GMM.index = GMM.index + 1
        # ax.set_title(savename)
        ax.set_aspect('auto')
        ax.set_adjustable('box')
        ax.set_box_aspect((4, 4, 3))
        # ax.set_title('Original Data')
        norm = None  # plt.Normalize(vmin=0.0, vmax=1.0, clip=True)
        ax.plot_surface(X, Y, skymap_array, cmap='Spectral_r', antialiased=True, rstride=1, cstride=2,
                        edgecolor=(0, 0, 0, 0.125), norm=norm, linewidth=1)
        ax.set_zlim(0, zlimup)
        ax.set_ylim(0, 0.5 * np.pi)
        ax.set_xlim(0, 2.0 * np.pi)
        ax.set_xlabel("φ")
        ax.set_ylabel("θ")
        x_ticks = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
        x_ticks_str = ['0', 'π/2', 'π', '3π/2', '2π']
        y_ticks = [0, 0.125 * np.pi, 0.25 * np.pi, 0.375 * np.pi, 0.5 * np.pi]
        y_ticks_str = ['0', 'π/8', 'π/4', '3π/8', 'π/2']
        # x_ticks = [0, np.pi,  2.0 * np.pi]
        # x_ticks_str = ['0', 'π', '2π']
        # y_ticks = [0, 0.25 * np.pi, 0.5 * np.pi]
        # y_ticks_str = ['0', 'π/4', 'π/2']
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_str)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_str)
        ax.view_init(elev=25., azim=-65.)
        # plt.savefig(savename, bbox_inches='tight', dpi=300)

    # display a GMM evaluation in an extended range and save it as svg
    @staticmethod
    def savefigoverflow(X, Y, skymap_array, savename, X_clipped, Y_clipped, skymap_array_clipped, zlimup):
        if GMM.fig is None:
            GMM.fig = plt.figure(figsize=(12, 8))
            plt.tight_layout()
            GMM.fig.show()

        # skymap_array /= np.max(skymap_array)
        # skymap_array_clipped /= np.max(skymap_array_clipped)
        # ax = plt.axes(projection='3d')
        ax = plt.subplot(GMM.rows, GMM.cols, GMM.index, projection='3d')
        GMM.index = GMM.index + 1
        # ax.set_title(savename)
        ax.set_aspect('auto')
        ax.set_adjustable('box')
        ax.set_box_aspect((4, 4, 3))
        # ax.set_title('Original Data')
        norm = None
        # plt.Normalize(vmin=0.0, vmax=1.0, clip=True)
        # for i in range(GMM.width):
        #   for j in range(GMM.height):
        #       if X[j][i] > 0.1 and X[j][i] < 6.2 and Y[j][i] > 0.1 and Y[j][i] < 1.2:
        # skymap_array[j][i] = -1
        ax.plot_surface(X, Y, skymap_array, cmap='Spectral_r', antialiased=True, rstride=1, cstride=2,
                        edgecolor=(0, 0, 0, 0.125), norm=norm, linewidth=1, zorder=1)
        # ax.plot_surface(X_clipped, Y_clipped, skymap_array_clipped, cmap='Spectral_r', antialiased=True, rstride=2, cstride=2,
        #              edgecolor=(0, 0, 0, 0.125), norm=norm, linewidth=1, zorder=0)
        ax.set_zlim(0, zlimup)
        ax.set_ylim(-0.25 * np.pi, 0.75 * np.pi)
        ax.set_xlim(-np.pi, 2.0 * np.pi)
        ax.set_xlabel("φ")
        ax.set_ylabel("θ")
        # x_ticks = [-3 * np.pi, -2 * np.pi, -np.pi,  0, np.pi, 2.0*np.pi, 3.0 * np.pi]
        # x_ticks_str = ['-3π', '-2π', '-π', '0', 'π', '2π', '3π']
        # y_ticks = [-0.5 * np.pi,-0.375 * np.pi,-0.25 * np.pi,-0.125 * np.pi, 0, 0.125 * np.pi, 0.25 * np.pi, 0.375 * np.pi, 0.5 * np.pi]
        # y_ticks_str = ['-π/2', '-3π/8', '-π/4', '-π/8', '0', 'π/8', 'π/4', '3π/8', 'π/2']
        # x_ticks = [-3 * np.pi, -1.5 * np.pi, 0, 1.5*np.pi, 3.0 * np.pi]
        # x_ticks_str = ['-3π', '-3π/2', '0', '3π/2', '3π']
        # y_ticks = [-0.5 * np.pi,-0.25 * np.pi,0,  0.25 * np.pi, 0.5 * np.pi]
        # y_ticks_str = ['-π/2', '-π/4', '0',  'π/4', 'π/2']
        x_ticks = [-np.pi, 0, np.pi, 2.0 * np.pi, 3.0 * np.pi]
        x_ticks_str = ['-π', '0', 'π', '2π', '3π']
        y_ticks = [-0.25 * np.pi, 0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi]
        y_ticks_str = ['-π/4', '0', 'π/4', 'π/2', '3π/4']
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_str)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_str)
        ax.view_init(elev=25., azim=-65.)
        # plt.savefig(savename, bbox_inches='tight', dpi=300)

    # generate a mesh with the given bounds
    @staticmethod
    def genGrid(xbound1, xbound2, ybound1, ybound2):
        xmin, xmax, nx = xbound1, xbound2, GMM.width
        ymin, ymax, ny = ybound1, ybound2, GMM.height
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    # fit and display different GMM evaluations (Figure 4 in the paper)
    @staticmethod
    def fitAndDisplayPipeline(skymap_image, num_components):
        # 23-4, 9.75
        ncols, nrows = skymap_image.size
        # Curve fit
        gaussians, mae, rmse, max_res, factor = utils.fit_gmm_linear(skymap_image, ncomponents=num_components)
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("Max Res:", max_res)
        print("Best Factor:", factor)

        # normalize weights
        weights = np.array([])
        for gauss in gaussians:
            weights = np.append(weights, gauss.weight)
        normalized_weights = weights / weights.sum()
        print("Weights : ", weights)
        print("Normalized Weights : ", normalized_weights)

        # refine Weights for TGMMs
        # Image is column major so we transpose it
        skymap_array = np.array(skymap_image.getdata()).reshape((nrows, ncols))

        X, Y = GMM.genGrid(0.0, np.pi * 2.0, 0.0, np.pi * 0.5)
        ys = skymap_array

        def fun(weights):
            return (GMM.tgmm_eval(X, Y, weights, gaussians) - ys).flatten()

        weights0 = normalized_weights
        refined_weights = least_squares(fun, weights0)
        print(refined_weights.x)
        print(refined_weights.x.sum())

        # save figures
        print(np.max(skymap_array))
        GMM.savefig(X, Y, skymap_array, 'radiance_data.svg', 1.0)

        skymap_fit = GMM.gmm_eval(X, Y, normalized_weights, gaussians)
        print(np.max(skymap_fit))
        # GMM.savefig(X, Y, skymap_fit, 'gmm_original_fit.svg')

        X_overflow, Y_overflow = GMM.genGrid(-np.pi, np.pi * 3.0, -np.pi * 0.25, np.pi * 0.75)
        skymap_fit_overflow = GMM.gmm_eval(X_overflow, Y_overflow, normalized_weights, gaussians)
        print(np.max(skymap_fit_overflow))
        GMM.savefigoverflow(X_overflow, Y_overflow, skymap_fit_overflow, 'gmm_overflow.svg', X, Y, skymap_fit, 0.4)

        skymap_fit = GMM.tgmm_eval(X, Y, normalized_weights, gaussians)
        print(np.max(skymap_fit))
        GMM.savefig(X, Y, skymap_fit, 'tgmm_original_fit.svg', 0.8)

        skymap_fit = GMM.tgmm_eval(X, Y, refined_weights.x / refined_weights.x.sum(), gaussians)
        print(np.max(skymap_fit))
        GMM.savefig(X, Y, skymap_fit, 'tgmm_with_least_squares.svg', 1.0)

        plt.tight_layout()
        plt.show()
        # save figures

        normalized_weights = refined_weights.x / refined_weights.x.sum()
        print("Refined Normalized Weights:", normalized_weights)

        refined_mse = np.mean((skymap_array - skymap_fit) ** 2)
        refined_rmse = np.sqrt(refined_mse)
        refined_mae = np.mean(np.abs(skymap_array - skymap_fit))

        print('RMSE after refitting', refined_rmse)
        print('MSE after refitting', refined_mse)
        print('MAE after refitting', refined_mae)

        return factor, normalized_weights, skymap_array, skymap_fit, gaussians, refined_mae, refined_rmse

    # create a model by fitting on the given dataset
    @staticmethod
    def createGMMModel():
        GMM.makeFolders()

        model_file_name = os.path.join(GMM.output_directory, 'model.csv')

        model_file = open(model_file_name, 'w', newline='')
        model_writer = csv.writer(model_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        model_writer.writerow(
            ['Turbidity', 'Azimuth', 'Elevation', 'Normalization', 'Mean X', 'Mean Y', 'Sigma X', 'Sigma Y', 'Weight',
             'Volume', 'MAE', 'RMSE'])
        model_file.close()

        for skymap_file in os.listdir(GMM.skymap_directory):
            print(skymap_file)
            if not skymap_file.endswith(".tiff"):
                continue

            elevation = int(skymap_file.split('_')[1])
            turbidity = int(skymap_file.split('_')[2].split('.')[0])

            if GMM.skip(turbidity, elevation):
                continue

            print("Fitting elevation: " + str(elevation) + ", turbidity: " + str(turbidity) +" on " + os.path.join(GMM.skymap_directory,
                                               skymap_file) + ". Model will be stored at: " + model_file_name)

            preview_skymap_file = skymap_file
            skymap_image = Image.open(os.path.join(GMM.skymap_directory, skymap_file))
            # skymap_image = ImageOps.flip(skymap_image)
            skymap_image = skymap_image.resize((GMM.width, GMM.height), Image.ANTIALIAS)

            factor, normalized_weights, skymap_array, skymap_fit, gaussians, mae, rmse = GMM.fit(skymap_image,
                                                                                                 GMM.MODEL_COMPONENT_COUNT)

            # write GMM
            model_file = open(model_file_name, 'a', newline='')
            model_writer = csv.writer(model_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            print('Writing resulting GMMs to file')
            for i in range(GMM.MODEL_COMPONENT_COUNT):
                gauss = gaussians[i]
                gauss.weight = normalized_weights[i]
                model_writer.writerow(
                    [turbidity, GMM.AZIMUTH, elevation, factor, gauss.meanx, gauss.meany, gauss.sigmax, gauss.sigmay,
                     gauss.weight, gauss.volume(), mae, rmse])
                print(gauss)
            model_file.close()

            if GMM.plot:
                GMM.plotData(turbidity, elevation, skymap_array, skymap_fit, gaussians, factor, normalized_weights, mae,
                             rmse)

    # GMM evaluation
    @staticmethod
    def gmm_eval(X, Y, weights, gaussians):
        skymap_fit = np.zeros(X.shape)
        for i in range(len(gaussians)):
            gauss = gaussians[i]
            skymap_fit += utils.gaussian(X, Y, gauss.meanx, gauss.meany, gauss.sigmax, gauss.sigmay, weights[i])
        return skymap_fit

    # tGMM evaluation
    @staticmethod
    def tgmm_eval(X, Y, weights, gaussians):
        skymap_fit = np.zeros(X.shape)
        for i in range(len(gaussians)):
            gauss = gaussians[i]
            skymap_fit += utils.gaussian_truncated(X, Y, gauss.meanx, gauss.meany, gauss.sigmax, gauss.sigmay,
                                                   weights[i], 0, 2.0 * np.pi, 0, 0.5 * np.pi)

        return skymap_fit

    # plot data after fitting
    @staticmethod
    def plotData(turbidity, elevation, skymap_array, skymap_fit, gaussians, factor, normalized_weights, mae, rmse):

        xmin, xmax, nx = 0.0, np.pi * 2.0, GMM.width
        ymin, ymax, ny = 0.0, np.pi / 2.0, GMM.height
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)

        rows = 2
        columns = 3
        index = 0
        # Plotting original data
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Turbidity: %d, Elevation: %d, MAE: %.2e, RMSE: %.2e' % (turbidity, elevation, mae, rmse),
                     fontsize=16)
        index += 1
        ax = plt.subplot(rows, columns, index, projection='3d')
        ax.set_title('Original Data')
        ax.plot_surface(X, Y, skymap_array, cmap='plasma', antialiased=True, rstride=4, cstride=4)
        ax.set_zlim(0, np.max(skymap_array))
        ax.set_ylim(0, 0.5 * np.pi)
        ax.set_xlim(0, 2.0 * np.pi)
        ax.view_init(elev=25., azim=-45.)

        # Plot the 3D figure of the fitted function and the residuals.
        index += 1
        ax = plt.subplot(rows, columns, index, projection='3d')
        ax.set_title('Fitted Data')
        ax.plot_surface(X, Y, skymap_fit, cmap='plasma', antialiased=True,
                        rstride=4,
                        cstride=4)
        ax.set_zlim(0, np.max(skymap_fit))
        ax.set_ylim(0, 0.5 * np.pi)
        ax.set_xlim(0, 2.0 * np.pi)
        ax.view_init(elev=25., azim=-45.)
        # Turn off tick labels
        # ax.set_zticklabels([])
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])

        # cset = ax.contourf(X, Y, skymap_array-skymap_fit, zdir='z', offset=-4, cmap='plasma')
        # ax.set_zlim(-4,np.max(skymap_fit))
        # plt.savefig('test' + '.pdf')

        # display the CDF
        # build the CDF
        skymap_cdf = np.zeros(skymap_array.shape)
        for i in range(GMM.MODEL_COMPONENT_COUNT):
            # if (i != 1):
            #  continue
            gauss = gaussians[i]
            skymap_cdf += normalized_weights[i] * gauss.truncated_cdf(X, Y, 0, 2.0 * np.pi, 0, 0.5 * np.pi)

        index += 1
        ax = plt.subplot(rows, columns, index, projection='3d')
        ax.set_title('Fitted Data CDF')
        ax.plot_surface(X, Y, skymap_cdf, cmap='plasma', antialiased=True,
                        rstride=2,
                        cstride=2)
        ax.set_zlim(0, np.max(skymap_cdf))
        ax.set_ylim(0, 0.5 * np.pi)
        ax.set_xlim(0, 2.0 * np.pi)
        ax.view_init(elev=25., azim=-45.)
        print('CDF Max', np.max(skymap_cdf))

        # sample the GMMs
        N = 1000
        x = np.random.uniform(size=N)
        y = np.random.uniform(size=N)
        z = np.random.uniform(size=N)

        # build cdf
        cdf = np.array([0.0])
        for i in range(1, GMM.MODEL_COMPONENT_COUNT + 1):
            value = cdf[i - 1] + normalized_weights[i - 1]
            cdf = np.append(cdf, value)
        print('CDF : ', cdf)
        bins = np.zeros(5)
        for i in range(N):
            dice = x[i]
            comp = 0
            for j in range(1, GMM.MODEL_COMPONENT_COUNT + 1):
                if dice < cdf[j]:
                    comp = j - 1
                    break
            bins[comp] += 1

            gauss = gaussians[comp]
            y[i], z[i] = gauss.sample(1)
            y[i] *= GMM.width / (2.0 * np.pi)
            z[i] *= GMM.height / (0.5 * np.pi)

        # display the original data with GMM samples in 2D
        index += 1
        ax = plt.subplot(rows, columns, index)
        ax.set_title('Original Data Top Down')
        ax.imshow(skymap_array, cmap='gray')

        # display the fitted data in 2D
        index += 1
        ax = plt.subplot(rows, columns, index)
        # ax.scatter(y, z)
        ax.set_title('Fitted Data Top Down')
        # ax.imshow(skymap_array, cmap='gray')
        ax.imshow(skymap_fit, cmap='gray')

        # display the original data with GMM samples in 2D
        index += 1
        ax = plt.subplot(rows, columns, index)
        ax.set_title('Original Data Top Down with GMM Samples')
        ax.imshow(skymap_array, cmap='gray')
        ax.scatter(y, z, s=1)

        plt.show()

    # generates the dataset for the fitting process. Requires sky radiance files to be present
    @staticmethod
    def generateDataset(input, output):
        import OpenEXR as exr
        import Imath
        from PIL import Image

        os.makedirs(output, exist_ok=True)

        def toLuminance(val):
            return 0.212671 * val[0] + 0.715160 * val[1] + 0.072169 * val[2]

        for skymap_file in os.listdir(input):
            if not skymap_file.endswith(".exr"):
                continue

            skymap_image_ref_base = exr.InputFile(os.path.join(input,skymap_file))
            dw = skymap_image_ref_base.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            out_size = (int)(size[0] * size[1]/2)
            sky_image_ref_rgb = np.zeros((3, out_size))
            for i, c in enumerate('RGB'):
                rgb32f_ref = np.fromstring(skymap_image_ref_base.channel(c, pixel_type), dtype=np.float32)
                sky_image_ref_rgb[i] = rgb32f_ref[0:out_size]

            row_size = (int)(size[1]/2)
            column_size = size[0]
            skymap_image_ref_luma = toLuminance(sky_image_ref_rgb)
            skymap_image_ref_luma = np.reshape(skymap_image_ref_luma, (row_size, column_size))
            for r in range((int)(row_size)):
                theta = 0.5*np.pi * (r + 0.5) / row_size
                skymap_image_ref_luma[r] *= np.sin(theta)

            max_v = np.max(skymap_image_ref_luma)
            skymap_image_ref_luma = skymap_image_ref_luma / max_v

            file = os.path.splitext(skymap_file)[0]
            filename = os.path.join(output, file + ".tiff")
            print('Writing to: ' + filename)

            im = Image.fromarray(np.float32(skymap_image_ref_luma), 'F').transpose(Image.FLIP_TOP_BOTTOM)
            im.save(filename)

    MODEL_COMPONENT_COUNT = 5  # Number of Gaussians, 5 in our specific case
    MODEL_PARAM_COUNT = 5  # Parameters of the Gaussian component functions (meanx, meany, sigmax, sigmay, weight)
    AZIMUTH = 90  # Not that it matters much but we set it to

    # width = 1024
    # height = 256
    width = 256
    height = 64
    num_rows = 0
    num_columns = 2
    row_index = 0
    skymap_directory = '../../dataset/hosek/hosek_sky_luminance'
    plot = False
    visualize_model = False
    visualize_model_name = None
    visualize_model_lim = 4
    compute_num_gaussians = False
    compute_num_gaussians_min = 1
    compute_num_gaussians_max = 8
    output_directory = 'fit'