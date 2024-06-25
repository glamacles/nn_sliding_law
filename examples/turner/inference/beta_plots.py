import matplotlib.pyplot as plt
import torch
import numpy as np
import firedrake as df
from firedrake.checkpointing import CheckpointFile

def plot_dependency_contour(feature, contour_feature):

    return None

def plot_dependency(feature, feature_idx, feature_list, points, model_filename, feature_filename, plot_filename, nsamples=300, direction=None):
    '''
    Params:
        feature - string, name of feature to plot
        feature_idx - int, index of feature to plot in feature_list
        feature_list - list{string}, list of string names of features in the order the NN expects them
        points - list{tuple{int}}, list of points to plot
        model_filename - string
        feature_filename - string, fd file of feature functions
        plot_filename - string
        nsamples - int, number of points to use to plot
        direction - int, if feature is vector valued the index of the direction to plot the feature in
    '''
    beta_model = torch.load(model_filename)
    with CheckpointFile(feature_filename, 'r') as save_file:
        mesh = save_file.load_mesh('mesh')
        plot_feature = save_file.load_function(mesh, feature)
        held = []
        for f in feature_list:
            held.append(save_file.load_function(mesh, f))

    if direction != None:
        feature_min = plot_feature.dat.data[:, direction].min()
        feature_max = plot_feature.dat.data[:, direction].max()
        feature_grid = np.linspace(feature_min, feature_max, nsamples)
    else:
        feature_min = plot_feature.dat.data[:].min()
        feature_max = plot_feature.dat.data[:].max()
        feature_grid = np.linspace(feature_min, feature_max, nsamples)

    leg_label = []
    for point in points:
        
        held_point = [val for f in held for val in  f(point).flatten()]
        held_point.pop(feature_idx)
        
        curve_coordinates = torch.vstack([torch.tensor(val * np.ones(feature_grid.shape)) for val in held_point])
        
        if feature_idx == 0:
            curve_coordinates = torch.cat([torch.unsqueeze(torch.tensor(feature_grid),0),
                                           curve_coordinates[feature_idx:]], 0).T
        elif feature_idx == curve_coordinates.size()[0]:
            curve_coordinates = torch.cat([curve_coordinates[:feature_idx],
                                           torch.unsqueeze(torch.tensor(feature_grid),0)], 0).T
        else:
            curve_coordinates = torch.cat([curve_coordinates[:feature_idx],
                                           torch.unsqueeze(torch.tensor(feature_grid),0),
                                           curve_coordinates[feature_idx:]], 0).T
            
        logbeta = beta_model(curve_coordinates)
        plt.plot(feature_grid, logbeta.detach().numpy())
        leg_label.append(str(point))

    if direction != None:
        plt.xlabel(feature + str(direction))
    else:
        plt.xlabel(feature)
    plt.ylabel("log(beta)")
    plt.legend(leg_label)
    plt.savefig(plot_filename)

