
import os
import sys
import pickle
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../../..')

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("once",category=DeprecationWarning)
    import firedrake as df
    from speceis.hybrid import CoupledModel, CoupledModelAdjoint, FenicsModel, VelocityIntegral, VelocityCost
import logging
logging.captureWarnings(True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)


# Define some model scaling parameters
len_scale = 50000
thk_scale = 5000
vel_scale = 100
beta_scale = 1000

# Establish a working directory
data_dir = '../meshes/mesh_1000/'
prefix = 'v1'
results_dir = f'{data_dir}/{prefix}/'

# Load and non-dimensionalize a mesh
mesh = df.Mesh(f'{data_dir}/mesh.msh',name='mesh')
mesh.coordinates.dat.data[:] -= (mesh.coordinates.dat.data.max(axis=0) + mesh.coordinates.dat.data.min(axis=0))/2.
mesh.coordinates.dat.data[:] /= len_scale

# Define some model configuration parameters
config = {'solver_type': 'gmres',
          'sliding_law': 'linear',
          'velocity_function_space':'MTW',
          'sia':False,
          'vel_scale': vel_scale,
          'thk_scale': thk_scale,
          'len_scale': len_scale,
          'beta_scale': beta_scale,
          'theta': 1.0,
          'thklim': 1./thk_scale,
          'alpha': 1000.0,
          'z_sea': 0.0,
          'boundary_markers':[1000,1001],
          'calve': 'b'}

# Define some model solver arguments
solver_args =         {'picard_tol':1e-3,
                       'momentum':0.5,
                       'max_iter':100,
                       'convergence_norm':'linf',
                       'update':True,
                       'enforce_positivity':True}

# Define an ice flow model and associated pytorch function  
model = CoupledModel(mesh,**config)
adjoint = CoupledModelAdjoint(model)
fm = FenicsModel

# Definte a fenics function for computing a misfit integral 
# over velocity and associate pytorch function
ui = VelocityIntegral(model,mode='lin',p=2.0,gamma=1.0)
um = VelocityCost

# Read in some velocity data at a bunch of years
years = [y for y in range(1985,2023)]
inds = [y-1985 for y in years]
not_inds = [y-1985 for y in range(1985,2023) if y not in years]

velocities = [None for i in range(len(years))]
velocities_0 = [None for i in range(len(years))]
velocities_tau = [None for i in range(len(years))]

# We just use to define a mask over terrain that is too steep
with open(f'{data_dir}/velocity/velocity.p','rb') as fi:
    [_,_,v_mask] = pickle.load(fi)
v_mask = torch.tensor(v_mask,dtype=float)

for i,y in enumerate(years):
    try:
        with open(f'{data_dir}/velocity/itslive_annual/velocity_{y}.p','rb') as fi:
            v = pickle.load(fi)/vel_scale
        v_tau = np.ones_like(v)
        v_tau[np.isnan(v)] = 0.0
        v_tau = v_tau[:,0]
        #v[np.isnan(v)] = 0.0#v_avg[np.isnan(v)]
        velocities[i] = v
        velocities_tau[i] = v_tau
    except FileNotFoundError:
        pass

# Set the precision to zero in regions with NaNs
v_avg = np.nanmean(velocities,axis=0)
v_tau = np.ones(v_avg.shape[0])
v_tau[np.isnan(v_avg[:,0]*v_avg[:,1])] = 0.
v_avg[np.isnan(v_avg[:,0]*v_avg[:,1])] = 0.

# Write the average velocity to file
output_Uobs = df.Function(ui.V,name='U_obs')
Uobs_file = df.File(f'{results_dir}/nn/U_obs.pvd')
output_Uobs.dat.data[:] = v_avg
Uobs_file.write(output_Uobs)

# Load geometry and mass balance info (mass balance info may not be used)
B = pickle.load(open('./init_data/B.p','rb'))
H0 = pickle.load(open('./init_data/H.p','rb'))
adot = pickle.load(open('./init_data/adot.p','rb'))
log_beta_ref = pickle.load(open('./init_data/log_beta_reference.p','rb')) # This is the traction field that I found in the bed inversion problem.
                                                                          # Only works with linear sliding law
beta2_ref = torch.exp(log_beta_ref)

# Running a model 
Ubar = torch.tensor(model.Ubar0.dat.data[:])
Udef = torch.tensor(model.Udef0.dat.data[:])
Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,1e-5,solver_args)

# Storing the output
U_file = df.File(f'{results_dir}/nn/U_s.pvd')
H_file = df.File(f'{results_dir}/nn/H.pvd')
model.project_surface_velocity()
U_file.write(model.U_s,time=0)
H_file.write(model.H0)

# Computing a vector of features
H_f = df.project(model.H0,model.Q_cg1)
B_f = df.project(model.B,model.Q_cg1)
V2 = df.VectorFunctionSpace(mesh,"CG",1)
S_grad = df.project(model.S_grad,V2)
B_grad = df.project(model.B_grad,V2)

features = torch.vstack((torch.tensor(B_f.dat.data[:]),torch.tensor(H_f.dat.data[:]),torch.linalg.norm(torch.tensor(S_grad.dat.data[:]),axis=1),torch.linalg.norm(torch.tensor(B_grad.dat.data[:]),axis=1))).T

