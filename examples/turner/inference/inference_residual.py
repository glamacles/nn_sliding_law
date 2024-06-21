
import os
import sys
import pickle
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../../..')

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("once",category=DeprecationWarning)
    import firedrake as df
    from firedrake.petsc import PETSc
    from speceis_dg.hybrid_diff import CoupledModel, CoupledModelAdjoint, FenicsModel, SurfaceIntegral, SurfaceCost, VelocityIntegral, VelocityCost, Residual, Projector
import logging
logging.captureWarnings(True)
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import pyevtk.hl
import time

len_scale = 50000
thk_scale = 5000
vel_scale = 100

#data_dir = '../meshes/mesh_1899/'
data_dir = '../meshes/mesh_1000/'
prefix = 'v1'

initialize = True
hot_start = False

if initialize:
    results_dir = f'{data_dir}/{prefix}/init/'
else:
    results_dir = f'{data_dir}/{prefix}/time/'

mesh = df.Mesh(f'{data_dir}/mesh.msh',name='mesh')
mesh.coordinates.dat.data[:] -= (mesh.coordinates.dat.data.max(axis=0) + mesh.coordinates.dat.data.min(axis=0))/2.
mesh.coordinates.dat.data[:] /= len_scale

config = {'solver_type': 'gmres',
          'sliding_law': 'linear',
          'velocity_function_space':'MTW',
          'sia':False,
          'vel_scale': vel_scale,
          'thk_scale': thk_scale,
          'len_scale': len_scale,
          'beta_scale': 1000.,
          'theta': 1.0,
          'thklim': 1./thk_scale,
          'alpha': 1000.0,
          'z_sea': 0.0,
          'boundary_markers':[1000,1001],
          'calve': 'b'}
  
model = CoupledModel(mesh,**config)
adjoint = CoupledModelAdjoint(model)
ui = VelocityIntegral(model,mode='lin',p=2.0,gamma=1.0)
res = Residual(model)

load_relative=False
if initialize:
    files=['cop30']
    times=['2013_175']
    dates=[int(s[:4]) for s in times]
else:
    if load_relative:
        files=['rel_1995','rel_2000','rel_2003','rel_2007','rel_2009','rel_2010','cop30','rel_2014','rel_2015','rel_2016','rel_2017','rel_2018','rel_2019','rel_2020','rel_2021']
    else:
        files=['1995','2000','2003','2007','2009','2010','cop30','2014','2015','2016','2017','2018','2019','2020','2021']

    times=['1995','2000','2003','2007','2009','2010','2013','2014','2015','2016','2017','2018','2019','2020','2021']
    dates=[int(s[:4]) for s in times]

years = [y for y in range(1985,2023)]
inds = [y-1985 for y in years]
not_inds = [y-1985 for y in range(1985,2023) if y not in years]

surfaces = [None for i in range(len(years))]
velocities = [None for i in range(len(years))]
velocities_0 = [None for i in range(len(years))]
velocities_tau = [None for i in range(len(years))]

with open(f'{data_dir}/velocity/velocity.p','rb') as fi:
    [v_avg,v_err,v_mask] = pickle.load(fi)
    v_avg/=vel_scale
v_avg[np.isnan(v_avg)] = 0.0
v_mask = torch.tensor(v_mask,dtype=float)
#v_mask[:] = True

for i,y in enumerate(years):
    try:
        with open(f'{data_dir}/velocity/itslive_annual/velocity_{y}.p','rb') as fi:
            v = pickle.load(fi)/vel_scale
        velocities_0[i] = v
    except FileNotFoundError:
        pass

v_avg = np.nanmean(velocities_0[:-2],axis=0)
v_avg[np.isnan(v_avg)] = 0.0

for i,y in enumerate(years):
    try:
        with open(f'{data_dir}/velocity/itslive_annual/velocity_{y}.p','rb') as fi:
            v = pickle.load(fi)/vel_scale
        v_tau = np.ones_like(v)
        v_tau[np.isnan(v)] = 0.0
        v_tau = v_tau[:,0]
        v[np.isnan(v)] = 0.0#v_avg[np.isnan(v)]
        velocities[i] = v
        velocities_tau[i] = v_tau
    except FileNotFoundError:
        v = v_avg
        v_tau = np.ones_like(v)
        velocities[i] = v
        velocities_tau[i] = v_tau[:,0]

for i,y in enumerate(years):
    try:
        f = dict(zip(dates,files))[y]
        with open(f'{data_dir}/surface_mesh/time_series/map_{f}.p','rb') as fi:
            surfaces[i] = pickle.load(fi)
    except KeyError:
        pass

data_ref = surfaces[years.index(2013)]

with open(f'{data_dir}/bed/bed_basis_sys.p','rb') as fi:
    data_bed = pickle.load(fi)

L_B_obs = data_bed['observation_basis']['coeff_map']
#h_B_obs = data_bed['observation_basis']['mean_map']

L_B_mod = data_bed['model_basis']['coeff_map']
#h_B_mod = data_bed['model_basis']['mean_map']

wmean_post_B = data_bed['coefficients']['post_mean']
G_B = data_bed['coefficients']['post_cov_root']
B_obs = data_bed['data']['z_obs']

B_mean = L_B_mod @ wmean_post_B 
B_map = L_B_mod @ G_B

with open(f'{data_dir}/beta/beta_basis.p','rb') as fi:
    beta_map_x,beta_map_t = pickle.load(fi)

thklim = torch.ones_like(B_mean)
thklim.data[:] = 1/thk_scale

fm = FenicsModel
sm = SurfaceCost
um = VelocityCost
pr = Projector

S_init = data_ref['elevation']
S_init[torch.isnan(S_init)] = 0

with open(f'{data_dir}/adot/adot_basis.p','rb') as fi:
    f,adot_map,adot_mean = pickle.load(fi)

n_hist_steps = 10
hist_length = 75

z_B = torch.zeros(B_map.shape[1],requires_grad=True)

z_beta_ref = torch.randn(beta_map_x.shape[1],requires_grad=True)
z_beta_t = torch.randn(beta_map_x.shape[1],beta_map_t.shape[1],requires_grad=True)

z_beta_ref.data[:]*=0.1
z_beta_t.data[:]*=0.001

z_adot = torch.zeros(adot_map.shape[1],requires_grad=True)

solver_args =         {'picard_tol':1e-3,
                       'momentum':0.5,
                       'max_iter':100,
                       'convergence_norm':'l2',
                       'update':True,
                       'enforce_positivity':True}

sigma_surf = 25/thk_scale
gamma_surf = 0.01/thk_scale

sigma_vel = 25/vel_scale
sigma2_vel = sigma_vel**2
gamma_vel = 0/vel_scale

rel_frac = 0.0
sigma2_mod = (1/thk_scale)**2

H_file = df.File(f'{results_dir}/adjoint/H.pvd')
H0_file = df.File(f'{results_dir}/adjoint/H0.pvd')
S_file = df.File(f'{results_dir}/adjoint/S.pvd')
Sobs_file = df.File(f'{results_dir}/adjoint/Sobs.pvd')
B_file = df.File(f'{results_dir}/adjoint/B.pvd')
Bstd_file = df.File(f'{results_dir}/adjoint/Bstd.pvd')
D_file = df.File(f'{results_dir}/adjoint/delta.pvd')
U_file = df.File(f'{results_dir}/adjoint/U_s.pvd')


Uobs_file = df.File(f'{results_dir}/adjoint/U_obs.pvd')
Sobs_file = df.File(f'{results_dir}/adjoint/S_obs.pvd')
log_beta_file = df.File(f'{results_dir}/adjoint/log_beta.pvd')
adot_file = df.File(f'{results_dir}/adjoint/adot.pvd')
adot0_file = df.File(f'{results_dir}/adjoint/adot0.pvd')
misfit_file = df.File(f'{results_dir}/adjoint/misfit.pvd')

output_H = df.Function(model.Q_thk,name='H')
output_S = df.Function(model.Q_thk,name='S')
output_f = df.Function(model.Q_thk,name='f')
output_log_beta = df.Function(model.Q_cg1,name='log_beta')
output_delta = df.Function(model.Q_thk,name='delta')
output_misfit = df.Function(model.Q_thk,name='misfit')
output_Uobs = df.Function(ui.V,name='U_obs')

output_Uobs.dat.data[:] = v_avg
Uobs_file.write(output_Uobs,time=0)

output_S.dat.data[:] = S_init
Sobs_file.write(output_S,time=0)

rho_velocity = 750.0
#rho_velocity = 1000.0
rho_surface = 2.0
zero = torch.zeros_like(S_init)

phi = df.TestFunction(model.Q_thk)
area = torch.tensor(df.assemble(phi*df.dx).dat.data[:])
M = torch.sqrt(area).reshape(-1,1)

Ubar_prev = torch.tensor(model.Ubar0.dat.data[:])
Udef_prev = torch.tensor(model.Udef0.dat.data[:])
Ubar_steady = torch.tensor(model.Ubar0.dat.data[:])
Udef_steady = torch.tensor(model.Udef0.dat.data[:])
B_steady = torch.tensor(model.B.dat.data[:])
H_steady = torch.tensor(model.H0.dat.data[:])

i = 0
tt = 0
inits = []

def closure():
    global Ubar_prev
    global Udef_prev
    global Ubar_steady
    global Udef_steady
    global B_steady
    global H_steady
    global i
    global tt
    global inits

    lbfgs.zero_grad()

    B = B_mean + B_map @ z_B
    S0 = S_init
    S0 = torch.maximum(B+thklim,S0)
    
    H0 = (S0-B)

    adot = f@(adot_mean + adot_map @ z_adot)

    log_beta_ref = beta_map_x @ z_beta_ref
    log_beta_t = beta_map_x @ (z_beta_ref.reshape(-1,1) + z_beta_t @ beta_map_t.T)
    beta2_ref = torch.exp(log_beta_ref)
    beta2 = torch.exp(log_beta_t)

    output_log_beta.dat.data[:] = log_beta_ref.detach().numpy()
    log_beta_file.write(output_log_beta,time=tt)

    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar_prev,Udef_prev,model,adjoint,0.0,5,solver_args)
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,10,solver_args)
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,20,solver_args)
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,40,solver_args)

    if i==0:
        n_steps = 25
    else:
        n_steps = 25
    
    for k in range(n_steps):
        Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,50,solver_args)

    adot_series = df.File(f'{results_dir}/adjoint/adot_series.pvd')
    U_series = df.File(f'{results_dir}/adjoint/U_series.pvd')
    Uobs_series = df.File(f'{results_dir}/adjoint/Uobs_series.pvd')
    misfit_series = df.File(f'{results_dir}/adjoint/misfit_series.pvd')
    beta_series = df.File(f'{results_dir}/adjoint/beta_series.pvd')

    output_H.dat.data[:] = H0.detach().numpy()
    H0_file.write(output_H,time=tt)
    adot0_file.write(model.adot,time=tt)
    
    model.project_surface_velocity()
    U_file.write(model.U_s,time=tt)

    U_series.write(model.U_s,time=1909)
    adot_series.write(model.adot,time=1909)
    """
    for k in range(n_hist_steps):
        Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,hist_length/n_hist_steps,solver_args)
        model.project_surface_velocity()
        ti = 1909 + (k+1)*hist_length/n_hist_steps
        U_series.write(model.U_s,time=ti)
        adot_series.write(model.adot,time=ti)
    """
    
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,0.01,solver_args)

    L_surface = 0.0
    L_velocity = 0.0

    S_count = 0
    U_count = torch.tensor(1e-10)

    surfs = []

    for j,(U_obs,U_tau,data,y) in enumerate(zip(velocities,velocities_tau,surfaces,years)):
        #Ubar_init,Udef_init = inits[j]
        if static_traction:
            beta2_i = beta2_ref
            adot_i = adot
        else:
            beta2_i = beta2[:,y-1985]
            adot_i = adot
            #Ubar,Udef,H0 = fm.apply(H0,B,beta2_i,adot_i,Ubar_init,Udef_init,model,adjoint,0.0,1.0,solver_args)
            Ubar,Udef,H0 = fm.apply(H0,B,beta2_i,adot_i,Ubar,Udef,model,adjoint,0.0,1.0,solver_args)
        #inits[j] = (Ubar.detach(),Udef.detach())
        S = (B + H0)
        
        if data is not None:
            S_obs = torch.nan_to_num(data['elevation'])
            #S_mask = torch.tensor((data['obs_count']>0)*data['steepness_mask']).to(bool)
            S_mask = torch.tensor((data['obs_count']>0)).to(bool)

            r_S = (S_obs - S)[S_mask]
            v_S = torch.ones_like(r_S)*gamma_surf

            g_S = r_S/sigma_surf*M[S_mask]
            q_S = v_S/sigma_surf*M[S_mask]

            L_surface_i = rho_surface*((g_S*g_S).sum() - (g_S * q_S).sum() / (1 + (q_S**2).sum()) * (g_S*q_S).sum())
            L_surface += L_surface_i
 
            S_count += 1
            
            output_misfit.dat.data[:] = (S - S_obs).detach().numpy()
            output_misfit.dat.data[data['obs_count']==0] = torch.nan
            misfit_series.write(output_misfit,time=y)
        
        #if U_obs is not None:
        #    if relative_velocity_loss:
        #        r_u = (pr.apply((Ubar - Ubar_steady) - 0.25*(Udef - Udef_steady),U_obs-v_avg,res))#[v_mask]
        #    else:
        #        r_u = (pr.apply(Ubar - 0.25*Udef,U_obs,res))
                #print(r_u.shape,v_mask.shape)

        #    v_u = torch.ones_like(r_u)*gamma_vel

        #    g_u = r_u/sigma_vel*M#[v_mask]
        #    q_u = v_u/sigma_vel*M#[v_mask]

        #    L_velocity_i = rho_velocity*((g_u * g_u)*v_mask[:,np.newaxis]).sum()# - (g_u * q_u).sum() / (1 + (q_u**2).sum()) * (g_u * q_u).sum())
        #    L_velocity = L_velocity + L_velocity_i

        #    U_count += 1

        if U_obs is not None:
            #tau_obs = 10*torch.ones(U_obs.shape[0])#1./np.maximum((np.linalg.norm(U_obs,axis=1)*rel_frac)**2,sigma2_vel)
            tau_obs = U_tau/np.maximum((np.linalg.norm(U_obs,axis=1)*rel_frac)**2,sigma2_vel)
            if not relative_velocity_loss:
                L_velocity_i = um.apply(Ubar,Udef,U_obs,tau_obs,v_mask,ui)*rho_velocity
            else:
                Ubar_rel = Ubar - Ubar_steady
                Udef_rel = Udef - Udef_steady
                U_obs_rel = U_obs - v_avg
                L_velocity_i = um.apply(Ubar_rel, Udef_rel,U_obs_rel,tau_obs,v_mask,ui)*rho_velocity
            L_velocity = L_velocity + L_velocity_i
            U_count += 1

        
        model.project_surface_velocity()
        U_series.write(model.U_s,time=y)
        output_Uobs.dat.data[:] = U_obs
        Uobs_series.write(output_Uobs,time=y)
        beta_series.write(model.beta2,time=y)
        adot_series.write(model.adot,time=y)
        surfs.append(S.detach().numpy())    

    D_series = df.File(f'{results_dir}/adjoint/D_series.pvd')
    for j,(S_tilde,y) in enumerate(zip(surfs,years)):
        output_delta.dat.data[:] = S_tilde - surfs[years.index(2013)]
        D_series.write(output_delta,time=y)

    L_prior = (z_B**2).sum() + (z_beta_ref**2).sum() + (z_beta_t**2).sum() + ((z_adot)**2).sum()
    if initialize:
        L_velocity = L_velocity/np.sqrt(U_count)
    else:
        L_velocity = L_velocity/np.sqrt(U_count)

    L =  L_velocity + L_surface + L_prior
    dV_0 = ((area*adot).sum()/area.sum()).detach()
    dV_1 = ((area*adot_i).sum()/area.sum()).detach()
    print(q,i,L.item(),L_velocity.item(),L_surface.item(),L_prior.item(),dV_0.item(),dV_1.item(),adot.max().detach().item(),adot_i.max().detach().item())

    if i%1==0:
        output_H.dat.data[:] = H0.detach().numpy()
        H_file.write(output_H,time=tt)

        output_S.dat.data[:] = S.detach().numpy()
        S_file.write(output_S,time=tt)

        B_file.write(model.B,time=tt)

        output_misfit.dat.data[:] = S.detach().numpy() - S_init.detach().numpy()
        misfit_file.write(output_misfit,time=tt)

        adot_file.write(model.adot,time=tt)

    if initialize:
        Ubar_steady.data[:] = Ubar.detach()
        Udef_steady.data[:] = Udef.detach()
        B_steady.data[:] = B.detach()
        H_steady.data[:] = H0.detach()

    L.backward()
    
    i+=1
    tt+=1
    return L


if initialize:
    lbfgs = torch.optim.LBFGS([z_B,z_beta_ref,z_adot],
                        history_size=10, lr=0.3,
                        line_search_fn="strong_wolfe",max_iter=50)

    static_traction = True
    relative_surface_loss = False
    relative_velocity_loss = False

    for q in range(0,10):
        lbfgs.step(closure)
        i=0

        with open(f'{results_dir}/states/state_{q:03d}.p','wb') as fi:
            pickle.dump((z_beta_ref.detach(),z_beta_t.detach(),z_B.detach(),z_adot.detach(),(Ubar_steady.detach().numpy(),Udef_steady.detach().numpy(),B_steady.detach().numpy(),H_steady.detach().numpy())),fi)

else:
    lbfgs = torch.optim.LBFGS([z_B,z_beta_ref,z_beta_t,z_adot,z_nse],
                        history_size=20,
                        line_search_fn="strong_wolfe",max_iter=50)

    initdir = f'{data_dir}/v5/init/states/'
    initfile = 'state_004.p'#max(os.listdir(initdir))
    with open(f'{initdir}/{initfile}','rb') as fi:
        data = pickle.load(fi)
        z_beta_ref.data[:],z_beta_t.data[:],z_B.data[:],z_adot.data[:],z_nse.data[:] = data[:5]
        Ubar_steady.data[:],Udef_steady.data[:],B_steady.data[:],H_steady.data[:] = [torch.from_numpy(x) for x in data[5]]

    static_traction = False
    relative_surface_loss = False
    relative_velocity_loss = True
    
    for q in range(0,10):
        lbfgs.step(closure)
        i = 0

        with open(f'{results_dir}/states/state_{q:03d}.p','wb') as fi:
            pickle.dump((z_beta_ref.detach(),z_beta_t.detach(),z_B.detach(),z_adot.detach(),z_nse.detach(),(Ubar_steady.detach().numpy(),Udef_steady.detach().numpy(),B_steady.detach().numpy(),H_steady.detach().numpy())),fi)




    


