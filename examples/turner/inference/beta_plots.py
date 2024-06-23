import matplotlib.pyplot as plt
import torch
import numpy as np
import firedrake as df
from firedrake.checkpointing import CheckpointFile

beta_model = torch.load('beta_model.pt')
# beta dependancies explicit in space
with CheckpointFile("features.h5", 'r') as save_file:
    mesh = save_file.load_mesh('mesh')
    H = save_file.load_function(mesh, 'H0')
    B = save_file.load_function(mesh, 'B')
    S_grad = save_file.load_function(mesh, 'S_grad')
    B_grad = save_file.load_function(mesh, 'B_grad')
    adot = save_file.load_function(mesh, 'adot')

point1 = (-0.05391, 0.14211)
point2 = (0.23102, -0.11072)

Hp1 = H(point1)
Hp2 = H(point2)

Hmax = H.dat.data[:].max()
Hmin = H.dat.data[:].min()
H_grid = np.linspace(Hmin, Hmax, 300)

Bp1 = B(point1)
Bp2 = B(point2)

Bmax = B.dat.data[:].max()
Bmin = B.dat.data[:].min()
B_grid = np.linspace(Bmin, Bmax, 300)


S_gradp1 = S_grad(point1)
S_gradp2 = S_grad(point2)

Sgradxmax = S_grad.dat.data[:,0].max()
Sgradxmin = S_grad.dat.data[:,0].min()
Sgradx_grid = np.linspace(Sgradxmin, Sgradxmax, 300)

B_gradp1 = B_grad(point1)
B_gradp2 = B_grad(point2)

Bgradxmax = B_grad.dat.data[:,0].max()
Bgradxmin = B_grad.dat.data[:,0].min()
Bgradx_grid = np.linspace(Bgradxmin, Bgradxmax, 300)

adotp1 = adot(point1)
adotp2 = adot(point2)

H_curvep1 = torch.vstack([torch.tensor(H_grid),
              torch.tensor(Bp1 * np.ones(H_grid.shape)),
              torch.tensor(S_gradp1[0] * np.ones(H_grid.shape)),
              torch.tensor(S_gradp1[1] * np.ones(H_grid.shape)),
              torch.tensor(B_gradp1[0] * np.ones(H_grid.shape)),
              torch.tensor(B_gradp1[1] * np.ones(H_grid.shape)),
              torch.tensor(adotp1 * np.ones(H_grid.shape)),
              ]).T

H_curvep2 = torch.vstack([torch.tensor(H_grid),
              torch.tensor(Bp2 * np.ones(H_grid.shape)),
              torch.tensor(S_gradp2[0] * np.ones(H_grid.shape)),
              torch.tensor(S_gradp2[1] * np.ones(H_grid.shape)),
              torch.tensor(B_gradp2[0] * np.ones(H_grid.shape)),
              torch.tensor(B_gradp2[1] * np.ones(H_grid.shape)),
              torch.tensor(adotp2 * np.ones(H_grid.shape)),
              ]).T


B_curvep1 = torch.vstack([torch.tensor(Hp1 * np.ones(B_grid.shape)),
                        torch.tensor(B_grid),
                        torch.tensor(S_gradp1[0] * np.ones(B_grid.shape)),
                        torch.tensor(S_gradp1[1] * np.ones(B_grid.shape)),
                        torch.tensor(B_gradp1[0] * np.ones(B_grid.shape)),
                        torch.tensor(B_gradp1[1] * np.ones(B_grid.shape)),
                        torch.tensor(adotp1 * np.ones(B_grid.shape)),
                        ]).T

B_curvep2 = torch.vstack([torch.tensor(Hp2 * np.ones(B_grid.shape)),
                        torch.tensor(B_grid),
                        torch.tensor(S_gradp2[0] * np.ones(B_grid.shape)),
                        torch.tensor(S_gradp2[1] * np.ones(B_grid.shape)),
                        torch.tensor(B_gradp2[0] * np.ones(B_grid.shape)),
                        torch.tensor(B_gradp2[1] * np.ones(B_grid.shape)),
                        torch.tensor(adotp2 * np.ones(B_grid.shape)),
                        ]).T

Sgradx_curvep1 = torch.vstack([torch.tensor(Hp1 * np.ones(B_grid.shape)),
                               torch.tensor(Bp1 * np.ones(H_grid.shape)),
                               torch.tensor(Sgradx_grid),
                               torch.tensor(S_gradp1[1] * np.ones(B_grid.shape)),
                               torch.tensor(B_gradp1[0] * np.ones(B_grid.shape)),
                               torch.tensor(B_gradp1[1] * np.ones(B_grid.shape)),
                               torch.tensor(adotp1 * np.ones(B_grid.shape)),
                               ]).T


Sgradx_curvep2 = torch.vstack([torch.tensor(Hp2 * np.ones(B_grid.shape)),
                               torch.tensor(Bp2 * np.ones(H_grid.shape)),
                               torch.tensor(Sgradx_grid),
                               torch.tensor(S_gradp2[1] * np.ones(B_grid.shape)),
                               torch.tensor(B_gradp2[0] * np.ones(B_grid.shape)),
                               torch.tensor(B_gradp2[1] * np.ones(B_grid.shape)),
                               torch.tensor(adotp2 * np.ones(B_grid.shape)),
                               ]).T

Bgradx_curvep1 = torch.vstack([torch.tensor(Hp1 * np.ones(B_grid.shape)),
                               torch.tensor(Bp1 * np.ones(H_grid.shape)),
                               torch.tensor(S_gradp1[0] * np.ones(B_grid.shape)),
                               torch.tensor(S_gradp1[1] * np.ones(B_grid.shape)),
                               torch.tensor(Bgradx_grid),
                               torch.tensor(B_gradp1[1] * np.ones(B_grid.shape)),
                               torch.tensor(adotp1 * np.ones(B_grid.shape)),
                               ]).T


Bgradx_curvep2 = torch.vstack([torch.tensor(Hp2 * np.ones(B_grid.shape)),
                               torch.tensor(Bp2 * np.ones(H_grid.shape)),
                               torch.tensor(S_gradp2[0] * np.ones(B_grid.shape)),
                               torch.tensor(S_gradp2[1] * np.ones(B_grid.shape)),
                               torch.tensor(Bgradx_grid),
                               torch.tensor(B_gradp2[1] * np.ones(B_grid.shape)),
                               torch.tensor(adotp2 * np.ones(B_grid.shape)),
                               ]).T

adot_curvep1 = torch.vstack([torch.tensor(Hp1 * np.ones(B_grid.shape)),
                             torch.tensor(Bp1 * np.ones(H_grid.shape)),
                             torch.tensor(S_gradp1[0] * np.ones(B_grid.shape)),
                             torch.tensor(S_gradp1[1] * np.ones(B_grid.shape)),
                             torch.tensor(B_gradp1[0] * np.ones(B_grid.shape)),
                             torch.tensor(B_gradp1[1] * np.ones(B_grid.shape)),
                             torch.tensor(adot_grid)
                             ]).T


adot_curvep2 = torch.vstack([torch.tensor(Hp2 * np.ones(B_grid.shape)),
                             torch.tensor(Bp2 * np.ones(H_grid.shape)),
                             torch.tensor(S_gradp2[0] * np.ones(B_grid.shape)),
                             torch.tensor(S_gradp2[1] * np.ones(B_grid.shape)),
                             torch.tensor(B_gradp2[0] * np.ones(B_grid.shape)),
                             torch.tensor(B_gradp2[1] * np.ones(B_grid.shape)),
                             torch.tensor(adot_grid)
                             ]).T



logbeta_by_Hp1 = beta_model(H_curvep1)
beta_by_Hp1 = torch.exp(logbeta_by_Hp1)

logbeta_by_Hp2 = beta_model(H_curvep2)
beta_by_Hp2 = torch.exp(logbeta_by_Hp2)

logbeta_by_Bp1 = beta_model(B_curvep1)
beta_by_Bp1 = torch.exp(logbeta_by_Bp1)


plt.plot(H_grid, beta_by_Hp1.detach().numpy())
plt.savefig("beta_by_H_point1.png")
plt.cla()

plt.plot(H_grid, beta_by_Hp2.detach().numpy())
plt.savefig("beta_by_H_point2.png")
plt.cla()

plt.plot(B_grid, beta_by_Bp1.detach().numpy())
plt.savefig("beta_by_B_point1.png")


# beta depedancies with synthetic data

#torch.tensor
