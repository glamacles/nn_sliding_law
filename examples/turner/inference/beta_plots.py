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
adotmax = adot.dat.data[:].max()
adotmin = adot.dat.data[:].min()
adot_grid = np.linspace(adotmin, adotmax, 300)


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

logbeta_by_Bp2 = beta_model(B_curvep2)
beta_by_Bp2 = torch.exp(logbeta_by_Bp2)

logbeta_by_Sgradxp1 = beta_model(Sgradx_curvep1)
beta_by_Sgradxp1 = torch.exp(logbeta_by_Sgradxp1)

logbeta_by_Sgradxp2 = beta_model(Sgradx_curvep2)
beta_by_Sgradxp2 = torch.exp(logbeta_by_Sgradxp2)

logbeta_by_Bgradxp1 = beta_model(Bgradx_curvep1)
beta_by_Bgradxp1 = torch.exp(logbeta_by_Bgradxp1)

logbeta_by_Bgradxp2 = beta_model(Bgradx_curvep2)
beta_by_Bgradxp2 = torch.exp(logbeta_by_Bgradxp2)

logbeta_by_adotp1 = beta_model(adot_curvep1)
beta_by_adotp1 = torch.exp(logbeta_by_adotp1)

logbeta_by_adotp2 = beta_model(adot_curvep2)
beta_by_adotp2 = torch.exp(logbeta_by_adotp2)




plt.plot(H_grid, logbeta_by_Hp1.detach().numpy())
plt.plot(H_grid, logbeta_by_Hp2.detach().numpy())
plt.xlabel("Thickness")
plt.ylabel("log(beta)")
plt.legend(["first point", "second point"])
plt.savefig("beta_by_H.png")
plt.cla()

plt.plot(B_grid, logbeta_by_Bp1.detach().numpy())
plt.plot(B_grid, logbeta_by_Bp2.detach().numpy())
plt.xlabel("Bed Height")
plt.ylabel("log(beta)")
plt.legend(["first point", "second point"])
plt.savefig("beta_by_B.png")
plt.cla()

plt.plot(Sgradx_grid, logbeta_by_Sgradxp1.detach().numpy())
plt.plot(Sgradx_grid, logbeta_by_Sgradxp2.detach().numpy())
plt.xlabel("x surface gradient")
plt.ylabel("log(beta)")
plt.legend(["first point", "second point"])
plt.savefig("beta_by_Sgradx.png")
plt.cla()

plt.plot(Bgradx_grid, logbeta_by_Bgradxp1.detach().numpy())
plt.plot(Bgradx_grid, logbeta_by_Bgradxp2.detach().numpy())
plt.xlabel("x bed gradient")
plt.legend(["first point", "second point"])
plt.ylabel("log(beta)")
plt.savefig("beta_by_bgradx.png")
plt.cla()

plt.plot(adot_grid, logbeta_by_adotp1.detach().numpy())
plt.plot(adot_grid, logbeta_by_adotp2.detach().numpy())
plt.xlabel("accumulation rate")
plt.ylabel("log(beta)")
plt.legend(["first point", "second point"])
plt.savefig("beta_by_accumulation.png")
