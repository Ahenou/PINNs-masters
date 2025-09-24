import os
os.system('cls') # clear console

#%% Importing libraries
import torch                        # import torch library
import torch.nn as nn               # import nn module for creating layers, models and functions
import numpy as np                  # import numpy library
import matplotlib
import matplotlib.pyplot as plt     # import pyplot library
from torch.autograd import grad, Variable # variable from numpy to torch

seed = 27
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#%% Defining device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set GPU as processing device
print(device)

#%% Problem setup: axisymmetric domain, material, and load
a = 0.01     # radius of loaded area [m]
R = 5.0      # domain radius [m]
Z = 5.0      # domain depth [m]
P0 = 1000.    # total vertical load [N]
miu = 0.3    # adim. (Poisson's ratio)
P  = P0/(np.pi*a**2)  # vertical pressure [Pa]


# Hyperparameters
epochs        = 4000  # number of training epochs
col_p         = 150   # number of collocation points
hidden_size   = 25    # number of neurons per hidden layer
learning_rate = 0.05 # learning rate parameter

#%% Defining NN for each stress component:
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1)

    def forward(self,r,z):
        # Concatenate the input features into a single tensor
        inputs = torch.cat([r, z], axis=1)
        phi = torch.tanh(self.fc1(inputs))
        phi = torch.tanh(self.fc2(phi))
        phi = torch.tanh(self.fc3(phi))
        phi = torch.tanh(self.fc4(phi))
        phi = self.fc5(phi)
        return phi

model = NN().to(device)                                          # transfer net to GPU for processing
mse_  = nn.MSELoss()                          # mean squared error as loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer for parameter updating

#%%Biharmonic
def diff(u,x,order=1):
    if order == 1:
        return grad(u.sum(), x, create_graph=True)[0]
    else:
        du_dx = diff(u,x,order=1)
        return diff(du_dx,x,order=order-1)

def laplacian(f,r,z):
    return diff(diff(f,r),r) + (1/r)*diff(f,r) + diff(diff(f,z),z)

def biharmonic(f,r,z):
    return laplacian(laplacian(f,r,z),r,z)

#Physical constraints
#stresses in terms of phi
def stresses(phi,r,z):
    phi = model(r,z)
    dphi_dr = diff(phi,r,1)
    d2phi_dr2 = diff(phi,r,2)
    d2phi_dz2 = diff(phi,z,2)

    #Love Stress formulas for axisymmetric case
    sigma_r     = diff((miu*laplacian(phi,r,z)-d2phi_dr2),z,1)
    sigma_theta = diff((miu*laplacian(phi,r,z)-(1/r)*dphi_dr),z,1)
    sigma_z     = diff(((2-miu)*laplacian(phi,r,z)-d2phi_dr2),z,1)
    tau_rz      = diff((1-miu)*laplacian(phi,r,z)-d2phi_dz2, r,1)

    return sigma_r, sigma_theta, sigma_z, tau_rz


#%% for variable creation
def to_var(x): return Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)

#for converting to numpy and save vectors
def to_np(x):  return x.data.cpu().numpy() # convert to numpy array

Loss_pde = []
Loss_sup1 = []
Loss_sup2 = []

#%%Defining collocation points
#Loss based on PDE
r_col = np.random.uniform(1e-5, R, (col_p, 1))  # random points in r direction
r_col = to_var(r_col)
z_col = np.random.uniform(1e-5, Z, (col_p, 1))  # random points in z direction
z_col = to_var(z_col)

#Loss based on BC (surface z=0) outer boundary r=a to r=R
r_sup = np.random.uniform(a+1e-5, R, (col_p, 1))  # random points in r direction
r_sup = to_var(r_sup) 
z_sup = np.zeros((col_p, 1))                    # z=0
z_sup = to_var(z_sup)


#%%Training loop
for epoch in range(epochs):

    if epoch == int(0.3*epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/3) # reduce learning rate
    elif epoch == int(0.7*epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/10) # reduce learning rate
        
    optimizer.zero_grad()               # set gradients to zero

    #Boundary loss
    phi_sup = model(r_sup,z_sup)  # Predict function values at boundary points
    sigma_r, sigma_theta, sigma_z, tau_rz = stresses(phi_sup,r_sup,z_sup)
    # Boundary condition at z=0
    #if z_col <= a: # within loaded area
    #    loss_sup1 = mse_(sigma_z + P, torch.zeros_like(sigma_z))  # sigma_z = -P for r<a, z=0
    #else:          # outside loaded area
    #    loss_sup1 = mse_(sigma_z, torch.zeros_like(sigma_z))      # sigma_z = 0 for r>a, z=0
    loss_sup1 = mse_(sigma_z, torch.zeros_like(sigma_z)) 
    loss_sup2 = mse_(tau_rz, torch.zeros_like(tau_rz))    # tau_rz = 0 for z=0
    
    #PDE loss
    f_col = model(r_col,z_col)  # Predict function values at collocation points
    loss_pde = mse_(biharmonic(f_col,r_col,z_col), torch.zeros_like(f_col))  # PDE loss
   
    loss = loss_pde + loss_sup1 + loss_sup2  # total loss

    Loss_pde = np.append(Loss_pde, to_np(loss_pde))
    Loss_sup1 = np.append(Loss_sup1, to_np(loss_sup1))
    Loss_sup2 = np.append(Loss_sup2, to_np(loss_sup2))

    loss.backward()                     # compute gradients
    optimizer.step()                    # update parameters

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Total loss: {loss.item()} | Loss PDE: {loss_pde.item()}, Loss BC1: {loss_sup1.item()}, Loss BC2: {loss_sup2.item()}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    # Print loss every 100 epochs

#%% Graphing loss
plt.semilogy(Loss_pde, label='Loss PDE')
plt.semilogy(Loss_sup1, label='Loss BC1')
plt.semilogy(Loss_sup2, label='Loss BC2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs')
plt.show()

# %%
