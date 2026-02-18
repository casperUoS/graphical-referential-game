from dsketch.raster.disttrans import point_edt2, line_edt2, curve_edt2_bruteforce
from dsketch.utils.pyxdrawing import draw_points, draw_line_segments
from dsketch.raster.composite import softor
from dsketch.raster.raster import exp

import numpy as np
from dmp import *
import torch

from utils import device, path

import matplotlib.pyplot as plt


##### Abstact Class ############################################################
class SensorimotorSystem(object):
  def __init__(self):
    if type(self) is SensorimotorSystem:
        raise Exception('SensorimotorSystem is an abstract class and cannot be instantiated directly')

  def get_utterances(self, actions):
    raise NotImplementedError('subclasses must override getUtterance()!')

##### DMP SS ###################################################################
class DMP_SensorimotorSystem(SensorimotorSystem):
  def __init__(self, params):
    super().__init__()

    ##### PARAMETERS
    self.n_bfs = params["n_bfs"]    # Nb of DMP weights
    self.dt    = params["dt"]       # Delta Time
    self.n     = params["n"]        # Nb of points in trajectory
    self.d     = params["d"]        # Image size
    self.th    = params["th"]       # Drawing thickness
    self.p     = 500                # Weights range [-p, p]
    T          = self.n * self.dt   # Total trajectory time

    ##### COORDINATE GRID
    r, c            = torch.linspace(-1, 1, self.d).to(device), torch.linspace(-1, 1, self.d).to(device)
    self.grid       = torch.meshgrid(r, c)
    self.grid       = torch.stack(self.grid, dim=2).to(device)
    self.coordpairs = None

    ##### DMP INSTANCE
    self.dmp = DMP(T, self.dt, n_bfs=self.n_bfs, a=10, b=10/4, w=None, s=0, g=1)

    #####
    self.out_loss = None
    self.pts      = None

  def get_utterances(self, actions):

    ##### TRAJECTORIES
    B  = actions.shape[0]         # Batch size
    w  = actions.reshape(B*2,-1)  # Weights for x, y
    w  = (w*2-1) * self.p         # Normalize in [-p,p]
    self.dmp.w = w
    self.dmp.reset()
    trajectories = torch.zeros(B*2,self.dmp.cs.N+1)
    for i in range(1,self.dmp.cs.N+1):
        trajectories[:,i],_,_,_ = self.dmp.step(k=1.3)
    trajectories_x, trajectories_y = trajectories[:B], trajectories[B:]

    ##### DISPLAY WINDOW
    trajectories_x1 = trajectories_x/10
    trajectories_x2 = trajectories_x1+0.5
    trajectories_y1 = trajectories_y/10
    trajectories_y2 = trajectories_y1+0.5

    # Compute a loss for points outside of view
    outL, outR, outB, outT = trajectories_x2[trajectories_x2<=0], (trajectories_x2[trajectories_x2>=1]-1), trajectories_y2[trajectories_y2<=0], (trajectories_y2[trajectories_y2>=1]-1)
    out_loss          = (torch.norm(outL)**2 + torch.norm(outR)**2 + torch.norm(outB)**2 + torch.norm(outT)**2) / 4

    ##### DRAWING
    # Convert trajectories into coordinates in [-1,1] range
    pts     = (torch.stack((trajectories_x2,trajectories_y2),2).reshape(B,-1,2))*2-1
    pts     = pts.to(device)
    npoints = pts.shape[1]

    # Compute all valid permuations of line start and end points
    self.coordpairs = torch.stack([torch.arange(0,npoints-1,1), torch.arange(1,npoints,1)], dim=1).to(device)
    lines           = torch.stack((pts[:,self.coordpairs[:,0]], pts[:,self.coordpairs[:,1]]), dim=-2).to(device)

    # Differentiable rasterization

    rasters = exp(line_edt2(lines, self.grid), self.th)

    imgs = softor(rasters)

    # Update some properties
    self.out_loss = out_loss
    self.pts      = pts

    return imgs.unsqueeze(1)

##### AVAILABLE SS IMPLEMENTATIONS #############################################
available_ss = {"dmp":DMP_SensorimotorSystem}
