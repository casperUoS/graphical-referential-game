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
    self.n_bfs     = params["n_bfs"]              # Nb of DMP weights
    self.dt        = params["dt"]                 # Delta Time
    self.n         = params["n"]                  # Nb of points in trajectory
    self.d         = params["d"]                  # Image size
    self.th        = params["th"]                 # Drawing thickness
    self.n_strokes = params["n_strokes"]   # Nb of strokes per utterance
    self.p         = 500                          # Weights range [-p, p]
    T              = self.n * self.dt             # Total trajectory time

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

    B = actions.shape[0]   # Batch size

    # actions shape: [B, n_strokes * 2 * n_bfs]
    # Split into n_strokes chunks, each of size 2*n_bfs
    stroke_actions = actions.reshape(B, self.n_strokes, -1)  # [B, n_strokes, 2*n_bfs]

    all_out_loss = []
    all_pts      = []
    stroke_imgs  = []

    for s in range(self.n_strokes):
        ##### TRAJECTORIES
        w = stroke_actions[:, s, :]       # [B, 2*n_bfs]
        w = w.reshape(B * 2, -1)          # [B*2, n_bfs]
        w = (w * 2 - 1) * self.p          # Normalize in [-p, p]
        self.dmp.w = w
        self.dmp.reset()
        trajectories = torch.zeros(B * 2, self.dmp.cs.N + 1)
        for i in range(1, self.dmp.cs.N + 1):
            trajectories[:, i], _, _, _ = self.dmp.step(k=1.3)
        trajectories_x, trajectories_y = trajectories[:B], trajectories[B:]

        ##### DISPLAY WINDOW
        trajectories_x2 = trajectories_x / 10 + 0.5
        trajectories_y2 = trajectories_y / 10 + 0.5

        # Compute a loss for points outside of view
        outL = trajectories_x2[trajectories_x2 <= 0]
        outR = trajectories_x2[trajectories_x2 >= 1] - 1
        outB = trajectories_y2[trajectories_y2 <= 0]
        outT = trajectories_y2[trajectories_y2 >= 1] - 1
        out_loss = (torch.norm(outL)**2 + torch.norm(outR)**2 +
                    torch.norm(outB)**2 + torch.norm(outT)**2) / 4
        all_out_loss.append(out_loss)

        ##### DRAWING
        pts     = (torch.stack((trajectories_x2, trajectories_y2), 2).reshape(B, -1, 2)) * 2 - 1
        pts     = pts.to(device)
        npoints = pts.shape[1]
        all_pts.append(pts)

        # Compute all valid consecutive line segments
        self.coordpairs = torch.stack([torch.arange(0, npoints - 1, 1),
                                       torch.arange(1, npoints,1)], dim=1).to(device)
        lines = torch.stack((pts[:, self.coordpairs[:, 0]],
                              pts[:, self.coordpairs[:, 1]]), dim=-2).to(device)

        # Differentiable rasterization
        rasters = exp(line_edt2(lines, self.grid), self.th)
        stroke_imgs.append(softor(rasters))   # [B, d, d]

    # Composite all strokes with softor (per-pixel max)
    imgs = stroke_imgs[0]
    for s in range(1, self.n_strokes):
        imgs = softor(torch.stack([imgs, stroke_imgs[s]], dim=1))

    # Update some properties
    self.out_loss = sum(all_out_loss) / self.n_strokes
    self.pts      = torch.stack(all_pts)  # list of [B, npoints, 2] per stroke

    return imgs.unsqueeze(1)

##### AVAILABLE SS IMPLEMENTATIONS #############################################
available_ss = {"dmp":DMP_SensorimotorSystem}
