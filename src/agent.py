import torch, math
import torch.nn as nn
import torch.optim as optim

from utils import *
from utils import device, path
from resnet import *

class AgentEBM():

  def __init__(self, sensorimotor_system, embedding_size=32, action_size=2*20, association_lr=0.0001, action_lr=0.01, use_img_perspectives=False, referent_shape=10,
                     T_max=1000, use_temp=True, use_baseline=True):
    ##### PARAMETERS
    self.use_img_perspectives = use_img_perspectives
    self.sensorimotor_system  = sensorimotor_system
    self.action_size          = action_size
    self.action_lr            = action_lr
    self.meanReward           = 0
    self.playedGames          = 0
    self.temp                 = torch.tensor([0.07]).to(device)
    self.temp.requires_grad_(True)

    self.use_temp     = use_temp
    self.use_baseline = use_baseline

    ##### ENCODER A | REFERENTS
    if not use_img_perspectives:
        self.encoderA = nn.Sequential(nn.Linear(referent_shape, embedding_size))
    else:
        self.encoderA = nn.Sequential(nn.Conv2d(1,  8,  3, stride=2, padding=1), nn.ReLU(True),
                                      nn.Conv2d(8,  16, 3, stride=2, padding=1), nn.ReLU(True),
                                      nn.Conv2d(16, 32, 3, stride=2, padding=0), nn.ReLU(True),
                                      nn.Flatten(), nn.Linear(5408, 128), nn.ReLU(True), nn.Linear(128, embedding_size))

    ##### ENCODER B | UTTERANCES
    self.encoderB = nn.Sequential(nn.Conv2d(1,  8,  3, stride=2, padding=1), nn.ReLU(True),
                                  nn.Conv2d(8,  16, 3, stride=2, padding=1), nn.ReLU(True),
                                  nn.Conv2d(16, 32, 3, stride=2, padding=0), nn.ReLU(True),
                                  nn.Flatten(), nn.Linear(6*6*32, 128), nn.ReLU(True), nn.Linear(128, embedding_size))

    self.encoderA.to(device)
    self.encoderB.to(device)
    self.set_eval()

    ##### OPTIMIZER
    params = list(self.encoderA.parameters())+list(self.encoderB.parameters())+[self.temp]
    self.optimizer_AB = optim.Adam(params, lr=association_lr)

  def set_train(self):
    self.encoderA.train()
    self.encoderB.train()

  def set_eval(self):
    self.encoderA.eval()
    self.encoderB.eval()

  def cosine_sims(self, embeddingsA, embeddingsB):
    embeddingsA_normalized  = embeddingsA / torch.norm(embeddingsA,dim=-1).unsqueeze(-1)
    embeddingsB_normalized  = embeddingsB / torch.norm(embeddingsB,dim=-1).unsqueeze(-1)
    sims = torch.matmul(embeddingsA_normalized, embeddingsB_normalized.T) * (torch.exp(torch.clamp(self.temp,math.log(0.01),math.log(100))) if self.use_temp else 1)
    return sims

  def get_direct_utterance(self,referents,targets,iterations=100,nb_search=32,verbose=True,symmetric=True, discriminative=False):

      B = targets.shape[0]
      referents_embeddings = self.encoderA(referents).detach()

      ##### ACTIONS INITIALIZATION
      utterances = torch.rand(B*nb_search,1,52,52).to(device)
      utterances.requires_grad_(True)

      params    = [utterances]
      optimizer = optim.Adam(params,lr=self.action_lr)

      ##### GET HISTORY OPTION
      utt_history   = torch.empty(B, 0,52,52)
      coord_history = torch.empty(B, 0,20+2)

      ##### ACTIONS GENERATION
      if verbose:
        print(f"--- Generating DIRECT utterances for {targets.shape[0]} targets given a context of {referents.shape[0]} referents", flush=True)
        print(f"-- {nb_search} simultaneous search per target", flush=True)
        print(f"-- Targets : {targets}", flush=True)
      for i in range(iterations):
        self.optimizer_AB.zero_grad()####?
        utterances_embeddings = self.encoderB(utterances).reshape(B,nb_search,-1)
        sims                  = self.cosine_sims(utterances_embeddings, referents_embeddings)

        if not discriminative:
            loss                  = -sims[torch.arange(0,B),:,targets].mean()
        else:
            targets = (torch.ones(B,nb_search) * torch.arange(0,B).unsqueeze(1)).long().to(device)
            loss1   = torch.nn.CrossEntropyLoss()(sims.transpose(1,2),targets)/2
            loss2   = torch.nn.CrossEntropyLoss()(sims.transpose(0,1),targets.T)/2
            loss    = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      ##### ACTIONS SELECTION
      utterances_embeddings = self.encoderB(utterances).reshape(B,nb_search,-1)
      sims                  = self.cosine_sims(utterances_embeddings, referents_embeddings)
      if not discriminative:
          losses                = -sims[torch.arange(0,B),:,targets]
      else:
          loss1   = torch.nn.CrossEntropyLoss(reduction="none")(sims.transpose(1,2),targets)/2
          loss2   = torch.nn.CrossEntropyLoss(reduction="none")(sims.transpose(0,1),targets.T).T/2
          losses  = loss1+loss2

      best_idx              = torch.argmin(losses,1)
      best_losses           = losses[torch.arange(0,B),best_idx].detach().cpu()

      self.set_train()

      return utterances.reshape(B,nb_search,1,52,52)[torch.arange(0,B),best_idx],None, None, best_losses

  def get_actions(self,referents,targets,iterations=100,nb_search=32,verbose=True,symmetric=True, get_history=False, discriminative=False):
      self.set_eval()

      B = targets.shape[0]
      referents_embeddings = self.encoderA(referents).detach()

      ##### ACTIONS INITIALIZATION
      actions = torch.rand(B*nb_search,self.action_size).to(device)
      actions.requires_grad_(True)

      params    = [actions]
      optimizer = optim.Adam(params,lr=self.action_lr)

      ##### GET HISTORY OPTION
      utt_history   = torch.empty(B, 0,52,52)
      coord_history = torch.empty(B, 0,20+2)

      ##### ACTIONS GENERATION
      if verbose:
        print(f"--- Generating Actions for {targets.shape[0]} targets given a context of {referents.shape[0]} referents", flush=True)
        print(f"-- {nb_search} simultaneous search per target", flush=True)
        print(f"-- Targets : {targets}", flush=True)
      for i in range(iterations):
        self.optimizer_AB.zero_grad()####?
        utterances            = self.sensorimotor_system.get_utterances(actions)
        utterances_embeddings = self.encoderB(utterances).reshape(B,nb_search,-1)
        sims                  = self.cosine_sims(utterances_embeddings, referents_embeddings)

        if not discriminative:
            loss                  = -sims[torch.arange(0,B),:,targets].mean() + 0.1 * self.sensorimotor_system.out_loss
        else:
            targets = (torch.ones(B,nb_search) * torch.arange(0,B).unsqueeze(1)).long().to(device)
            loss1   = torch.nn.CrossEntropyLoss()(sims.transpose(1,2),targets)/2
            loss2   = torch.nn.CrossEntropyLoss()(sims.transpose(0,1),targets.T)/2
            loss    = loss1 + loss2 + 0.1 * self.sensorimotor_system.out_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if get_history:
            utt_history   = torch.cat((utt_history, utterances[:,0,:,:].reshape(B,nb_search,52,52).detach().cpu()),dim=1)
            coord_history = torch.cat((coord_history,self.sensorimotor_system.pts.reshape(B,nb_search,-1).detach().cpu()),dim=1)

      ##### ACTIONS SELECTION
      utterances            = self.sensorimotor_system.get_utterances(actions)
      utterances_embeddings = self.encoderB(utterances).reshape(B,nb_search,-1)
      sims                  = self.cosine_sims(utterances_embeddings, referents_embeddings)
      if not discriminative:
          losses                = -sims[torch.arange(0,B),:,targets]
      else:
          loss1   = torch.nn.CrossEntropyLoss(reduction="none")(sims.transpose(1,2),targets)/2
          loss2   = torch.nn.CrossEntropyLoss(reduction="none")(sims.transpose(0,1),targets.T).T/2
          losses  = loss1+loss2

      best_idx              = torch.argmin(losses,1)
      best_losses           = losses[torch.arange(0,B),best_idx].detach().cpu()

      self.set_train()

      if get_history:
          utt_history = torch.cat((utt_history, utterances[:,0,:,:].reshape(B,nb_search,52,52).detach().cpu()),dim=1)
          coord_history = torch.cat((coord_history,self.sensorimotor_system.pts.reshape(B,nb_search,-1).detach().cpu()),dim=1)

          return utt_history, coord_history
      else:
          return utterances.reshape(B,nb_search,1,52,52)[torch.arange(0,B),best_idx], actions.reshape(B,nb_search,-1)[torch.arange(0,B),best_idx], self.sensorimotor_system.pts.reshape(B,nb_search,-1)[torch.arange(0,B),best_idx], best_losses

  def get_referentSelections(self,referents,utterances):
    self.set_eval()

    utterances_embeddings = self.encoderB(utterances)
    referents_embeddings  = self.encoderA(referents)
    sims                  = self.cosine_sims(utterances_embeddings, referents_embeddings)
    return torch.argmax(sims, dim=1)


  def update_speaker(self, referents, targets, utterances, rewards, ood=False):
    self.set_train()

    ##### BASELINE
    if self.use_baseline:
        if self.playedGames == 0 : baseline = 0
        else:                      baseline = self.meanReward/self.playedGames
    else:
        baseline = 0

    self.meanReward  += torch.mean(rewards)
    self.playedGames += 1

    ### ASSOCIATIONS UPDATES
    B,N = referents.shape[0], 32
    if not self.use_img_perspectives:
        referents_embeddings  = self.encoderA(referents.to(device))
        utterances_embeddings = self.encoderB(utterances.detach())
        sims                  = self.cosine_sims(referents_embeddings,utterances_embeddings)
        targets = torch.arange(0,B).long().to(device)
        loss1   = torch.nn.CrossEntropyLoss(reduction="none")(sims,targets)/2
        loss2   = torch.nn.CrossEntropyLoss(reduction="none")(sims.T,targets)/2
        loss    = torch.matmul(loss1,(rewards-baseline)).mean() / (rewards.shape[0]) + torch.matmul(loss2,(rewards-baseline)).mean() / (rewards.shape[0])
    else:
        referents_imgs        = convert_to_imgs(referents,N,ood=ood)
        referents_embeddings  = self.encoderA(referents_imgs.reshape(-1,1,112,112)).reshape(B,N,-1)
        utterances_embeddings = self.encoderB(utterances.detach())
        sims                  = self.cosine_sims(referents_embeddings,utterances_embeddings)
        targets = (torch.ones(B,N) * torch.arange(0,B).unsqueeze(1)).long().to(device)
        loss1   = torch.nn.CrossEntropyLoss(reduction="none")(sims.transpose(1,2),targets)/2
        loss2   = torch.nn.CrossEntropyLoss(reduction="none")(sims.transpose(0,1),targets.T)/2
        loss    = torch.matmul(loss1.T,(rewards-baseline)).mean() / (rewards.shape[0]) + torch.matmul(loss2,(rewards-baseline)).mean() / (rewards.shape[0])

    self.optimizer_AB.zero_grad()
    loss.backward()
    self.optimizer_AB.step()

  def update_listener(self, referents, targets, utterances, ood=False):
    self.set_train()

    N,B                   = referents.shape[0], 32

    if not self.use_img_perspectives:
      referents_embeddings  = self.encoderA(referents)
      utterances_embeddings = self.encoderB(utterances.detach())
      sims                  = self.cosine_sims(referents_embeddings,utterances_embeddings)
      targets = torch.arange(0,N).long().to(device)
      loss1   = torch.nn.CrossEntropyLoss()(sims,targets)/2
      loss2   = torch.nn.CrossEntropyLoss()(sims.T,targets)/2
      loss    = loss1 + loss2
    else:
      referents_imgs        = convert_to_imgs(referents,B,ood=ood)
      referents_embeddings  = self.encoderA(referents_imgs.reshape(-1,1,112,112)).reshape(N,B,-1)
      utterances_embeddings = self.encoderB(utterances.detach())
      sims    = self.cosine_sims(referents_embeddings,utterances_embeddings)
      targets = (torch.ones(N,B) * torch.arange(0,N).unsqueeze(1)).long().to(device)
      loss    = torch.nn.CrossEntropyLoss()(sims.transpose(1,2),targets)/2 + torch.nn.CrossEntropyLoss()(sims.transpose(0,1),targets.T)/2

    self.optimizer_AB.zero_grad()
    loss.backward()
    self.optimizer_AB.step()
