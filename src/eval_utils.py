from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch, os, scipy, gc
import numpy as np

from display import *
from train import *
from agent import *
from utils import *
from data import *
from ss import *

''' OBJECT DETECTION METRICS '''

def precision(pred,y):
  return torch.sum(pred[y.bool()]) / torch.sum(pred)

def recall(pred,y):
  return torch.sum(pred[y.bool()]) / torch.sum(y)

def f1_score(pred,y):
  prec = precision(pred,y)
  rec  = recall(pred,y)

  if (prec == 0 or rec==0):
      return 0
  else:
      return 2 * (prec*rec) / (prec+rec)

''' COMPOSITIONAL REFERENT -> STRING KEY 'R[f1,f2...]' '''

def ref_str(ref):
    return "R"+str((ref==1).nonzero(as_tuple=True)[0].tolist())

''' PAIRWISE COHERENCE DISTANCES '''

def mean_pairwise_shape_sim(utterances_coords):
  N, losses = utterances_coords.shape[0], []
  for i in range(N):
    for j in range(i,N):
      if i != j:
        loss = scipy.spatial.distance.directed_hausdorff(utterances_coords[i].reshape(-1,2).detach(),utterances_coords[j].reshape(-1,2).detach())[0]
        losses.append(loss)
  return np.mean(losses)

''' POPULATION EVALUATION '''

def eval_population(agents, dataset, eval_set_idxs, nb_epochs = 100, use_p = True, shared_p = False, n=100, gen="descriptive",config=None):
    str_eval_dataset = [ref_str(dataset[eval_set_idxs][i]) for i in range(dataset[eval_set_idxs].shape[0])]
    print(f"---------- Evaluating a population of {len(agents)} agents in {nb_epochs} epochs.", flush=True)
    print(f"---------- Eval Dataset:\n{str_eval_dataset}", flush=True)

    results = {}

    for i in range(len(agents)):
        dl = DataLoader(eval_set_idxs,batch_size=32,shuffle=False)
        print(f"----- Evaluating Agent {i}", flush=True)

        results[i] = {"auto":{},"social":{},"refs":[],"utts":torch.empty(0,52,52),"utts_coords":torch.empty(0,20+2),"reps_refs":torch.empty(0,32),"reps_utts":torch.empty(0,32),"cosines":[]}
        for key in ["auto","social"]:
            results[i][key]["outcomes"]      = []
            results[i][key]["precisions"]    = []
            results[i][key]["recalls"]       = []
            results[i][key]["f1-scores"]     = []
            results[i][key]["failure-cases"] = {"speaker_targets":[], "listener_targets":[], "listener_choices":[]}

        for e in range(nb_epochs):
            print(f"- Epoch {e+1}/{nb_epochs}...", flush=True)

            ### Select Speaker & Listener
            speaker     = agents[i]
            agents_idxs = list(range(len(agents)))
            agents_idxs.remove(i)
            listener = agents[np.random.choice(agents_idxs)]

            ### Evaluate over eval dataset
            for batch_idxs in dl:
                gc.collect()
                batch_refs = dataset[batch_idxs]

                ### If we use perspectives, convert referents into images.
                if use_p:
                    context_p = convert_to_imgs(batch_refs)                     # Listener's Context perspective
                    if shared_p: batch_refs_p = context_p                       # Shared Speaker's Target perspective
                    else:        batch_refs_p = convert_to_imgs(batch_refs)     # Unshared Speaker's Target perspective
                else:
                    context_p    = batch_refs.to(device)
                    batch_refs_p = batch_refs.to(device)

                ### Get Speaker's utterances
                targets = torch.arange(0,batch_refs.shape[0])
                if config['no_ss']:
                    utterances, _, coords, losses = speaker.get_direct_utterance(batch_refs_p, targets, iterations=n,
                                                                        nb_search=64, verbose=False)
                else:
                    utterances,_,coords,losses = speaker.get_actions(batch_refs_p, targets, iterations=n, nb_search=64, verbose=False, discriminative=(gen=="discriminative"))
                utterances, losses    = utterances.detach().cpu(), losses.detach().cpu()

                ### Get Referents keys & Refs/Utts Embeddings
                with torch.no_grad():
                    embeddings_refs, embeddings_utts = speaker.encoderA(batch_refs_p.to(device)).detach().cpu(), speaker.encoderB(utterances.to(device)).detach().cpu()
                    results[i]["reps_refs"]   = torch.cat((results[i]["reps_refs"],embeddings_refs))
                    results[i]["reps_utts"]   = torch.cat((results[i]["reps_utts"],embeddings_utts))
                    results[i]["refs"]       += [ref_str(ref) for ref in batch_refs]
                    results[i]["utts"]        = torch.cat((results[i]["utts"], utterances[:,0,:,:]))
                    results[i]["utts_coords"] = torch.cat((results[i]["utts_coords"], coords))
                    results[i]["cosines"]    += [-loss for loss in losses.detach().cpu().tolist()]

                ### Compute performances
                for key in ["auto", "social"]:

                    ### Get Listener's choices
                    agent       = speaker if (key == "auto") else listener
                    choices     = agent.get_referentSelections(context_p.to(device), utterances.to(device)).detach().cpu()
                    chosen_refs = batch_refs[choices]

                    ### Evaluate choices
                    results[i][key]["outcomes"]    += (choices==targets).long().tolist()
                    results[i][key]["precisions"]  += [precision(chosen_refs[j].long(),batch_refs[j].long()) for j in range(batch_refs.shape[0])]
                    results[i][key]["recalls"]     += [recall(chosen_refs[j].long(),batch_refs[j].long()) for j in range(batch_refs.shape[0])]
                    results[i][key]["f1-scores"]   += [f1_score(chosen_refs[j].long(),batch_refs[j].long()) for j in range(batch_refs.shape[0])]

                    ### Record Failures
                    failure_mask = (choices!=targets).detach().cpu()
                    fail_shown   = batch_refs_p[failure_mask].detach().cpu()
                    fail_expect  = context_p[failure_mask].detach().cpu()
                    fail_chosen  = context_p[choices][failure_mask].detach().cpu()
                    fail_utt     = utterances[failure_mask].detach().cpu()

                    results[i][key]["failure-cases"]["speaker_targets"]  += [fail_shown[i] for i in range(fail_shown.shape[0])]
                    results[i][key]["failure-cases"]["listener_targets"] += [fail_expect[i] for i in range(fail_expect.shape[0])]
                    results[i][key]["failure-cases"]["listener_choices"] += [fail_chosen[i] for i in range(fail_chosen.shape[0])]
    print("\nDone! :)", flush=True)
    return results

def eval_population_alation(agents, dataset, eval_set_idxs, nb_epochs = 100, use_p = True, shared_p = False, n=100, gen="descriptive",config=None):
    str_eval_dataset = [ref_str(dataset[eval_set_idxs][i]) for i in range(dataset[eval_set_idxs].shape[0])]
    print(f"---------- Evaluating a population of {len(agents)} agents in {nb_epochs} epochs.", flush=True)
    print(f"---------- Eval Dataset:\n{str_eval_dataset}", flush=True)

    results = {}

    for i in range(len(agents)):
        dl = DataLoader(eval_set_idxs,batch_size=32,shuffle=False)
        print(f"----- Evaluating Agent {i}", flush=True)

        results[i] = {"auto":{},"social":{},"refs":[],"utts":torch.empty(0,52,52),"utts_coords":torch.empty(0,20+2),"reps_refs":torch.empty(0,32),"reps_utts":torch.empty(0,32),"cosines":[]}
        for key in ["auto","social"]:
            results[i][key]["outcomes"]      = []
            results[i][key]["precisions"]    = []
            results[i][key]["recalls"]       = []
            results[i][key]["f1-scores"]     = []
            results[i][key]["failure-cases"] = {"speaker_targets":[], "listener_targets":[], "listener_choices":[]}

        for e in range(nb_epochs):
            print(f"- Epoch {e+1}/{nb_epochs}...", flush=True)

            ### Select Speaker & Listener
            speaker     = agents[i]
            agents_idxs = list(range(len(agents)))
            agents_idxs.remove(i)
            listener = agents[np.random.choice(agents_idxs)]

            ### Evaluate over eval dataset
            for batch_idxs in dl:
                gc.collect()
                batch_refs = dataset[batch_idxs]

                ### If we use perspectives, convert referents into images.
                if use_p:
                    context_p = convert_to_imgs(batch_refs)                     # Listener's Context perspective
                    if shared_p: batch_refs_p = context_p                       # Shared Speaker's Target perspective
                    else:        batch_refs_p = convert_to_imgs(batch_refs)     # Unshared Speaker's Target perspective
                else:
                    context_p    = batch_refs.to(device)
                    batch_refs_p = batch_refs.to(device)

                ### Get Speaker's utterances
                targets = torch.arange(0,batch_refs.shape[0])
                if config['no_ss']:
                    utterances, _, _, losses = speaker.get_direct_utterance(batch_refs_p, targets, iterations=n,
                                                                        nb_search=64, verbose=False)
                else:
                    raise ValueError("Not the right function")

                utterances, losses    = utterances.detach().cpu(), losses.detach().cpu()

                ### Get Referents keys & Refs/Utts Embeddings
                with torch.no_grad():
                    embeddings_refs, embeddings_utts = speaker.encoderA(batch_refs_p.to(device)).detach().cpu(), speaker.encoderB(utterances.to(device)).detach().cpu()
                    results[i]["reps_refs"]   = torch.cat((results[i]["reps_refs"],embeddings_refs))
                    results[i]["reps_utts"]   = torch.cat((results[i]["reps_utts"],embeddings_utts))
                    results[i]["refs"]       += [ref_str(ref) for ref in batch_refs]
                    results[i]["utts"]        = torch.cat((results[i]["utts"], utterances[:,0,:,:]))

                    results[i]["cosines"]    += [-loss for loss in losses.detach().cpu().tolist()]

                ### Compute performances
                for key in ["auto", "social"]:

                    ### Get Listener's choices
                    agent       = speaker if (key == "auto") else listener
                    choices     = agent.get_referentSelections(context_p.to(device), utterances.to(device)).detach().cpu()
                    chosen_refs = batch_refs[choices]

                    ### Evaluate choices
                    results[i][key]["outcomes"]    += (choices==targets).long().tolist()
                    results[i][key]["precisions"]  += [precision(chosen_refs[j].long(),batch_refs[j].long()) for j in range(batch_refs.shape[0])]
                    results[i][key]["recalls"]     += [recall(chosen_refs[j].long(),batch_refs[j].long()) for j in range(batch_refs.shape[0])]
                    results[i][key]["f1-scores"]   += [f1_score(chosen_refs[j].long(),batch_refs[j].long()) for j in range(batch_refs.shape[0])]

                    ### Record Failures
                    failure_mask = (choices!=targets).detach().cpu()
                    fail_shown   = batch_refs_p[failure_mask].detach().cpu()
                    fail_expect  = context_p[failure_mask].detach().cpu()
                    fail_chosen  = context_p[choices][failure_mask].detach().cpu()
                    fail_utt     = utterances[failure_mask].detach().cpu()

                    results[i][key]["failure-cases"]["speaker_targets"]  += [fail_shown[i] for i in range(fail_shown.shape[0])]
                    results[i][key]["failure-cases"]["listener_targets"] += [fail_expect[i] for i in range(fail_expect.shape[0])]
                    results[i][key]["failure-cases"]["listener_choices"] += [fail_chosen[i] for i in range(fail_chosen.shape[0])]
    print("\nDone! :)", flush=True)
    return results


''' TOPOGRAPHIC CORRELATION '''

def get_topo(agents, P = 100, use_p = True, shared_p = False, n=100):
    speaker = agents[0]

    values_dA, values_dB, values_sims = torch.empty(0), torch.empty(0), torch.empty(0)
    for i in range(5):
        for j in range(i,5):
            if (i!=j):
                referent, featA, featB = torch.zeros(5), torch.zeros(5), torch.zeros(5)
                referent[[i,j]] = 1
                featA[i]        = 1
                featB[j]        = 1

                featA, featB, compo = ref_str()

                if use_p:
                    referent_p = convert_to_imgs(referent.unsqueeze(0))
                else:
                    referent_p = referent.unsqueeze(0)

                targets    = torch.tensor([0]).to(device)

                ### We record X (utterance, coord) pairs
                utts, coords = speaker.get_actions(referent_p, targets, iterations=n, nb_search=64, verbose=False, get_history=True)
                utts, coords = utts.squeeze(), coords.squeeze()

                X = utts.shape[0]

                ### D1 & D2 : X values
                dA, dB = torch.zeros(X), torch.zeros(X)
                feats  = torch.cat((featA.unsqueeze(0),featB.unsqueeze(0)))
                for p in range(P):

                    if use_p:
                        feats_p = convert_to_imgs(feats)
                    else:
                        feats_p = feats

                    targets      = torch.arange(0,2)
                    _,_,coords_feats,_ = speaker.get_actions(feats_p, targets, iterations=n, nb_search=64, verbose=False)
                    coords_featA, coords_featB = coords_feats[0], coords_feats[1]
                    for x in range(X):
                        dA[x] += scipy.spatial.distance.directed_hausdorff(coords[x].reshape(-1,2).detach().cpu(), coords_featA.reshape(-1,2).detach().cpu())[0] / P
                        dB[x] += scipy.spatial.distance.directed_hausdorff(coords[x].reshape(-1,2).detach().cpu(), coords_featB.reshape(-1,2).detach().cpu())[0] / P

                ### COS
                sims            = torch.zeros(X)
                referent_p      = convert_to_imgs(referent.unsqueeze(0),P)[0]
                ref_embeddings  = speaker.encoderA(referent_p.to(device))

                batch_size = 256
                batch_nb   = math.ceil(X/256)
                for b in range(batch_nb):
                    batch_utts      = utts[b*batch_size:b*batch_size+batch_size]
                    utts_embeddings = speaker.encoderB(batch_utts.unsqueeze(1).to(device))
                    cosines = speaker.cosine_sims(ref_embeddings, utts_embeddings).detach().cpu()
                    sims[b*batch_size:b*batch_size+batch_size] = torch.mean(cosines,dim=0)

                values_dA   = torch.cat((values_dA,dA))
                values_dB   = torch.cat((values_dB,dB))
                values_sims = torch.cat((values_sims,sims))

    return values_sims, values_dA, values_dB

def get_topo_per_compo(results_basic, results_compo):
    results_basic, results_compo = results_basic[0], results_compo[0]

    results_topo = {}

    for i in range(5):
        for j in range(i,5):
            if i != j:
                values_dA, values_dB, colors = [], [], []

                featA, featB, compo  = torch.zeros(5), torch.zeros(5), torch.zeros(5),
                featA[i]     = 1
                featB[j]     = 1
                compo[[i,j]] = 1
                strA, strB, strCompo = ref_str(featA), ref_str(featB), ref_str(compo)

                maskA = (np.array(results_basic["refs"]) == strA)
                uttsA = results_basic["utts_coords"][maskA]

                maskB = (np.array(results_basic["refs"]) == strB)
                uttsB = results_basic["utts_coords"][maskB]

                uttsCompo = results_compo["utts_coords"]

                for k in range(uttsCompo.shape[0]):
                    values_dA.append(0)
                    values_dB.append(0)

                    fA, fB = results_compo["refs"][k][2], results_compo["refs"][k][5]

                    if fA in strCompo and fB in strCompo: colors.append(3)
                    elif (fA in strA or fB in strA):      colors.append(2)
                    elif (fA in strB or fB in strB):      colors.append(1)
                    else:                                 colors.append(0)

                    for p in range(uttsA.shape[0]):
                        values_dA[-1] += scipy.spatial.distance.directed_hausdorff(uttsCompo[k].reshape(-1,2).detach().cpu(), uttsA[p].reshape(-1,2).detach().cpu())[0] / uttsA.shape[0]
                        values_dB[-1] += scipy.spatial.distance.directed_hausdorff(uttsCompo[k].reshape(-1,2).detach().cpu(), uttsB[p].reshape(-1,2).detach().cpu())[0] / uttsA.shape[0]

                results_topo[strCompo] = {}
                results_topo[strCompo]["dA"] = values_dA
                results_topo[strCompo]["dB"] = values_dB
                results_topo[strCompo]["colors"] = colors

    return results_topo
''''''''''''''' SPECIFIC RESULTS '''''''''''''''''''''''''''''''''''''''''''''

''' GET LEXICON '''
def get_lexicon_example(results):
    referents        = results[0]["refs"]
    unique_referents = np.unique(results[0]["refs"]).tolist()
    nb_agents        = len(list(results.keys()))
    lexicon          = torch.zeros(nb_agents, len(unique_referents), 52, 52)

    for i in range(nb_agents):
        for j,ref in enumerate(unique_referents):
            mask           = (np.array(referents) == ref)
            max_cosine_idx = np.argmax(np.array(results[i]["cosines"])[mask])
            lexicon[i][j]  = results[i]["utts"][mask][max_cosine_idx]

    return lexicon

''' GET A-COHERENCE, P-COHERENCE, R-COHERENCE '''
def get_coherences(results):
    referents        = results[0]["refs"]
    unique_referents = np.unique(results[0]["refs"]).tolist()

    nb_perspectives = int(len(referents) / len(unique_referents))

    a_coherences, p_coherences, r_coherences = [], [], []
    for ref in unique_referents:
        ref_mask = (np.array(referents) == ref)

        ### P-COHERENCE
        for i in results.keys():
            coord_set = results[i]["utts_coords"][ref_mask]
            p_coherences.append(mean_pairwise_shape_sim(coord_set))

        ### A-COHERENCE
        for j in range(ref_mask.sum()):
            coord_set = torch.empty(0,20 + 2)
            for i in results.keys():
                coord_set = torch.cat((coord_set,results[i]["utts_coords"][ref_mask][j].unsqueeze(0)))
            a_coherences.append(mean_pairwise_shape_sim(coord_set))

    ### R-COHERENCE
    for p in range(nb_perspectives):
        for i in results.keys():
            coord_set = torch.empty(0,20 + 2)
            for ref in unique_referents:
                ref_mask  = (np.array(referents) == ref)
                coord_set = torch.cat((coord_set,results[i]["utts_coords"][ref_mask][p].unsqueeze(0)))
            r_coherences.append(mean_pairwise_shape_sim(coord_set))

    return a_coherences, p_coherences, r_coherences

''' GET MEAN PERFORMANCES '''

def get_performances(results, type="social"):
    success, precision, recall, f1 = [], [], [], []
    for i in results.keys():
        success   += results[i][type]["outcomes"]
        precision += results[i][type]["precisions"]
        recall    += results[i][type]["recalls"]
        f1        += results[i][type]["f1-scores"]
    return success, precision, recall, f1
