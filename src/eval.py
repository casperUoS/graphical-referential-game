from eval_utils import *
import warnings

warnings.filterwarnings("ignore")

local = (path == "/home/ylemesle/Bureau/ebng/Code/ebng/src/eval_framework/")

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def eval(exp_name, seed, no_ss, P=10, N=100):
    print(f"##### EVALUATION - EXPERIMENT {exp_name} - SEED {seed}", flush=True)

    ''' PATH TO RESULTS '''
    seed_path = path + "/results/" + exp_name + "/seed" + str(seed) + "/"
    path_eval = seed_path + "/Eval/"
    print(f"--------------- Recording results in {path_eval}", flush=True)

    ''' AGENTS EVALUATION '''
    if not os.path.exists(path_eval + "eval.pt"):

        ''' LOAD RESULTS '''
        eval_dict = {}
        exp, config = load_exp(seed_path)

        ''' SET SEED '''
        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ''' COMPOSITIONS BINS '''
        bins = list(range(1, config["n_features"] + 1))

        ''' DATASET & BINS INDEXES'''
        dataset = torch.load(generate_systematic_dataset(n=config["n_features"], m_list=bins))
        dataset_idxs = torch.arange(0, dataset.shape[0])
        bin_idxs = {}
        for bin in bins:
            bin_mask = (torch.sum(dataset, 1) == bin)
            bin_idxs[bin] = dataset_idxs[bin_mask]

        transfer_idxs = None
        if "transfer_refs" in config.keys() and None is not config["transfer_refs"]:
            idxs_compo = bin_idxs[2].tolist()
            transfer_idxs = []
            for idx in idxs_compo:
                if not (dataset[idx] in config["transfer_refs"].tolist()):
                    transfer_idxs.append(idx)

        ''' BASIC HISTORY '''

        print(f"### BASIC HISTORY...", flush=True)

        eval_dict["train_outcomes"] = exp["logs"]["graph_outcomes"]  # Basic Games outcomes (1 per iteration)

        eval_dict["history_basic"] = {}
        eval_dict["history_basic"]["cP"] = []  # Basic P-Coherences (1 per 500 iteration)
        eval_dict["history_basic"]["cA"] = []  # Basic A-Coherences (1 per 500 iteration)
        eval_dict["history_basic"]["cR"] = []  # Basic R-Coherences (1 per 500 iteration)
        eval_dict["history_basic"]["lexicon"] = []  # Lexicons           (1 per 500 iteration)

        steps = 1000 if config["use_img_perspectives"] else 100

        eval_dict["steps"] = steps

        if steps == 100:
            print("ONE-HOT EVAL", flush=True)
            N = 1

        nb_iterations = (len(eval_dict["train_outcomes"]) // steps) * steps
        for i in range(0, (nb_iterations + 1 if not local else (steps + 1)), steps):
            print(f"> Iteration {i} / {nb_iterations}", flush=True)

            agents = load_history(seed_path, config, i)[:10]  # Only evaluate a subset of 10 agents maximum

            ### Basic
            results_basic_h = eval_population(agents, dataset, bin_idxs[1], nb_epochs=P,
                                              use_p=config["use_img_perspectives"],
                                              shared_p=config["shared_perspective"], n=N, no_ss=no_ss)
            lexicon = get_lexicon_example(results_basic_h)
            cA, cP, cR = get_coherences(results_basic_h)

            ### Update history
            eval_dict["history_basic"]["cP"].append(cP)
            eval_dict["history_basic"]["cA"].append(cA)
            eval_dict["history_basic"]["cR"].append(cR)
            eval_dict["history_basic"]["lexicon"].append(lexicon)

        ### Last Population Results
        eval_dict["results_basic"] = results_basic_h
        ####################################################################################################################

        results_basic = eval_dict["results_basic"]

        eval_dict["descriptive"] = {}
        eval_dict["discriminative"] = {}

        for key1 in ["descriptive", "discriminative"]:

            ''' COMPO (2-feats) PERFORMANCES '''
            eval_dict[key1]["results_compo"] = {"auto": {}, "social": {}, "utts": None}

            if None is transfer_idxs:
                results_compo = eval_population(agents, dataset, bin_idxs[2], nb_epochs=P,
                                                use_p=config["use_img_perspectives"],
                                                shared_p=config["shared_perspective"], n=N, gen=key1)
            else:
                results_compo = eval_population(agents, dataset, torch.tensor(transfer_idxs), nb_epochs=P,
                                                use_p=config["use_img_perspectives"],
                                                shared_p=config["shared_perspective"], n=N, gen=key1)

            for key2 in ["auto", "social"]:
                o, p, r, f1 = get_performances(results_compo, type=key2)
                eval_dict[key1]["results_compo"][key2]["o"] = o  # Compo Mean Outcomes
                eval_dict[key1]["results_compo"][key2]["p"] = p  # Compo Mean Precision
                eval_dict[key1]["results_compo"][key2]["r"] = r  # Compo Mean Recall
                eval_dict[key1]["results_compo"][key2]["f1"] = f1  # Compo Mean F1

            ''' COMPO (2-feats) FAILURE CASES (of agent 0) '''
            eval_dict[key1]["compo_fails"] = {"auto": results_compo[0]["auto"]["failure-cases"],
                                              "social": results_compo[0]["social"]["failure-cases"]}

            ''''''''' ADDITIONAL FIGURES (for agent 0) '''''''''

            eval_dict[key1]["results_compo"]["utts"] = results_compo[0]["utts"]
            eval_dict[key1]["results_compo"]["refs"] = results_compo[0]["refs"]

            ''' COMPO (2-feats) MATRIX (agent 0)'''
            print("### COMPOSITIONAL MATRIX...", flush=True)

            matrix = torch.zeros(5, 5, 52, 52)
            matrix_refs = []
            for i in range(5):
                for j in range(5):
                    referent = torch.zeros(5)
                    referent[i] = 1
                    referent[j] = 1
                    ref = ref_str(referent)

                    if (i == j):
                        matrix_refs.append(ref)
                        mask = (np.array(results_basic[0]["refs"]) == ref)
                        max_cosine_idx = np.argmax(np.array(results_basic[0]["cosines"])[mask])
                        matrix[i][j] = results_basic[0]["utts"][mask][max_cosine_idx]
                    else:
                        mask = (np.array(results_compo[0]["refs"]) == ref)
                        max_cosine_idx = np.argmax(np.array(results_compo[0]["cosines"])[mask])
                        matrix[i][j] = results_compo[0]["utts"][mask][max_cosine_idx]

            eval_dict[key1]["compo_matrix"] = {}
            eval_dict[key1]["compo_matrix"]["utts"] = matrix
            eval_dict[key1]["compo_matrix"]["refs"] = matrix_refs

            ''' TOPOGRAPHY CORRELATION '''
            print("### TOPOGRAPHIC UTTERANCES ANALYSIS...", flush=True)

            eval_dict[key1]["topography_corr"] = get_topo_per_compo(results_basic, results_compo)

            ''' TSNEs - BASICS & COMPO (2-feats) (agent 0)'''
            print("### EMBEDDINGS T-SNEs...", flush=True)

            ### BIN 1
            unique_referents_basic = np.unique(results_basic[0]["refs"])
            embeddings_refs_basic, embeddings_utts_basic = results_basic[0]["reps_refs"], results_basic[0]["reps_utts"]

            tsne_basics = {"refs": None, "utts": None, "colors": None}

            tsne_basics["refs"] = TSNE().fit_transform(embeddings_refs_basic.numpy())
            tsne_basics["utts"] = TSNE().fit_transform(embeddings_utts_basic.numpy())
            tsne_basics["colors"] = [np.where(unique_referents_basic == ref)[0][0] for ref in results_basic[0]["refs"]]
            tsne_basics["str"] = results_basic[0]["refs"]

            ### BIN 2
            unique_referents_compo = np.unique(results_compo[0]["refs"])
            embeddings_refs_basic, embeddings_utts_basic = results_basic[0]["reps_refs"], results_basic[0]["reps_utts"]

            tsne_compos = {}
            for unique_ref in unique_referents_compo.tolist():
                tsne_compos[unique_ref] = {"refs": None, "utts": None, "colors": None}

                ref_mask = (np.array(results_compo[0]["refs"]) == unique_ref)
                embeddings_ref_compo = results_compo[0]["reps_refs"][ref_mask]
                embeddings_utt_compo = results_compo[0]["reps_utts"][ref_mask]

                tsne_compos[unique_ref]["refs"] = TSNE().fit_transform(
                    torch.cat((embeddings_refs_basic, embeddings_ref_compo)).numpy())
                tsne_compos[unique_ref]["utts"] = TSNE().fit_transform(
                    torch.cat((embeddings_utts_basic, embeddings_utt_compo)).numpy())
                tsne_compos[unique_ref]["colors"] = tsne_basics["colors"] + [-1] * np.sum(ref_mask)

            eval_dict[key1]["tsne_basics"] = tsne_basics
            eval_dict[key1]["tsne_compos"] = tsne_compos

        ''' SAVE RESULTS '''

        print(f"##### SAVING RESULTS IN {path_eval} ...", flush=True)
        torch.save(eval_dict, path_eval + "eval.pt")
        print("##### EVALUATION RESULTS SAVED! :)", flush=True)
    else:
        eval_dict = torch.load(path_eval + "eval.pt")

        exp, config = load_exp(seed_path)

        ''' SET SEED '''
        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ''' COMPOSITIONS BINS '''
        bins = list(range(1, config["n_features"] + 1))

        ''' DATASET & BINS INDEXES'''
        dataset = torch.load(generate_systematic_dataset(n=config["n_features"], m_list=bins))
        dataset_idxs = torch.arange(0, dataset.shape[0])
        bin_idxs = {}
        for bin in bins:
            bin_mask = (torch.sum(dataset, 1) == bin)
            bin_idxs[bin] = dataset_idxs[bin_mask]

        steps = 1000
        nb_iterations = (len(eval_dict["train_outcomes"]) // steps) * steps
        agents = load_history(seed_path, config, nb_iterations)[:10]  # Only evaluate a subset of 10 agents maximum

        results_basic = eval_dict["results_basic"]

        for key1 in ["descriptive"]:
            results_compo = eval_population(agents, dataset, bin_idxs[2], nb_epochs=P,
                                            use_p=config["use_img_perspectives"], shared_p=config["shared_perspective"],
                                            n=N, gen=key1)
            eval_dict[key1]["results_compo"]["utts"] = results_compo[0]["utts"]
            eval_dict[key1]["results_compo"]["refs"] = results_compo[0]["refs"]

        torch.save(eval_dict, path_eval + "eval.pt")
    display_eval(exp_name, seed)


def eval_ablation(exp_name, seed):
    seed_path = os.path.join(path, 'results', exp_name, "seed{}".format(str(seed)))
    path_eval = os.path.join(seed_path, "Eval")

    if not os.path.exists(path_eval):
        os.mkdir(path_eval)

    ''' LOAD RESULTS '''
    eval_dict = {}
    exp, config = load_exp(seed_path)

    ''' SET SEED '''
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ''' LOAD AGENTS '''
    steps = 1000 if config["use_img_perspectives"] else 100

    if steps == 100:
        N = 1
    else:
        N = 100

    nb_iterations = (len(exp["logs"]["graph_outcomes"]) // steps) * steps
    agents = load_history(seed_path, config, nb_iterations)

    ''' COMPUTE LEXICON '''
    data_path = generate_systematic_dataset(n=config["n_features"], m_list=[1])
    dataset = torch.load(data_path, map_location=device)

    if config["use_img_perspectives"]:
        perspectives = convert_to_imgs(dataset, ood=config["ood"])
    else:
        perspectives = dataset

    targets = torch.arange(0, 5).long().to(device)

    utts1, _, _, _ = agents[0].get_direct_utterance(perspectives, targets, iterations=config["action_it"],
                                                    nb_search=config["action_bs"], verbose=False)
    utts1 = utts1.permute(1, 0, 2, 3)
    utts2, _, _, _ = agents[1].get_direct_utterance(perspectives, targets, iterations=config["action_it"],
                                                    nb_search=config["action_bs"], verbose=False)
    utts2 = utts2.permute(1, 0, 2, 3)

    lexicon = torch.cat((utts1, utts2))

    display_lexicon(lexicon.detach().cpu(), ["R[0]", "R[1]", "R[2]", "R[3]", "R[4]"])
    plt.savefig(path_eval + "Lexicon_Instance.pdf")
    plt.close()

    ''' GEN PERFORMANCES '''
    # Generalization
    data_path = generate_systematic_dataset(n=config["n_features"], m_list=[1,2,3,4,5])
    dataset = torch.load(data_path, map_location=device)
    dataset_idxs = torch.arange(0, dataset.shape[0])
    bin_idxs = {}
    bins = list(range(1, config["n_features"] + 1))
    for bin in bins:
        bin_mask = (torch.sum(dataset, 1) == bin)
        bin_idxs[bin] = dataset_idxs[bin_mask]

    results = eval_population_alation(agents, dataset, bin_idxs[2], nb_epochs=100, n=100, use_p=config["use_img_perspectives"], config=config)

    auto_perf = (np.mean(results[0]["auto"]["outcomes"]) + np.mean(results[1]["auto"]["outcomes"])) / 2
    social_perf = (np.mean(results[0]["social"]["outcomes"]) + np.mean(results[1]["social"]["outcomes"])) / 2

    final_results = {"auto": None, "social": None}
    final_results["auto"] = auto_perf
    final_results["social"] = social_perf

    torch.save(final_results, path_eval + "Results_Compo.pt")


''' SEED AGGREGATION '''
'''
def seed_aggregation(exp_name):

    path_exp = path+exp_name

    eval_dict_agg = {}

    for seed_path in os.blabla:
        path_eval = seed_path+"/Eval/"


        if type(metr)

    return None
'''

if __name__ == "__main__":

    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        seed = 1

    if local:
        exp_name = "base-compo"
    else:
        exp_name = os.environ["SLURM_JOB_NAME"]

    ''' Evaluate Population '''
    eval(exp_name, seed)

'''
# COMPO PERFORMANCES (PER BIN)

print(f"### COMPOSITIONAL PERFORMANCES PER BIN...")

bin_performance = {"auto":{},"social":{}}
for key in ["auto", "social"]:
    bin_performance[key] = {bin:{} for bin in bins[1:]}

for bin in bins[1:]:
    print(f"> Bin {bin}")
    if bin == 2:    results_compo_bin = results_compo
    else:           results_compo_bin = eval_population(agents, dataset, bin_idxs[bin], nb_epochs = P, use_p = config["use_img_perspectives"], shared_p = config["shared_perspective"], n=N)
    for key in ["auto", "social"]:
        o, p, r, f1   = get_performances(results_compo_bin,type = key)
        bin_performance[key][bin]["o"]  = o
        bin_performance[key][bin]["p"]  = p
        bin_performance[key][bin]["r"]  = r
        bin_performance[key][bin]["f1"] = f1
eval_dict["compo_performance"] = bin_performance
'''
