import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import path, device

import os

def display_legend(ax, legends, fontsize=16):
    for marker, color, label in legends:
        ax.scatter([],[],marker=marker,label=label,c=color,s=100)
    ax.legend(loc="upper right",fontsize=fontsize)

def display_graph(x,y,xlabel="",ylabel="",xlim=None,ylim=None,label=None,marker=None):
    plt.plot(x,y,marker=marker,label=label)
    if None is not xlim: plt.xlim(xlim[0],xlim[1])
    if None is not ylim: plt.ylim(ylim[0],ylim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def display_lexicon(lexicon, refs_str):
    nb_agents, nb_referents = lexicon.shape[0], lexicon.shape[1]
    ##### Display Lexicon
    fig, axis = plt.subplots(nb_agents+1,nb_referents+1,figsize=((1 + nb_referents)*2,nb_agents*2))
    for i in range(nb_referents):
        axis[0][0].text(0,0,"")
        axis[0][0].axis("off")

        for j in range(nb_agents):
            axis[j+1][0].text(0.5,0.5,f"A{j}")
            axis[j+1][0].axis("off")

            axis[0][i+1].text(0.3,0,f"{refs_str[i]}")
            axis[0][i+1].axis("off")

            show = torch.zeros(62,62)
            show[5:57,5:57] = lexicon[j][i]

            axis[j+1][i+1].imshow(show)
            axis[j+1][i+1].axis("off")


    return fig

def display_referent_coherence(utts_per_agent, ref):
    nb_perspectives = utts_per_agent[0].shape[0]
    nb_agents       = len(utts_per_agent)

    fig, axis = plt.subplots(nb_agents,nb_perspectives+1,figsize=((2 + nb_perspectives)*2,nb_agents*2))

    for i in range(nb_agents):
        axis[i][0].text(0.5,0.5,f"A{i}",fontsize=25)
        axis[i][0].axis("off")
        for j in range(nb_perspectives):
            show            = torch.zeros(62,62)
            show[5:57,5:57] = utts_per_agent[i][j]
            axis[i][j+1].imshow(show)
            axis[i][j+1].axis("off")

    return fig

def display_failures(failures, key_path):

    for key in ["auto","social"]:
        compo_fails = failures[key]

        nb = min(10,len(compo_fails["speaker_targets"]))

        if nb > 0:
            idxs      = np.random.choice(range(0,len(compo_fails["speaker_targets"])),nb,replace=False)
            s_targets = np.array(compo_fails["speaker_targets"])[idxs]
            l_targets = np.array(compo_fails["listener_targets"])[idxs]
            l_choices = np.array(compo_fails["listener_choices"])[idxs]

            fig, axis = plt.subplots(3,nb+1,figsize=((1 + nb)*2,4*2))
            axis[0,0].text(0.5,0.5,"Shown to\nListener")
            axis[0,0].axis("off")
            axis[1,0].text(0.5,0.5,"Speaker's\nTarget")
            axis[1,0].axis("off")
            axis[2,0].text(0.5,0.5,"Speaker's\nChoise")
            axis[2,0].axis("off")
            for i in range(nb):
                print(i)
                axis[0,i+1].imshow(s_targets[i][0])
                axis[0,i+1].axis("off")
                axis[1,i+1].imshow(l_targets[i][0])
                axis[1,i+1].axis("off")
                axis[2,i+1].imshow(l_choices[i][0])
                axis[2,i+1].axis("off")
        else:
            plt.figure()
            plt.text(0.5,0.5,"No fail cases!")
        plt.savefig(key_path+"5-Compo_Failures_"+key+".pdf")
        plt.close()

def display_matrix(compo_matrix, key_path):
    matrix = compo_matrix["utts"]
    refs   = compo_matrix["refs"]

    H, W = matrix.shape[0], matrix.shape[1]
    fig, axis = plt.subplots(H+1,W+1,figsize=((W+1)*2,(H+1)*2))
    axis[0][0].text(0.5,0.5,"")
    axis[0][0].axis("off")
    for i in range(1,H+1):
        axis[i][0].text(0.5,0.5,refs[i-1])
        axis[i][0].axis("off")
        axis[0][i].text(0.5,0.5,refs[i-1])
        axis[0][i].axis("off")

        for j in range(i+1,W+1):
            axis[i][j].axis("off")
        for j in range(1,i+1):

            axis[i][j].imshow(matrix[i-1][j-1])

            if i == j:
                axis[i][j].spines["top"].set_color("orange")
                axis[i][j].spines["top"].set_linewidth(4)
                axis[i][j].spines["bottom"].set_color("orange")
                axis[i][j].spines["bottom"].set_linewidth(4)
                axis[i][j].spines["left"].set_color("orange")
                axis[i][j].spines["left"].set_linewidth(4)
                axis[i][j].spines["right"].set_color("orange")
                axis[i][j].spines["right"].set_linewidth(4)
                axis[i][j].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labelright=False, labeltop=False)
            else:
                axis[i][j].axis("off")


    plt.savefig(key_path+"6-Compo_Matrix.pdf")
    plt.close()

################################################################################
##### DISPLAY EVALUATION FIGURES ###############################################
def display_eval(exp_name,seed):
    eval_path = path+"/results/"+exp_name+"/seed"+str(seed)+"/Eval/"
    eval_dict = torch.load(eval_path+"eval.pt", weights_only=False)

    if "steps" not in eval_dict.keys():
        eval_dict["steps"] = 1000

    for key in ["descriptive","discriminative"]:

        key_path = eval_path+key+"/"

        if not os.path.exists(key_path):
            os.mkdir(key_path)

        print("---------- Generating Figures...", flush=True)

        ''' POPULATION HISTORY | BASIC '''
        fig,ax = plt.subplots()

        outcomes, cA, cP, cR = eval_dict["train_outcomes"], eval_dict["history_basic"]["cA"], eval_dict["history_basic"]["cP"], eval_dict["history_basic"]["cR"]

        iter_100     = list(range(1, len(outcomes), 100))
        outcomes_100 = [np.mean(outcomes[max(0,i-100):i]) for i in iter_100]
        ax.plot(iter_100, outcomes_100, label="Success Rate", color="blue")
        ax.set_ylabel("Success (%)")
        ax.set_xlabel("Iterations (5 games)")
        ax.set_ylim(0,1.1)

        iter_500 = list(range(0, (len(cA)-1)*eval_dict["steps"]+1, eval_dict["steps"]))
        cA_means = [np.mean(cA[i]) for i in range(len(cA))]
        cP_means = [np.mean(cP[i]) for i in range(len(cP))]
        cR_means = [np.mean(cR[i]) for i in range(len(cR))]
        ax2 = ax.twinx()
        ax2.plot(iter_500, cA_means, label="A-Coherence", linestyle="--", color="green")
        ax2.plot(iter_500, cP_means, label="P-Coherence", linestyle="--", color="purple")
        ax2.plot(iter_500, cR_means, label="R-Coherence", linestyle=":",  color="red")
        ax2.set_ylabel("Mean Hausdorff Distance")

        fig.legend()
        plt.savefig(key_path+"1-Training_History_Basic.pdf")
        plt.close()

        ''' POPULATION HISTORY | COMPO '''

        for key2 in ["auto","social"]:
            compo_perf = eval_dict[key]["results_compo"][key2]
            o,p,r,f1   = np.mean(compo_perf["o"]), np.mean(compo_perf["p"]), np.mean(compo_perf["r"]), np.mean(compo_perf["f1"])

            plt.figure()
            plt.bar(["o","p","r","f1"],[o.mean(),p.mean(),r.mean(),f1.mean()])
            plt.ylim(0,1.1)
            plt.savefig(key_path+"2-Compo_Perf_"+key2+".pdf")
            plt.close()

        '''
        for key in ["auto", "social"]:
            history  = eval_dict["history_compo"][key]
            o,p,r,f1 = history["o"], history["p"], history["r"], history["f1"]
            o_means  = [np.mean(o[i]) for i in range(len(o))]
            p_means  = [np.mean(p[i]) for i in range(len(p))]
            r_means  = [np.mean(r[i]) for i in range(len(r))]
            f1_means = [np.mean(f1[i]) for i in range(len(f1))]

            plt.figure()
            plt.plot(iter_500, o_means,  label="Success Rate", linestyle="-",  color="blue")
            plt.plot(iter_500, f1_means, label="F1-Score",     linestyle="-",  color="green")
            plt.plot(iter_500, p_means,  label="Precision",    linestyle="--", color="orange")
            plt.plot(iter_500, r_means,  label="Recall",       linestyle=":", color="red")
            plt.ylim(0,1.1)
            plt.xlabel("Iterations (5 games)")
            plt.legend()
            plt.savefig(key_path+"2-Training_History_Compo_"+key+".pdf")
            plt.close()
        '''

        ''' SUCCESSFUL LEXICON EXAMPLE '''
        lexicon = eval_dict["history_basic"]["lexicon"][-1]

        print("LEXICON", flush=True)
        print(lexicon.shape, flush=True)
        print(lexicon[0,2], flush=True)
        print(lexicon[1,2], flush=True)
        print(lexicon[1,4], flush=True)

        fig     = display_lexicon(lexicon, ["R[0]","R[1]","R[2]","R[3]","R[4]"])
        plt.savefig(key_path+"3-Lexicon_Instance.pdf")
        plt.close()

        ''' LEXICON COHERENCE FIGURES '''
        results_basic = eval_dict["results_basic"]

        for ref in np.unique(results_basic[0]["refs"]):
            mask = (np.array(results_basic[0]["refs"]) == ref)

            utts_per_agent = []

            for a in results_basic.keys():
                utts = results_basic[a]["utts"][mask]
                utts_per_agent.append(utts[0:10])

            fig = display_referent_coherence(utts_per_agent, ref)
            plt.savefig(key_path+"4-Lexicon_Coherences_Examples_"+ref+".pdf")
            plt.close()


        ''' COMPO PERFORMANCES (PER BIN)'''

        '''
        for key in ["auto", "social"]:

            o_graph, p_graph, r_graph, f1_graph = [],[],[],[]
            for bin in [2,3,4,5]:
                bin = eval_dict["compo_performance"][key][bin]
                o,p,r,f1 = bin["o"], bin["p"], bin["r"], bin["f1"]
                o_graph.append(np.mean(o))
                p_graph.append(np.mean(p))
                r_graph.append(np.mean(r))
                f1_graph.append(np.mean(f1))

            plt.figure()
            plt.plot([2,3,4,5], o_graph, label="Success Rate",  linestyle="-",  color="blue")
            plt.plot([2,3,4,5], p_graph, label="Precision",     linestyle=":",  color="orange")
            plt.plot([2,3,4,5], r_graph, label="Recall",        linestyle="--", color="red")
            plt.plot([2,3,4,5], f1_graph,label="F1 Score",      linestyle="-",  color="green")
            plt.xticks([2,3,4,5],["2","3","4","5"])
            plt.savefig(key_path+"4-Compo_Bins_"+key+".pdf")
            plt.legend()
            plt.close()
        '''

        ''' COMPO (2-feats) FAILURE CASES '''

        if eval_dict["steps"] != 100:
            display_failures(eval_dict[key]["compo_fails"],key_path)

        ''' COMPO (2-feats) MATRIX '''

        display_matrix(eval_dict[key]["compo_matrix"],key_path)

        ''' TOPOGRAPHY CORRELATION '''

        results_topo = eval_dict[key]["topography_corr"]
        topo_colors  = ["black","orange","blue","green"]

        for ref in results_topo.keys():

            A, B = ref[2], ref[5]

            legends = [("o","green","R["+A+","+B+"]"), ("o","blue","R["+A+",X]"), ("o","orange","R[X,"+B+"]"), ("x","black","R[X,X]")]

            dA, dB, colors = np.array(results_topo[ref]["dA"]), np.array(results_topo[ref]["dB"]), np.array(results_topo[ref]["colors"])

            mask_other       = (colors == 0)
            mask_B           = (colors == 1)
            mask_A           = (colors == 2)
            mask_COMPO       = (colors == 3)

            plot1 = plt.scatter(dA[mask_other],dB[mask_other] , marker="x", s=12, color="black", alpha=1)
            plot2 = plt.scatter(dA[mask_B],dB[mask_B]         , marker="o", s=12, color="orange", alpha=0.4)
            plot3 = plt.scatter(dA[mask_A],dB[mask_A]         , marker="o", s=12, color="blue", alpha=0.4)
            plot4 = plt.scatter(dA[mask_COMPO],dB[mask_COMPO] , marker="o", s=12, color="green", alpha=1)

            strA = "R["+ref[2]+"]"
            strB = "R["+ref[5]+"]"
            plt.xlabel(f"Hausdorff Distance ({strA})",fontsize=20)
            plt.ylabel(f"Hausdorff Distance ({strB})",fontsize=20)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.tight_layout()

            display_legend(plt, legends)

            plt.savefig(key_path+"7-Topography_"+ref+".pdf")
            plt.close()

        ''' TSNEs - BASICS & COMPO (2-feats) '''

        fig, axis = plt.subplots(1,2,figsize=(20,10))

        tsne_basics    = eval_dict[key]["tsne_basics"]
        colors_classes = ["blue","orange","purple","green","red"]
        colors         = [colors_classes[i] for i in tsne_basics["colors"]]
        labels         = tsne_basics["str"]

        unique_label_indexes = np.unique(labels,return_index=True)[1]
        legend_colors = np.array(colors)[unique_label_indexes].tolist()
        legend_labels = np.array(labels)[unique_label_indexes].tolist()

        legends = [("o",legend_colors[i],legend_labels[i]) for i in range(len(legend_labels))]

        axis[1].scatter(tsne_basics["refs"][:,0],tsne_basics["refs"][:,1],c=colors)
        axis[0].scatter(tsne_basics["utts"][:,0],tsne_basics["utts"][:,1],c=colors)
        axis[1].set_title("Utterances Embeddings",fontsize=40)
        axis[0].set_title("Referents Embeddings",fontsize=40)
        display_legend(axis[1], legends, 20)
        axis[1].set_xticks([])
        axis[1].set_yticks([])
        axis[0].set_xticks([])
        axis[0].set_yticks([])
        plt.tight_layout()
        plt.savefig(key_path+"8-TSNE-R1.pdf")
        plt.close()

        tsne_compo = eval_dict[key]["tsne_compos"]
        for feat in tsne_compo.keys():
            legends_compo      = legends+[("s","black",feat)]
            n_basics, n_compos = len(colors), (tsne_compo[feat]["utts"].shape[0] - len(colors))
            tsne_utts    = tsne_compo[feat]["utts"]
            tsne_refs    = tsne_compo[feat]["refs"]

            print(tsne_compo[feat]["refs"].shape, flush=True)
            ### Plot both tsne for utts & refs
            fig, axis = plt.subplots(1,2,figsize=(20,10))
            axis[1].scatter(tsne_utts[:n_basics,0],tsne_utts[:n_basics,1],c=colors)
            axis[0].scatter(tsne_refs[:n_basics,0],tsne_refs[:n_basics,1],c=colors)
            for i in range(n_compos):
                axis[1].scatter(tsne_utts[n_basics+i,0],tsne_utts[n_basics+i,1],c="black",marker="s")
                axis[0].scatter(tsne_refs[n_basics+i,0],tsne_refs[n_basics+i,1],c="black",marker="s")

            display_legend(axis[1], legends_compo, 30)

            axis[1].set_title("Utterances Embeddings",fontsize=40)
            axis[0].set_title("Referents Embeddings",fontsize=40)

            axis[1].set_xticks([])
            axis[1].set_yticks([])
            axis[0].set_xticks([])
            axis[0].set_yticks([])
            plt.tight_layout()
            plt.savefig(key_path+"9-TSNE-R2-"+str(feat)+".pdf")
            plt.close()

    print("---------- Done! :)", flush=True)


''' DISPLAY EVAL OVER ALL SEEDS '''
def display_eval_aggregation(exp_name):
    return None


if __name__ == "__main__":
    display_eval("base",1)
