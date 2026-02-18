import torch, os, random
import numpy as np
from utils import *
from agent import *
from ss import *
from data import *

''' LAUNCH EXPERIMENT '''


##### LAUNCH EXPERIMENTS #######################################################
def launch_exp(config):
    exp_name, seed = config["exp_name"], config["seed"]
    print(f"##### TRAINING - EXPERIMENT {exp_name} - SEED {seed}")
    print(f"##### CONFIG:\n {config}\n\n")

    ''' SET SEED '''
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ''' SEED PATH & FOLDERS '''
    exp_path = os.path.join(path, 'results', config["exp_name"] + "/")
    seed_path = os.path.join(exp_path, "seed" + str(seed) + "/")

    os.mkdir(seed_path)
    os.mkdir(seed_path + "Eval/")
    os.mkdir(seed_path + "Agents/")
    os.mkdir(seed_path + "history/")
    torch.save(config, seed_path + "config.pt")

    ''' TRAIN SET '''
    n_features, bins = config["n_features"], config["bins"]
    train_path = generate_systematic_dataset(n=n_features, m_list=bins)
    train_set = torch.load(train_path, map_location=device)

    if None is not config["transfer_refs"]:
        r1_mask = (torch.sum(train_set, 1) == 1)
        train_set = torch.cat((train_set[r1_mask], config["transfer_refs"]))

    ''' SENSORIMOTOR SYSTEM '''
    ss_class, ss_params = config["ss_class"], config["ss_params"]
    sensorimotor_system = available_ss[ss_class](ss_params)

    ''' AGENTS '''
    agents_path = config["agents_path"]
    if None is not agents_path: assert (os.path.exists(agents_path))

    agents = []
    for i in range(config["nb_agents"]):
        use_temp = True if not ("use_temp" in config.keys()) else config["use_temp"]
        use_baseline = True if not ("use_baseline" in config.keys()) else config["use_baseline"]
        agent = AgentEBM(sensorimotor_system, config["embedding_size"], config["action_size"], config["assoc_lr"],
                         config["action_lr"], config["use_img_perspectives"], config["nb_features"],
                         config["max_iterations"], use_temp=use_temp, use_baseline=use_baseline)
        if None is not agents_path:
            agent_dict = torch.load(agents_path + "agent" + str(i) + ".pt")
            agent.temp = agent_dict["temp"].to(device)
            agent.encoderA.load_state_dict(agent_dict["encoderA"])
            agent.encoderB.load_state_dict(agent_dict["encoderB"])
        agents.append(agent)

    ''' LANGUAGE GAME '''
    train_population(agents, train_set, config, sensorimotor_system, seed_path)


''' TRAIN POPULATION - LANGUAGE GAME '''


def train_population(agents, train_set, config, sensorimotor_system, seed_path):
    print(f"### STARTING LANGUAGE GAMES...")

    ''' LOGS & PARAMETERS '''
    logs = {"graph_outcomes": []}
    nb_agents, nb_referents = len(agents), min(train_set.shape[0], config["referents_bs"])
    nb_iter = config["max_iterations"]

    ''' ITERATION '''
    for i in range(nb_iter):

        ''' ABSTRACT REFERENTS BATCH'''
        referents_batch = get_random_batch(train_set, nb_referents)

        ''' PERCEIVED REFERENTS BATCH '''
        if config["use_img_perspectives"]:
            referents_speaker = convert_to_imgs(referents_batch, ood=config["ood"])
            if config["shared_perspective"]:
                referents_listener = referents_speaker.detach()
            else:
                referents_listener = convert_to_imgs(referents_batch, ood=config["ood"])
        else:
            referents_speaker = referents_batch.to(device)
            referents_listener = referents_batch.to(device)

        ''' RANDOM (SPEAKER, LISTENER) PAIR '''
        agents_ids = list(range(len(agents)))
        idS, idL = np.random.choice(agents_ids, 2, replace=False)
        speaker, listener = agents[idS], agents[idL]

        ''' SPEAKER'S UTTERANCES '''
        targets = torch.arange(0, nb_referents).long().to(device)

        if config["no_ss"] == False:
            utterances, _, _, _ = speaker.get_actions(referents_speaker, targets, iterations=config["action_it"],
                                                      nb_search=config["action_bs"], verbose=False)
        else:
            utterances, _, _, _ = speaker.get_direct_utterance(referents_speaker, targets,
                                                               iterations=config["action_it"],
                                                               nb_search=config["action_bs"], verbose=False)
        utterances.detach_()

        ''' LISTENER'S CHOICES & GAME OUTCOMES '''
        outcomes = (listener.get_referentSelections(referents_listener, utterances) == targets).float()

        ''' ASSOCIATION UPDATES '''
        speaker.update_speaker(referents_batch, targets, utterances, outcomes, ood=config["ood"])
        listener.update_listener(referents_batch, targets, utterances, ood=config["ood"])

        ''' LOGS & POPULATION HISTORY'''
        logs["graph_outcomes"].append(torch.mean(outcomes).item())

        if i % 100 == 0:
            mean_outcomes = np.mean(logs["graph_outcomes"][-100:])
            print(f"> Iteration {i} / {nb_iter} |\t\t{mean_outcomes * 100}% success rate")

        if i == 0 or (config["use_img_perspectives"] and (i + 1) % 100 == 0) or (
                not config["use_img_perspectives"] and (i + 1) % 100 == 0):
            print(f"\n# Saving Population-{i}...")
            print((config["use_img_perspectives"] and (i + 1) % 1000))
            torch.save(logs, seed_path + "logs.pt")
            save_history(seed_path, agents, (i if i == 0 else i + 1))
            print("# Done!\n")

    print("### ENDING LANGUAGE GAMES...")
    save_results(seed_path, agents, logs)
    print("### TRAINING RESULTS SAVED! :)\n\n\n")


''' SAVE RESULTS '''


def save_results(seed_path, agents, logs):
    agents_path = seed_path + "Agents/"
    torch.save(logs, seed_path + "logs.pt")
    for i, agent in enumerate(agents):
        agent_dict = {}
        agent_dict["encoderA"] = agent.encoderA.state_dict()
        agent_dict["encoderB"] = agent.encoderB.state_dict()
        agent_dict["temp"] = agent.temp.cpu()
        torch.save(agent_dict, agents_path + "agent" + str(i) + ".pt")


def load_exp(seed_path):
    results = {}

    ''' LOAD CONFIG '''
    config = torch.load(os.path.join(seed_path, "config.pt"))
    assert (os.path.exists(seed_path))

    ''' LOAD LOGS '''
    results["logs"] = torch.load(os.path.join(seed_path, "logs.pt"), map_location=device)

    ''' LOAD AGENTS '''
    ss_class, ss_params = config["ss_class"], config["ss_params"]
    sensorimotor_system = available_ss[ss_class](ss_params)
    agents_path = os.path.join(seed_path,"Agents/")
    agents_path = os.path.join(seed_path,"Agents/")
    agents = []
    for i in range(config["nb_agents"]):
        use_temp = True if not ("use_temp" in config.keys()) else config["use_temp"]
        use_baseline = True if not ("use_baseline" in config.keys()) else config["use_baseline"]
        agent = AgentEBM(sensorimotor_system, config["embedding_size"], config["action_size"], config["assoc_lr"],
                         config["action_lr"], config["use_img_perspectives"], config["nb_features"],
                         config["max_iterations"], use_temp, use_baseline)
        agent_dict = torch.load(os.path.join(agents_path, "agent" + str(i) + ".pt"), map_location=device)
        agent.temp = agent_dict["temp"].to(device)
        agent.encoderA.load_state_dict(agent_dict["encoderA"])
        agent.encoderB.load_state_dict(agent_dict["encoderB"])
        agent.encoderA.eval()
        agent.encoderB.eval()
        agents.append(agent)
    results["agents"] = agents

    return results, config


''' SAVE POPULATION HISTORY (ITERATION i) '''


def save_history(seed_path, agents, iteration):
    history_path = seed_path + "history/" + str(iteration) + "/"
    os.mkdir(history_path)
    for i, agent in enumerate(agents):
        agent_dict = {}
        agent_dict["encoderA"] = agent.encoderA.state_dict()
        agent_dict["encoderB"] = agent.encoderB.state_dict()
        agent_dict["temp"] = agent.temp.cpu()
        torch.save(agent_dict, history_path + "agent" + str(i) + ".pt")


''' LOAD POPULATION HISTORY (ITERATION i) '''


def load_history(seed_path, config, iteration):
    ''' SENSORIMOTOR SYSTEM '''
    ss_class, ss_params = config["ss_class"], config["ss_params"]
    sensorimotor_system = available_ss[ss_class](ss_params)

    ''' HISTORY PATH '''
    history_path = os.path.join(seed_path,"history/", str(iteration))

    if not os.path.exists(history_path):
        history_path = seed_path + "history/" + str(iteration - 1) + "/"

    ''' LOAD AGENTS '''
    agents = []
    for i in range(config["nb_agents"]):
        use_temp = True if not ("use_temp" in config.keys()) else config["use_temp"]
        use_baseline = True if not ("use_baseline" in config.keys()) else config["use_baseline"]
        agent = AgentEBM(sensorimotor_system, config["embedding_size"], config["action_size"], config["assoc_lr"],
                         config["action_lr"], config["use_img_perspectives"], config["nb_features"],
                         config["max_iterations"], use_temp, use_baseline)
        agent_dict = torch.load(os.path.join(history_path, "agent" + str(i) + ".pt"), map_location=device)
        agent.temp = agent_dict["temp"].to(device)
        agent.encoderA.load_state_dict(agent_dict["encoderA"])
        agent.encoderB.load_state_dict(agent_dict["encoderB"])
        agent.encoderA.eval()
        agent.encoderB.eval()
        agents.append(agent)

    return agents
