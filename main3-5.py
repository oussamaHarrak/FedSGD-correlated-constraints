from model import SimpleMNISTModel
from data_loader import get_dataloaders_iid,get_dataloaders_one_label_per_client,get_dataloaders_group_iid_ap,get_dataloaders_label_per_ap,get_dataloaders_label_distribution_per_ap
from server import average_models,variance_gradients
from client import compute_stochastic_gradient,compute_stochastic_gradient_printed,compute_full_gradient,compute_online_bias
from server import average_gradients,average_gradients_unbiaised,average_gradients_unbiaised_corr,apply_gradient,apply_gradient_clipped,monte_carlo_expectation_corr,real_expectation_corr,average_gradients_unbiaised_3groups_corr,monte_carlo_expectation_3groups_corr,apply_gradient_normalized
import matplotlib.pyplot as plt
from collections import Counter
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

def set_seed(seed = 0): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def test_model(model, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.view(x.size(0), -1)  # Flatten the 28x28 image
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


#For smoothing the plot
def smooth(y, box_pts=10):
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')

def clip_gradient(g, max_norm):

    total_norm = torch.sqrt(sum((v.norm() ** 2 for v in g.values())))
    clip_coef = min(1.0, max_norm / (total_norm + 1e-6))  
    return clip_coef


#Function to run the experiments 
def run_experiment(M1, M2, M3, e1, e2 , e3 , N=500):
    #dataloaders = get_dataloaders_iid() #get_dataloaders_corr2()
    #dataloaders = get_dataloaders_one_label_per_client()
    #dtaloaders = get_dataloaders_iid()
    
    group1 = list(range(M1))
    group2 = list(range(M1, M1 + M2))
    group3 = list(range(M1 + M2 , M1 + M2 + M3))
    M = M1 + M2 + M3
    
    ap_to_clients = {
    1: [0, 1, 2],
    2: [3 , 4, 5, 6],
    3: [7, 8, 9]
    }

    ap_to_labels = {
    1: [0],
    2: [1],
    3: [2]

    }   
      
    #dataloaders = get_dataloaders_group_iid_ap(32 , group1 , group2 , group3) #get_dataloaders_one_label_per_client(32,10) #get_dataloaders_one_label_per_client(32 , M , 0)
    dataloaders = get_dataloaders_label_per_ap(32 , ap_to_clients , ap_to_labels )
    for cid, loader in dataloaders.items():
        labels = []
        for _, y in loader:
            labels.extend(y.tolist())
        print(f"Client {cid}: label counts = {dict(Counter(labels))}")

    #get_dataloaders_iid(32 , M , 0) #get_dataloaders_group_iid_ap(32 , group1 , group2 , group3)
    #get_dataloaders_label_distribution_per_ap(32, ap_to_clients , ap_to_labels)#get_dataloaders_label_per_ap(32, ap_to_clients , ap_to_labels)#get_dataloaders_one_label_per_client(32,10) #get_dataloaders_label_per_ap(32, ap_to_clients , ap_to_labels)#get_dataloaders_one_label_per_client(32,10) #get_dataloaders_label_per_ap(32, ap_to_clients , ap_to_labels) #get_dataloaders_one_label_per_client(32,10) #get_dataloaders_label_per_ap(32, ap_to_clients , ap_to_labels) #get_dataloaders_one_label_per_client(32,10) #get_dataloaders_label_per_ap(32, ap_to_clients , ap_to_labels) #get_dataloaders_one_label_per_client(32,10) #get_dataloaders_group_iid_ap(32 , [0,1] , [2,3,4,5] , [6,7,8,9]) #get_dataloaders_one_label_per_client(32,10)
    client_lengths = [len(dataloader.dataset) for dataloader in dataloaders.values()]

    clients = list(range(M1+M2+M3))
    epsilons_access = {}
    epsilons_access[0] = e1
    epsilons_access[1] = e2
    epsilons_access[2] = e3


    epsilons_active = {0 : 0.7,1 : 0.7 ,2  : 0.7,3 : 0.4,4 : 0.4, 5 : 0.4 , 6 : 0.4 , 7 : 0.7 , 8: 0.7 , 9 : 0.7}
    epsilons_nonactive = {0 : 0.9,1 : 0.9,2 : 0.9,3 : 0.9,4 : 0.9 , 5: 0.9 , 6 : 0.9 , 7 : 0.9 , 8 : 0.9 , 9 : 0.9}
   
    global_biased = SimpleMNISTModel()
    """
    for name, param in global_biased.named_parameters():
        print(f"global biased : {name}: {param.data}")
    """

    lr = 0.01 #0.0447 #0.01 $1/sqrt(T) = 0,00005
    K = 1
    c = 0
    biased_loss_rnds= []
    unbiased_loss_rnds = []
    unbiased_th_loss_rnds = []
    rounds_biaised = []
    
    avg_grads_biased = []
    avg_grads_biased_total = []


    print("Biased Training")
    # Biased Training
    keys = ['fc.0.weight','fc.0.bias','fc.2.weight','fc.2.bias']
    for rnd in range(N):
        M_t = 0
        grads_all = []
        X1 = np.random.binomial(1, 1 - epsilons_access[0])
        X2 = np.random.binomial(1, 1 - epsilons_access[1])
        X3 = np.random.binomial(1, 1 - epsilons_access[2])

        for cid in dataloaders:

            if cid in group1 : 
                if X1 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            elif cid in group2:
                if X2 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            else :
                if X3 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            #Calculate the probability of participation of the client
            if np.random.binomial(1, 1 - epsilon):
                M_t += 1
                local = SimpleMNISTModel()
                local.load_state_dict(global_biased.state_dict()) #Get the latest global model 
                grad = compute_stochastic_gradient(local, dataloaders[cid], K)

                #grad = compute_full_gradient(local , dataloaders[cid])
                """
                if cid in [0,1,2,7,8,9] :
                    grad = {key: -1 * grad[key] for key in keys}"
                """
            else :
                grad  = {key: 0.0 for key in keys}

            grads_all.append(grad)

         
        if M_t != 0 :
            c += 1
            avg_grad = average_gradients(grads_all,M_t)#,client_lengths)
            apply_gradient(global_biased, avg_grad, lr)
            avg_grad_array = np.concatenate([v.ravel() for v in avg_grad.values()])
            norm = np.linalg.norm(avg_grad_array)
            avg_grads_biased_total.append(norm ** 2)
            if (rnd + 1) % 100 == 0 : 
                print(f"\n--- Communication Round {rnd+1} ---")
                rounds_biaised.append(rnd)
                avg_grads_biased.append(norm**2)
                biased_loss_rnd = test_model(global_biased, dataloaders[1])   
                biased_loss_rnds.append(biased_loss_rnd)
                print(f"loss : {biased_loss_rnd}")
        else : 
            if (rnd + 1) % 100 == 0 : 
                print(f"\n--- Communication Round {rnd+1} ---")
                rounds_biaised.append(rnd - 1)
                avg_grads_biased.append(avg_grads_biased_total[c - 1])
                biased_loss_rnd = test_model(global_biased, dataloaders[1])   
                biased_loss_rnds.append(biased_loss_rnd)
                print(f"loss : {biased_loss_rnd}")
            
     # Unbiased Training
    print("Unbiased Training")
    
    global_unbiased = SimpleMNISTModel()
    """
    for name, param in global_unbiased.named_parameters():
        print(f"global_unbiased : {name}: {param.data}")
    """
    avg_grads_unbiased = []
    M_t = M
    for rnd in range(N):
        grads_all = []
        for cid in dataloaders:
            local = SimpleMNISTModel()
            local.load_state_dict(global_unbiased.state_dict())
            #print(f"client's used data : {cid}")
            grad = compute_stochastic_gradient(local, dataloaders[cid], K)
            #grad = compute_full_gradient(local , dataloaders[cid])
            """
            if cid in [0,1,2,7,8,9] :
                grad = {key: -1 * grad[key] for key in keys}
            """
            grads_all.append(grad)
        avg_grad = average_gradients(grads_all,M_t)#,client_lengths)
        apply_gradient(global_unbiased, avg_grad, lr)
        if (rnd + 1) % 100 == 0 : 
            print(f"\n--- Communication Round {rnd+1} ---")
            avg_grad_array = np.concatenate([v.ravel() for v in avg_grad.values()])
            norm = np.linalg.norm(avg_grad_array)
            avg_grads_unbiased.append(norm**2)
            unbiased_loss_rnd = test_model(global_unbiased, dataloaders[1]) 
            unbiased_loss_rnds.append(unbiased_loss_rnd)
            print(f"loss : {unbiased_loss_rnd}")    


    
    
    
    # Debiased Training (dividing by the biais)
    print("Debiased Training (dividing by the biais)")
    global_debiased = SimpleMNISTModel()
    """
    for name, param in global_debiased.named_parameters():
        print(f"global_debiased : {name}: {param.data}")
    """
    epsilons = []
    debiased_loss_rnds = []
    
    """
    biases_real = []
    ap_list = [1,2]
    
    for m in clients :
        bias2 = real_expectation_corr(m,M_t, epsilons_active, epsilons_nonactive ,epsilons_access,ap_list,group1,group2)
        biases_real.append(bias2)
    """
    """
    biases_monte = []
    for m in clients: 
        bias = monte_carlo_expectation_3groups_corr(m, group1, group2,group3, epsilons_access, epsilons_active, epsilons_nonactive, N=10000)
        biases_monte.append(bias)
    print(f"estimated biases via monte : {biases_monte}")
    """
    #print(f"biases_monte : {biases_monte}")
    
    #mean_bias = np.mean(biases)


    lr = 0.01
    avg_grads_debiased = []
    avg_grads_debiased_total = []
    rounds_debiased = []
    biases_total = []
    X = [[] for _ in range(M)]
    c = 0
    delta = 0.01
    for rnd in range(N):
        """
        if rnd < 400 :
            delta = 0.1
        else :
            delta = 0.01
        """
        M_t = 0
       
        grads_all = []  
        biases = []
        biases_no_beta = []
        participations = np.zeros(M)
        beta = 1 - 1/sqrt(rnd + 1)
        #beta = (rnd + 0.001)/N
        
        X1 = np.random.binomial(1, 1 - epsilons_access[0])
        X2 = np.random.binomial(1, 1 - epsilons_access[1])
        X3 = np.random.binomial(1, 1 - epsilons_access[2])

        for cid in dataloaders:
            if cid in group1 : 
                if X1 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            elif cid in group2 :
                if X2 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            else : 
                if X3 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            epsilons.append(epsilon)
            #Calculate the probability of participation of the client
            if np.random.binomial(1, 1 - epsilon):
                participations[cid] = 1
                local = SimpleMNISTModel()
                local.load_state_dict(global_debiased.state_dict()) #Get the latest global model 
                grad = compute_stochastic_gradient(local, dataloaders[cid], K)
                #grad = compute_stochastic_gradient(local, dataloaders[cid], K)
                #grad = compute_full_gradient(local , dataloaders[cid])

                M_t += 1
            else :
                grad  = {key: 0.0 for key in keys}
            grads_all.append(grad)
            
        
        #delta = max(0.3 / (rnd + 1)**0.3, 0.05)
        #delta = 0.5
        #delta = 0.1/(rnd + 1)
        #print(f"M_t : {M_t}")
        for cid in dataloaders : 
            if participations[cid] == 1 : 
                X[cid].append(1/M_t)
            else :
                X[cid].append(0)
            
            bias = compute_online_bias(X[cid], delta)
            biases_no_beta.append(bias)
            biases.append(bias**beta)
        #print(f"biases : {biases}")
        mean_bias = np.mean(biases)
        biases_total.append(biases)
        
        #print(f"X : {X  }")
        if M_t != 0:
            c += 1
            max_norm = min(10 , 1 + 5 / sqrt(rnd + 1))
            #print(f"grads_all : {grads_all}")
            #print(f"biases : {biases}")
            avg_grad = average_gradients_unbiaised_3groups_corr(grads_all,group1,group2,group3, biases , M_t)#,client_lengths) 
            lr = 0.01 #* mean_bias
            avg_grad_array = np.concatenate([v.ravel() for v in avg_grad.values()])
            norm = np.linalg.norm(avg_grad_array)
            apply_gradient(global_debiased, avg_grad, lr)
            avg_grads_debiased_total.append(norm ** 2)
            if (rnd + 1) % 100 == 0 : 
                print(f"\n--- Communication Round {rnd+1} ---")
                rounds_debiased.append(rnd)
                avg_grads_debiased.append(norm**2)
                debiased_loss_rnd = test_model(global_debiased, dataloaders[1])   
                debiased_loss_rnds.append(debiased_loss_rnd)
                print(f"loss : {debiased_loss_rnd}")
            
        else : 
            if (rnd + 1) % 100 == 0 : 
                print(f"\n--- Communication Round {rnd+1} ---")
                rounds_debiased.append(rnd - 1)
                avg_grads_debiased.append(avg_grads_debiased_total[c - 1])
                debiased_loss_rnd = test_model(global_debiased, dataloaders[1])   
                debiased_loss_rnds.append(debiased_loss_rnd)
                print(f"loss : {debiased_loss_rnd}")
           

   
    return biased_loss_rnds , unbiased_loss_rnds , debiased_loss_rnds , avg_grads_biased , avg_grads_unbiased , avg_grads_debiased ,  rounds_biaised , rounds_debiased , biases_total

    

def main(): 

    #One fixed seed
    """
    set_seed(42)
    biased_loss_rnds , unbiased_loss_rnds , debiased_loss_rnds , avg_grads_biased , avg_grads_unbiased , avg_grads_debiased , rounds , rounds2 , biases_total = run_experiment(M1=3, M2=4 , M3 = 3, e1=0.94, e2=0.02, e3 = 0.03 ,N=4001) 
    
    
    biased_losses =   [i[0] for i in biased_loss_rnds]
    unbiased_losses = [i[0] for i in unbiased_loss_rnds]
    debiased_losses = [i[0] for i in debiased_loss_rnds]
    
   

    plt.plot(rounds, avg_grads_biased, label="Biased", linestyle='-', marker='o')
    plt.plot(rounds2, avg_grads_debiased , label="Debiaised", linestyle='-', marker='o',color = 'red')
    plt.plot(rounds, avg_grads_unbiased, label="Unbiased", linestyle='--', marker='s')

    plt.xlabel("Communication Round")
    plt.ylabel("Squared norm of the gradient of the loss function")
    plt.title("Comparison of Squared norm of the gradient Over Rounds (e1=0.94, e2=0.02 , e3 = 0.03) \\ iid labels per ap, M1 = 3, M2 = 4 , M3 = 3, lr = 0.01, beta = (t + 0.001)/T ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.ylim(0, 4) 
    plt.show()   

    """
    """
    plt.plot(rounds, biased_losses, label="Biased", linestyle='-', marker='o')
    plt.plot(rounds2, debiased_losses , label="Debiaised", linestyle='-', marker='o',color = 'red')
    plt.plot(rounds, unbiased_losses , label="Unbiased", linestyle='--', marker='s')

    plt.xlabel("Communication Round")
    plt.ylabel("Training Loss")
    plt.title("Comparison of Training Loss Over Rounds (e1=0.94, e2=0.02 , e3 = 0.03) \\ iid labels per ap , M1 = 3, M2 = 4 , M3 = 3, lr = 0.01 ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.ylim(0, 4) 
    plt.show() 
    """

    seeds = [1]
    all_biased, all_unbiased, all_debiased = [], [], []
    rounds_shared = None
    rounds2_shared = None

    for seed in seeds:
        set_seed(seed)
        (
        biased_loss_rnds, 
        unbiased_loss_rnds, 
        debiased_loss_rnds, 
        avg_grads_biased, 
        avg_grads_unbiased, 
        avg_grads_debiased, 
        rounds, 
        rounds2, 
        biases_total
        ) = run_experiment(M1=3, M2=4, M3=3, e1=0.9, e2=0.02, e3=0.9, N=3001)
        biased_losses =   [i[0] for i in biased_loss_rnds]
        unbiased_losses = [i[0] for i in unbiased_loss_rnds]
        debiased_losses = [i[0] for i in debiased_loss_rnds]
        print(f"len(avg_grads_biased)  : {len(avg_grads_biased)}")
        print(f"len(avg_grads_unbiased) : {len(avg_grads_unbiased)}")
        print(f"len(avg_grads_debiased) : {len(avg_grads_debiased)}")
        all_biased.append(avg_grads_biased)
        all_unbiased.append(avg_grads_unbiased)
        all_debiased.append(avg_grads_debiased)

        #print(f"rounds : {rounds} , rounds2 : {rounds2}")
        if rounds_shared is None:
            rounds_shared = rounds
            rounds2_shared = rounds2


    biased_array = np.stack(all_biased)
    unbiased_array = np.stack(all_unbiased)
    debiased_array = np.stack(all_debiased)


    biased_mean = biased_array.mean(axis=0)
    unbiased_mean = unbiased_array.mean(axis=0)
    debiased_mean = debiased_array.mean(axis=0)

    biased_std = biased_array.std(axis=0)
    unbiased_std = unbiased_array.std(axis=0)
    debiased_std = debiased_array.std(axis=0)

  
    plt.figure(figsize=(10, 6))

    def plot_with_std(x, mean, std, label, color, linestyle):
        plt.plot(x, mean, label=label, linestyle=linestyle, color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    min_len = min(len(rounds_shared), len(unbiased_mean))


    plt.plot(rounds_shared, biased_losses, label="Biased", linestyle='-', marker='o')
    plt.plot(rounds2_shared, debiased_losses , label="Debiaised", linestyle='-', marker='o',color = 'red')
    plt.plot(rounds_shared, unbiased_losses , label="Unbiased", linestyle='--', marker='s')

    plt.xlabel("Communication Round")
    plt.ylabel("Training Loss")
    plt.title("Comparison of Training Loss Over Rounds 3 labels one label per ap")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.ylim(0, 4) 
    plt.show() 


    plot_with_std(rounds_shared, biased_mean, biased_std, "Biased", 'blue', '-')
    plot_with_std(rounds2_shared, debiased_mean, debiased_std, "Debiased", 'red', '-')
    plot_with_std(rounds_shared, unbiased_mean[:min_len], unbiased_std[:min_len], "Unbiased", 'green', '--')
    #plot_with_std(rounds_shared, debiased_mean, debiased_std, "Debiased", 'red', '-')
    #plot_with_std(rounds2_shared, unbiased_mean[:min_len], unbiased_std[:min_len], "Unbiased", 'green', '--')

    plt.xlabel("Communication Round")
    plt.ylabel("Squared norm of the gradient of the loss function")
    plt.title("Gradient Norm Over Rounds 3 labels one label per ap ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



















    """
    biased_accuracies   = [i[1] for i in biased_loss_rnds]
    unbiased_accuracies = [i[1] for i in unbiased_loss_rnds]
    unbiased_th_accuracies = [i[1] for i in unbiased_loss_th_rnds]

    plt.plot(rounds, smooth(biased_accuracies), label="Biased", linestyle='-', marker='o')
    plt.plot(rounds2, smooth(unbiased_th_accuracies) , label="Debiaised", linestyle='-', marker='o',color = 'red')
    plt.plot(rounds, smooth(np.array(unbiased_accuracies)[rounds]), label="Unbiased", linestyle='--', marker='s')


    plt.xlabel("Communication Round")
    plt.ylabel("Training Accuracy")
    plt.title("Comparison of Training Accuracy Over Rounds (e1 = 0.8 , e2 = 0.2) one label per client with higher probabilities for client in AP2 ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.ylim(0, 4) 
    plt.show()  
    """
    """
    biased_lossess = []
    unbiased_lossess = []
    unbiased_th_lossess = []
    
    for i in range(2):
        biased_loss_rnds , unbiased_loss_rnds , unbiased_loss_th_rnds , rounds , rounds2 = run_experiment(M1=3, M2=2, e1=0.8, e2=0.2 , N=50) 
        biased_losses = [i[0] for i in biased_loss_rnds]
        unbiased_losses = [i[0] for i in unbiased_loss_rnds]
        unbiased_th_losses = [i[0] for i in unbiased_loss_th_rnds]
        biased_lossess.append(pd.Series(smooth(biased_losses), index = rounds))
        unbiased_lossess.append(pd.Series(smooth(unbiased_losses[rounds]) , index = rounds))
        unbiased_th_lossess.append(pd.Series(smooth(unbiased_th_losses),index = rounds2))
    
    biased_lossess_mean = pd.concat(biased_lossess , axis = 1).mean(axis = 1)
    unbiased_lossess_mean = pd.concat(unbiased_lossess, axis = 1).mean(axis = 1)
    unbiased_th_lossess_mean = pd.concat(unbiased_th_lossess , axis = 1).mean(axis=1)
    """
    #biased_loss_rnds_2 , unbiased_loss_rnds_2 , rounds2 = run_experiment(M1=3, M2=2, e1=0.7, e2=0.9 , N=500) 
    #rounds = rounds_biaised[:-50]
    #print(f"biased_loss_rnds : {biased_loss_rnds}")
    #print(f"unbiased_loss_rnds : {unbiased_loss_rnds}")
    #print(f"unbiased_loss_th_rnds : {unbiased_loss_th_rnds}")
    #biased_losses = [biased_loss_rnds[i][0] for i in range(len(biased_loss_rnds))]
    #unbiased_losses = [unbiased_loss_rnds[i][0] for i in range(len(unbiased_loss_rnds))]
    #unbiased_loss_th_rnds = [unbiased_loss_th_rnds[i][0] for i in range(len(unbiased_loss_th_rnds))]
    """
    biased_accuracies= [i[1] for i in biased_loss_rnds]
    unbiased_accuracies = [i[1] for i in unbiased_loss_rnds]
    unbiased_th_accuracies = [i[1] for i in unbiased_loss_th_rnds]

    biased_losses = [i[0] for i in biased_loss_rnds]
    unbiased_losses = [i[0] for i in unbiased_loss_rnds]
    unbiased_th_losses = [i[0] for i in unbiased_loss_th_rnds]
    print(f"rounds : {rounds}  of len : {len(rounds)}, biased_losses : {biased_losses} of len : {len(biased_losses)} , len(smooth(biased_losses))  : {len(smooth(biased_losses))}")
    """
    #print(f"len(rounds) : {len(rounds)} , len(biased_losses) : {biased_losses} , len(smooth(biased_losses)) : {len(smooth(biased_losses))}")
    #plt.figure(figsize=(10, 6))
    #plt.plot(rounds, smooth(biased_loss_rnds), label="Biased", linestyle='-', marker='o')
    """
    print(f"biased_loss_rnds_1 : {biased_loss_rnds_1}")
    print(f"unbiased_loss_th_rnds : {unbiased_loss_th_rnds}")
    print(f"unbiased_loss_rnds_1 : {unbiased_loss_rnds_1}")
    print(f"rounds : {rounds}")
    print(f"rounds2 : {rounds2}")
    """
    """
    plt.plot(rounds, biased_accuracies, label="Biased (e1 = 0.5 , e2 = 0.6)", linestyle='-', marker='o')
    plt.plot(rounds2, unbiased_accuracies , label="Unbiaised th", linestyle='-', marker='o',color = 'red')
    plt.plot(rounds, np.array(unbiased_th_accuracies)[rounds], label="Unbiased", linestyle='--', marker='s')
    """
    """
    plt.plot(roundss_mean, biased_lossess_mean, label="Biased (e1 = 0.5 , e2 = 0.6)", linestyle='-', marker='o')
    plt.plot(roundss2_mean, unbiased_lossess_mean , label="Unbiaised th", linestyle='-', marker='o',color = 'red')
    plt.plot(roundss_mean, np.array(unbiased_th_lossess_mean)[rounds], label="Unbiased", linestyle='--', marker='s')
    """
    """
    plt.plot(rounds, biased_losses, label="Biased (e1 = 0.5 , e2 = 0.6)", linestyle='-', marker='o')
    plt.plot(rounds2, unbiased_th_losses , label="Unbiaised th", linestyle='-', marker='o',color = 'red')
    plt.plot(rounds, np.array(unbiased_losses)[rounds], label="Unbiased", linestyle='--', marker='s')
    
    """
    #plt.plot(rounds, unbiased_loss_rnds[rounds_biaised], label="Unbiased (Real)", linestyle='--', marker='s')
    #plt.plot(rounds, unbiased_th_loss_rnds, label="Unbiased (Theoretical)", linestyle='-.', marker='^')
    """
    plt.xlabel("Communication Round")
    plt.ylabel("Test Loss")
    plt.title("Comparison of Test Loss Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0, 4) 
    plt.show()  
    """
    
    
    #run_experiment(M1=1, M2=4, e1=0.5, e2=0.1)
    #run_experiment(M1=3, M2=2, e1=0.1, e2=0.5)
    """
    for (M1, M2) in [(4,1),(3,2),(2, 3),(1, 4)]:
        for e1 in [0.1, 0.3, 0.5, 0.7]:
            e2 = 0.1  
            print(f"\nRunning experiment with M1={M1}, M2={M2}, e1={e1}, e2={e2}")
            run_experiment(M1, M2, e1, e2)
    """

    """
    biased_all = []
    unbiased_all = []
    unbiased_th_all = []

    for _ in range(5):
        biased_loss_rnds, unbiased_loss_rnds, unbiased_loss_th_rnds, rounds, rounds2 = run_experiment(
            M1=3, M2=2, e1=0.6, e2=0.3, N=200
        )
    
        biased_curve = [i[0] for i in biased_loss_rnds]
        unbiased_curve = [i[0] for i in unbiased_loss_rnds]
        unbiased_th_curve = [i[0] for i in unbiased_loss_th_rnds]

        biased_all.append(np.array(biased_curve))   
        unbiased_all.append(np.array(unbiased_curve))
        unbiased_th_all.append(np.array(unbiased_th_curve))

    min_len = min(
        min(len(arr) for arr in biased_all),
        min(len(arr) for arr in unbiased_all),
        min(len(arr) for arr in unbiased_th_all)
    )

    biased_all = np.stack([arr[:min_len] for arr in biased_all])
    unbiased_all = np.stack([arr[:min_len] for arr in unbiased_all])
    unbiased_th_all = np.stack([arr[:min_len] for arr in unbiased_th_all])

    biased_mean = biased_all.mean(axis=0)
    unbiased_mean = unbiased_all.mean(axis=0)
    unbiased_th_mean = unbiased_th_all.mean(axis=0)

    round_axis = np.arange(min_len)
    plt.figure(figsize=(6,4))
    plt.plot(round_axis, biased_mean,      label='Biased (mean of 10)')
    plt.plot(round_axis, unbiased_mean,    label='Unbiased (mean of 10)')
    plt.plot(round_axis, unbiased_th_mean, label='Unbiased-th (mean of 10)')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend(); plt.tight_layout()
    plt.show()
    """
    
if __name__ == "__main__":
    main()
