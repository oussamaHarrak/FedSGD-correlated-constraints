from model import SimpleMNISTModel
from data_loader import get_dataloaders_iid,get_dataloaders_one_label_per_client,get_dataloaders_group_iid_ap
from server import average_models,variance_gradients
from client import compute_stochastic_gradient
from server import average_gradients,average_gradients_unbiaised,average_gradients_unbiaised_corr,apply_gradient,apply_gradient_clipped,monte_carlo_expectation_corr,real_expectation_corr,average_gradients_unbiaised_3groups_corr,monte_carlo_expectation_3groups_corr,monte_carlo_expectation_4groups_corr,average_gradients_unbiaised_4groups_corr
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#Function to run the experiments 
def run_experiment(M1, M2, M3 , M4 , e1, e2 , e3 , e4 , N=500):
    #dataloaders = get_dataloaders_iid() #get_dataloaders_corr2()
    #dataloaders = get_dataloaders_one_label_per_client()
    #dtaloaders = get_dataloaders_iid()

    group1 = list(range(M1))
    group2 = list(range(M1, M1 + M2))
    group3 = list(range(M1 + M2 , M1 + M2 + M3))
    group4 = list(range(M1 + M2 + M3 , M1 + M2 + M3 + M4))

    dataloaders = get_dataloaders_group_iid_ap(32 , [0,1,2,3,4] , [5,6] , [7,8] , [9]) #get_dataloaders_one_label_per_client(32,10)  #get_dataloaders_group_iid_ap(32 , [0,1,2] , [3,4] , [5,6] , [7,8,9]) #get_dataloaders_one_label_per_client(32,10) #get_dataloaders_group_iid_ap(32 , [0,1,2] , [3,4] , [5,6] , [7,8,9]) #get_dataloaders_one_label_per_client(32,10) #get_dataloaders_group_iid_ap(32 , [0,1] , [2,3,4,5] , [6,7,8,9]) #get_dataloaders_one_label_per_client(32,10)
    client_lengths = [len(dataloader.dataset) for dataloader in dataloaders.values()]

    clients = list(range(M1+M2+M3+M4))
    epsilons_access = {}
    epsilons_access[0] = e1
    epsilons_access[1] = e2
    epsilons_access[2] = e3
    epsilons_access[3] = e4 

    e_active1 = 0.7 #Probability of error when the access point 1 is error-free
    e_nonactive1 = 0.9 #Probability of error when the access point 1 is not error-free

    e_active2 = 0.2  #Probability of error when the access point 2 is error-free
    e_nonactive2 = 0.8  #Probability of error when the access point 3 is not error-free

    e_active3 = 0.2  #Probability of error when the access point 2 is error-free
    e_nonactive3 = 0.8  #Probability of error when the access point 3 is not error-free

    """
    #Trying with smaller values of epsilons 
    e_active1 = 0.1  #Probability of error when the access point 1 is error-free
    e_nonactive1 = 0.1 #Probability of error when the access point 1 is not error-free

    e_active2 = 0.1  #Probability of error when the access point 2 is error-free
    e_nonactive2 = 0.1  #Probability of error when the access point 3 is not error-free
    """

    epsilons_active = {0 : 0.02,1 : 0.01 ,2  : 0.05,3 : 0.08,4 : 0.09, 5 : 0.95  , 6 : 0.97 , 7 : 0.95 , 8: 0.97 , 9 : 0.94}
    epsilons_nonactive = {0 : 0.9,1 : 0.8,2 : 0.9,3 : 0.8,4 : 0.9 , 5: 0.8 , 6 : 0.8 , 7 : 0.8 , 8 : 0.9 , 9 : 0.8}
    
    """
    for cid in group1: epsilons_active[cid] = e_active1
    for cid in group1: epsilons_nonactive[cid] = e_nonactive1

    for cid in group2: epsilons_active[cid] = e_active2
    for cid in group2: epsilons_nonactive[cid] = e_nonactive2
    """
    global_biased = SimpleMNISTModel()
    #lr = 0.0001/0.3
    lr = 0.01# 0.0447 #0.01
    K = 1

    biased_loss_rnds= []
    unbiased_loss_rnds = []
    unbiased_th_loss_rnds = []
    rounds_biaised = []
    

    # Biased Training
    for rnd in range(N):
        M_t = 0
        print(f"\n--- Communication Round {rnd+1} ---")
        grads_all = []
        X1 = np.random.binomial(1, 1 - epsilons_access[0])
        X2 = np.random.binomial(1, 1 - epsilons_access[1])
        X3 = np.random.binomial(1, 1 - epsilons_access[2])
        X4 = np.random.binomial(1, 1 - epsilons_access[3])

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
            elif cid in group3:
                if X3 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            else :
                if X4 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]

            #Calculate the probability of participation of the client
            if np.random.binomial(1, 1 - epsilon):
                M_t += 1
                local = SimpleMNISTModel()
                local.load_state_dict(global_biased.state_dict()) #Get the latest global model 
                grad = compute_stochastic_gradient(local, dataloaders[cid], K)
                grads_all.append(grad)
         
        if M_t != 0 :
          
            avg_grad = average_gradients(grads_all,M_t)#,client_lengths)
            apply_gradient(global_biased, avg_grad, lr)
            biased_loss_rnd = test_model(global_biased, dataloaders[1]) 
            rounds_biaised.append(rnd)
            #biased_loss_rnds.append(biased_loss_rnd)
            biased_loss_rnds.append(biased_loss_rnd )#if grads_all else np.nan)
            print(f"loss : {biased_loss_rnd}")
    biased_loss = test_model(global_biased, dataloaders[1]) 
        
     # Unbiased Training
    print("Unbiased Training")
   
    global_unbiased = SimpleMNISTModel()
    M_t = 10    
    for rnd in range(N):
        print(f"\n--- Communication Round {rnd+1} ---")
        grads_all = []
        for cid in dataloaders:
            local = SimpleMNISTModel()
            local.load_state_dict(global_unbiased.state_dict())
            grad = compute_stochastic_gradient(local, dataloaders[cid], K)
            grads_all.append(grad)
        avg_grad = average_gradients(grads_all,M_t)#,client_lengths)
        apply_gradient(global_unbiased, avg_grad, lr)
        unbiased_loss_rnd = test_model(global_unbiased, dataloaders[1]) 
        unbiased_loss_rnds.append(unbiased_loss_rnd)
        print(f"loss : {unbiased_loss_rnd}")    


    unbiased_loss = test_model(global_unbiased, dataloaders[1])
    
    #Quantifiying the biais
    param_bias = sum(torch.norm(p1 - p2).item()
                     for p1, p2 in zip(global_biased.parameters(), global_unbiased.parameters()))
    

    #Increasing the value of the stepsize gives us the loss of the unbiaised estimation of the gradients (using the theoritical formula)
    initial_lr = 0.0001
    decay_rate = 0.001
    #lr = 0.001
    #lr = 0.0001
    
    #lr = 0.01 * 0.16        
    # UnBiased Theoritical Training (dividing by the biais)
    print("UnBiased Theoritical Training (dividing by the biais)")
    global_unbiased_th = SimpleMNISTModel()
    epsilons = []
    unbiased_loss_th_rnds = []
    #optimizer = torch.optim.Adam(global_unbiased_th.parameters(), lr=0.01)
    #previous_loss = 10000
    """
   
    biases_real = []
    ap_list = [1,2]
    
    for m in clients :
        bias2 = real_expectation_corr(m,M_t, epsilons_active, epsilons_nonactive ,epsilons_access,ap_list,group1,group2)
        biases_real.append(bias2)
    """
    biases = []
    for m in clients: 
        bias = monte_carlo_expectation_4groups_corr(m, group1, group2,group3,group4, epsilons_access, epsilons_active, epsilons_nonactive, N=10000)
        biases.append(bias)
    mean_bias = np.mean(biases)

    #print(f"monte carlo biases : {biases}")
    #print(f"Real biases : {biases_real}")
    lr = 0.01 * mean_bias #0.0447 * mean_bias#0.01 * mean_bias

    rounds_unbiased_th = []
    for rnd in range(N):
        M_t = 0
        print(f"\n--- Communication Round {rnd+1} ---")
        grads_all = []  
        X1 = np.random.binomial(1, 1 - epsilons_access[0])
        X2 = np.random.binomial(1, 1 - epsilons_access[1])
        X3 = np.random.binomial(1, 1 - epsilons_access[2])
        X4 = np.random.binomial(1, 1 - epsilons_access[3])

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
            elif cid in group3:
                if X3 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            else :
                if X4 : #The access point is error-free so we take the error-free probabilities
                    epsilon = epsilons_active[cid]
                else : #The access point is not error-free so we take the error probabilities
                    epsilon = epsilons_nonactive[cid]
            epsilons.append(epsilon)
            #Calculate the probability of participation of the client
            if np.random.binomial(1, 1 - epsilon):
                local = SimpleMNISTModel()
                local.load_state_dict(global_unbiased_th.state_dict()) #Get the latest global model 
                grad = compute_stochastic_gradient(local, dataloaders[cid], K)
                #print(f"gradient : {grad}")

                #print(f"gradient : {grad}")
                #print(f"gradient for client {cid} : {grad}")
                grads_all.append(grad)
                M_t += 1
            
            """
            else :
                 
                grads_all.append({'fc.0.weight': torch.tensor([[0.0]]), 'fc.0.bias': torch.tensor([0.0]),
                                  'fc.2.weight': torch.tensor([[0.0]]), 'fc.2.bias': torch.tensor([0.0])})
            """
            
        #print(f"Number of participating clients : {len(grads_all)}")
        if M_t != 0:
            rounds_unbiased_th.append(rnd)
            avg_grad = average_gradients_unbiaised_4groups_corr(grads_all,group1,group2,group3,group4, biases)#,client_lengths) 
            #avg_grad = average_gradients_unbiaised_corr_2(grads_all,epsilons,M_t)
            #print(f"Variance of gradient estimation : {variance}")
            apply_gradient(global_unbiased_th, avg_grad, lr)
        
            #print(f"Variance of the gradient estimator : {variance_gradients(group1, group2, grads_all , epsilons_access, epsilons_active, epsilons_nonactive , M_t)}")
            #apply_gradient_clipped(global_unbiased_th, avg_grad, lr, 1.0)
            #apply_gradient_clipped(global_unbiased_th, avg_grad, lr , 5)
            #apply_gradient_adam(global_unbiased_th , avg_grad , optimizer)
            unbiased_th_loss_rnd = test_model(global_unbiased_th, dataloaders[1])   
            unbiased_loss_th_rnds.append(unbiased_th_loss_rnd)
            """
            breaking the loop
            if previous_loss < unbiased_th_loss_rnd : 
                break
            previous_loss = unbiased_th_loss_rnd
            """
            #unbiased_th_loss_rnds.append(unbiased_th_loss_rnd)
            print(f"loss : {unbiased_th_loss_rnd}")
            
        #unbiased_th_loss_rnds.append(unbiased_th_loss_rnd if grads_all else np.nan)
    unbiased_th_loss = test_model(global_unbiased_th, dataloaders[1]) 

    #print(f"M1={M1}, M2={M2}, e1={e1}, e2={e2}")
    #print(f"BIASED Loss: {biased_loss:.4f}")
    #print(f"UNBIASED Loss (Real): {unbiased_loss:.4f}") #without epsilons 
    #print(f"UNBIASED Loss (Theoritical by dividing by the biais): {unbiased_th_loss:.4f}") #without epsilons 
    #print(f"Parameter Bias Magnitude: {param_bias:.6f}")
    return biased_loss_rnds , unbiased_loss_rnds , unbiased_loss_th_rnds  ,  rounds_biaised , rounds_unbiased_th
    #rounds = list(range(1, N  + 1)) #Unbiaised
    

def main(): 

    
    set_seed(42)
    biased_loss_rnds , unbiased_loss_rnds , unbiased_loss_th_rnds , rounds , rounds2 = run_experiment(M1=5, M2= 2, M3 = 2, M4 = 1, e1=0.02, e2=0.93 , e3 = 0.95  ,e4 = 0.92 ,N=500) 
    
    
    biased_losses = [i[0] for i in biased_loss_rnds]
    unbiased_losses = [i[0] for i in unbiased_loss_rnds]
    unbiased_th_losses = [i[0] for i in unbiased_loss_th_rnds]
    
 
    
    plt.plot(rounds, smooth(biased_losses), label="Biased", linestyle='-', marker='o')
    plt.plot(rounds2, smooth(unbiased_th_losses) , label="Debiaised", linestyle='-', marker='o',color = 'red')
    plt.plot(rounds, smooth(np.array(unbiased_losses)[rounds]), label="Unbiased", linestyle='--', marker='s')

    plt.xlabel("Communication Round")
    plt.ylabel("Training Loss")
    plt.title("Comparison of Training Loss Over Rounds (e1 = 0.02 , e2 = 0.93, e3 = 0.95 ,e4 = 0.92) 1 label per client , M1 = 5, M2 = 2 , M3 = 2 , M4 = 1. lr = 0.01 ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.ylim(0, 4) 
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
