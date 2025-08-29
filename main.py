from model import SimpleMNISTModel
from data_loader import get_dataloaders_iid
from server import average_models
from client import compute_stochastic_gradient
from server import average_gradients,average_gradients_unbiaised, apply_gradient , real_variance_gradients , monte_carlo_expectation
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
def run_experiment(M1, M2, e1, e2, N=500):
    #dataloaders,client_lengths = get_dataloaders()
    dataloaders = get_dataloaders_iid()
    group1 = list(range(M1))
    group2 = list(range(M1, M1 + M2))
    clients = list(range(M1+M2))
    epsilons = np.zeros(M1+M2)
    for cid in group1: epsilons[cid] = e1
    for cid in group2: epsilons[cid] = e2

    global_biased = SimpleMNISTModel()
    lr = 0.1
    K = 5

    biased_loss_rnds= []
    unbiased_loss_rnds = []
    unbiased_th_loss_rnds = []
    rounds_biaised = []
    # Biased Training
    print("Biased Training")
    for rnd in range(0):
        M_t = 0
        print(f"\n--- Communication Round {rnd+1} ---")
        grads_all = []
        for cid in dataloaders:
            #Calculate the probability of participation of the client
            if np.random.binomial(1, 1 - epsilons[cid]):
                M_t += 1
                local = SimpleMNISTModel()
                local.load_state_dict(global_biased.state_dict()) #Get the latest global model 
                grad = compute_stochastic_gradient(local, dataloaders[cid], K)
                grads_all.append(grad)
        #print(f"Number of participating clients : {len(grads_all)}")
        if M_t != 0:
            avg_grad = average_gradients(grads_all,M_t)
            apply_gradient(global_biased, avg_grad, lr)
            biased_loss_rnd = test_model(global_biased, dataloaders[1]) 
            rounds_biaised.append(rnd)
            #biased_loss_rnds.append(biased_loss_rnd)
            biased_loss_rnds.append(biased_loss_rnd )#if grads_all else np.nan)
            print(f"loss : {biased_loss_rnd}")
    biased_loss = test_model(global_biased, dataloaders[1]) 

     # Unbiased Training
    M_t = 5
    print("Unbiaised Training")
    global_unbiased = SimpleMNISTModel()
    for rnd in range(0):
        print(f"\n--- Communication Round {rnd+1} ---")
        grads_all = []
        for cid in dataloaders:
            local = SimpleMNISTModel()
            local.load_state_dict(global_unbiased.state_dict())
            grad = compute_stochastic_gradient(local, dataloaders[cid], K)
            grads_all.append(grad)
        avg_grad = average_gradients(grads_all,M_t)
        apply_gradient(global_unbiased, avg_grad, lr)
        unbiased_loss_rnd = test_model(global_unbiased, dataloaders[1]) 
        unbiased_loss_rnds.append(unbiased_loss_rnd)
        print(f"loss : {unbiased_loss_rnd}")


    unbiased_loss = test_model(global_unbiased, dataloaders[1])
    #print(f"Server - Test Loss: {unbiased_loss:.4f}")

    #Quantifiying the biais
    param_bias = sum(torch.norm(p1 - p2).item()
                     for p1, p2 in zip(global_biased.parameters(), global_unbiased.parameters()))


    #Reducing the value of the stepsize gives us the loss of the unbiaised estimation of the gradients (using the theoritical formula)
    lr = 0.01
    # UnBiased Theoritical Training (dividing by the biais)
    global_unbiased_th = SimpleMNISTModel()
    biases = []
    for m in clients:
        bias = monte_carlo_expectation(m,epsilons,N = 10000)
        biases.append(bias)
    #print(f"biases : {biases}")
    print("Unbiaised with dividing")
    for rnd in range(N):
        M_t = 0
        print(f"\n--- Communication Round {rnd+1} ---")
        grads_all = []
        for cid in dataloaders:
            #Calculate the probability of participation of the client
            if np.random.binomial(1, 1 - epsilons[cid]):
                local = SimpleMNISTModel()
                local.load_state_dict(global_unbiased_th.state_dict()) #Get the latest global model 
                grad = compute_stochastic_gradient(local, dataloaders[cid], K)
                grads_all.append(grad)
                M_t += 1
                
        #print(f"Number of participating clients : {len(grads_all)}")
        if M_t != 0:

            avg_grad = average_gradients_unbiaised(grads_all,epsilons,biases)
            #print(f"variance of gradient estimator : {real_variance_gradients(grads_all , epsilons)}")
            apply_gradient(global_unbiased_th, avg_grad, lr)
            unbiased_th_loss_rnd = test_model(global_unbiased_th, dataloaders[1]) 
            #unbiased_th_loss_rnds.append(unbiased_th_loss_rnd)
            print(f"loss : {unbiased_th_loss_rnd}")
        unbiased_th_loss_rnds.append(unbiased_th_loss_rnd if grads_all else np.nan)
    unbiased_th_loss = test_model(global_unbiased_th, dataloaders[1]) 


    print(f"M1={M1}, M2={M2}, e1={e1}, e2={e2}")
    print(f"BIASED Loss: {biased_loss:.4f}")
    print(f"UNBIASED Loss (Real): {unbiased_loss:.4f}") #without epsilons 
    print(f"UNBIASED Loss (Theoritical by dividing by the biais): {unbiased_th_loss:.4f}") #without epsilons 
    print(f"Parameter Bias Magnitude: {param_bias:.6f}")
    return biased_loss_rnds , unbiased_loss_rnds , rounds_biaised
    #rounds = list(range(1, N  + 1)) #Unbiaised
    

def main():

    biased_loss_rnds_1 , unbiased_loss_rnds_1 , rounds = run_experiment(M1=3, M2=2, e1=0.7, e2=0.7 , N=1000) #Clients with bigger datasets having lower participation probability
    #biased_loss_rnds_2 , unbiased_loss_rnds_2 , rounds2 = run_experiment(M1=3, M2=2, e1=0.9, e2=0.9 , N=500) #Clients with bigger datasets having lower participation probability

    #rounds = rounds_biaised[:-50]
    plt.figure(figsize=(10, 6))
    #plt.plot(rounds, smooth(biased_loss_rnds), label="Biased", linestyle='-', marker='o')
    #plt.plot(rounds2, smooth(biased_loss_rnds_2), label="epsilon = 0.9", linestyle='-', marker='o',color = 'red')
    plt.plot(rounds, smooth(biased_loss_rnds_1), label="epsilon = 0.7", linestyle='-', marker='o')
    plt.plot(rounds, np.array(unbiased_loss_rnds_1)[rounds], label="epsilon = 0", linestyle='--', marker='s')
    

    #plt.plot(rounds, unbiased_loss_rnds[rounds_biaised], label="Unbiased (Real)", linestyle='--', marker='s')
    #plt.plot(rounds, unbiased_th_loss_rnds, label="Unbiased (Theoretical)", linestyle='-.', marker='^')

    plt.xlabel("Communication Round")
    plt.ylabel("Test Loss")
    plt.title("Comparison of Test Loss Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0, 1) 
    plt.show()  
    
    
    
    #run_experiment(M1=1, M2=4, e1=0.5, e2=0.1)
    #run_experiment(M1=3, M2=2, e1=0.1, e2=0.5)
    """
    for (M1, M2) in [(4,1),(3,2),(2, 3),(1, 4)]:
        for e1 in [0.1, 0.3, 0.5, 0.7]:
            e2 = 0.1  
            print(f"\nRunning experiment with M1={M1}, M2={M2}, e1={e1}, e2={e2}")
            run_experiment(M1, M2, e1, e2)
    """

   
if __name__ == "__main__":
    main()
