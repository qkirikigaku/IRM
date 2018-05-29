## Make figures from result

import make_simulation_data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ## Read result
    input_file = 'result/parameter.txt'
    input = open(input_file, 'r')
    log_posterior = input.readline()
    clusters = input.readline().split()
    cluster_num1 = int(clusters[0])
    cluster_num2 = int(clusters[1])
    input_conf = open('IRM.conf', 'r')
    K = int(input_conf.readline().split()[1])
    L = int(input_conf.readline().split()[1])
    S1 = []
    S2 = []
    S1_result = input.readline().split()
    for x in S1_result:
        S1.append(int(x))
    S2_result = input.readline().split()
    for x in S2_result:
        S2.append(int(x))

    Table1 = np.zeros(cluster_num1)
    Table2 = np.zeros(cluster_num2)
    for S in S1:
        Table1[S] += 1
    for S in S2:
        Table2[S] += 1

    aligned_Table1 = np.argsort(Table1)[::-1]
    for i,x in enumerate(S1):
        S1[i] = np.where(aligned_Table1 == x)[0][0]
    aligned_Table2 = np.argsort(Table2)[::-1]
    for i,x in enumerate(S2):
        S2[i] = np.where(aligned_Table2 == x)[0][0]
    Table1 = np.sort(Table1)[::-1]
    Table2 = np.sort(Table2)[::-1]

    mat = np.zeros([cluster_num1, cluster_num2])
    for i in range(cluster_num1):
        for j in range(cluster_num2):
            mat[i,j] = np.random.rand()
    
    ## Calculate theta
    original_R = np.zeros([K,L])
    input_file = 'data/Simulation_data.txt'
    data = open(input_file, 'r')
    data.readline()
    for k in range(K):
        line = data.readline()
        R_list = line.split()
        for l in range(L):
            original_R[k,l] = int(R_list[l])

    theta_sum = np.zeros([cluster_num1, cluster_num2])
    theta = np.zeros([cluster_num1, cluster_num2])
    for k in range(K):
        i = S1[k]
        for l in range(L):
            j = S2[l]
            theta[i,j] += original_R[k,l]
            theta_sum[i,j] += 1
    for i in range(cluster_num1):
        for j in range(cluster_num2):
            theta[i,j] /= theta_sum[i,j]

    ## draw
    aligned_R = np.zeros([K,L])
    for k in range(K):
        for l in range(L):
            i = S1[k]
            j = S2[l]
            aligned_R[make_simulation_data.trans(k, S1, Table1), make_simulation_data.trans(l, S2, Table2)] = theta[i,j]
    fig_aligned_R = sns.heatmap(aligned_R, cmap = 'magma')
    sns.plt.savefig('result/aligned_R_theta.png')
    plt.close()

    ## Predicted_rate

if __name__ == '__main__':
    main()
