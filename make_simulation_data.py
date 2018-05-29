## Make simulated data

import CRP

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

def main():

    ## Set parameters from IRM.conf
    input = open('IRM.conf', 'r')
    
    ## Varibables given by Data
    K = int(input.readline().split()[1])
    L = int(input.readline().split()[1])

    ## Hyper paramters
    a = float(input.readline().split()[1])
    b = float(input.readline().split()[1])
    alpha = float(input.readline().split()[1])

    ## Generate s^1, s^2
    S1, Table1 = CRP.generate([0], [1], alpha, K)
    #print(S1, Table1)
    S2, Table2 = CRP.generate([0], [1], alpha, L)
    #print(S2, Table2)
    cluster_num1 = len(Table1)
    cluster_num2 = len(Table2)

    theta = np.zeros([cluster_num1, cluster_num2]) ## Generate theta
    for i in range(cluster_num1):
        for j in range(cluster_num2):
            theta[i,j] = np.random.beta(a, b)

    R = np.zeros([K,L]) ## Generate R
    for k in range(K):
        i = S1[k]
        for l in range(L):
            j = S2[l]
            R[k,l] = np.random.binomial(1, theta[i,j])

    ## Save
    output = open('data/Simulation_data.txt', 'w') ## K, L, R
    output.write(str(K) + ' ' + str(L) + '\n')
    for k in range(K):
        for l in range(L):
            output.write(str(int(R[k,l])) + ' ')
        output.write('\n')
    output.close()
    output = open('data/Configuration.txt', 'w') ## a, b, alpha, Table1, Table2, theta
    output.write(str(a) + ' ' + str(b) + ' ' + str(alpha) + '\n')
    for i in range(cluster_num1):
        output.write(str(Table1[i]) + ' ')
    output.write('\n')
    for i in range(cluster_num2):
        output.write(str(Table2[i]) + ' ')
    output.write('\n')
    for i in range(cluster_num1):
        for j in range(cluster_num2):
            output.write(str(theta[i,j]) + ' ')
        output.write('\n')
    for x in S1:
        output.write(str(x) + ' ')
    output.write('\n')
    for x in S2:
        output.write(str(x) + ' ')
    output.write('\n')
    output.close()

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

    ## draw
    fig_theta = sns.heatmap(theta)
    sns.plt.savefig('data/theta.png')
    plt.close()
    fig_R = sns.heatmap(R)
    sns.plt.savefig('data/R.png')
    plt.close()
    aligned_R = np.zeros([K,L])
    for k in range(K):
        for l in range(L):
            aligned_R[trans(k, S1, Table1), trans(l, S2, Table2)] = R[k,l]
    fig_aligned_R = sns.heatmap(aligned_R)
    sns.plt.savefig('data/aligned_R.png')
    plt.close()
    aligned_R_ij = np.zeros([K,L])
    for k in range(K):
        for l in range(L):
            i = S1[k]
            j = S2[l]
            aligned_R_ij[trans(k, S1, Table1), trans(l, S2, Table2)] = theta[i,j]
    fig_aligned_R_ij = sns.heatmap(aligned_R_ij, cmap='magma')
    sns.plt.savefig('data/aligned_R_theta.png')
    plt.close()

def trans(index, Seq, Table):
    temp_table = Seq[index]
    target_index = 0
    for x in Table[:temp_table]:
        target_index += x
    for x in Seq[:index]:
        if(x == temp_table):
            target_index += 1
    return target_index

if __name__ == '__main__':
    main()
