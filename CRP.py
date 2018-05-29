import numpy as np

def generate(Seq, Table, alpha, max_customer_num):
    temp_customer_num = len(Seq) + 1
    probability = []
    for i in range(len(Table)):
        probability.append(Table[i] / (temp_customer_num-1+alpha))
    probability.append(alpha / (temp_customer_num-1+alpha))
    cast = np.random.multinomial(1, probability)
    temp_table = cast.tolist().index(1)
    Seq.append(temp_table)
    if(temp_table < len(Table)):
        Table[temp_table] += 1
    elif(temp_table == len(Table)):
        Table.append(1)
    else: print('ERROR ', Seq, Table)

    if(temp_customer_num == max_customer_num):
        return Seq, Table
    else:
        #print(Seq, Table)
        generate(Seq, Table, alpha, max_customer_num)
        return Seq, Table
