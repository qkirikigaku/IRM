#ifndef IRM_H
#define IRM_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <boost/math/special_functions/gamma.hpp>

#define PRINT(x,y) std::cout << x << " : " << y << std::endl
#define START_TABLE_NUM 3
#define MAX_K 500
#define MAX_L 500
#define MAX_I 10
#define MAX_J 10

using namespace std;

void PRINT_VEC(string Name, vector<int> &Vector_a){
    cout << Name << endl;
    cout << "[ ";
    for (int i=0; i < Vector_a.size(); i++){
        cout << Vector_a[i] << ", ";
    }
    cout << "]";
    cout << endl;
}

void PRINT_VEC(string Name, vector<double> &Vector_a){
    cout << Name << endl;
    cout << "[ ";
    for (int i=0; i < Vector_a.size(); i++){
        cout << Vector_a[i] << ", ";
    }
    cout << "]";
    cout << endl;
}

void PRINT_VEC2(string Name, vector<vector<int> > &Vector_a){
    cout << Name << endl;
    cout << "[ ";
    for (int i=0; i < Vector_a.size(); i++){
        cout << "[ ";
        for (int j=0; j < Vector_a[i].size(); j++){
            cout << Vector_a[i][j] << ", ";
        }
        cout << "]";
    }
    cout << "]";
    cout << endl;
}

void PRINT_VEC3(string Name, vector<vector<vector<int> > > &Vector_a){
    cout << Name << endl;
    cout << "[ ";
    for (int i=0; i < Vector_a.size(); i++){
        cout << "[";
        for (int j=0; j < Vector_a[i].size(); j++){
            cout << "[";
            for (int k=0; k < Vector_a[i][j].size(); k++){
                cout << Vector_a[i][j][k] << ", ";
            }
            cout << "],";
        }
        cout << "],";
    }
    cout << "]";
    cout << endl;
}

class IRM {
private:
    int K, L;
    double a, b, alpha;
    vector<vector<int> > R; // Relational Data Matrix (K*L)
    int cluster_num_I, cluster_num_J, temp_cluster_num_I, temp_cluster_num_J, New_cluster_num_I, New_cluster_num_J;
    vector<int> Sk, temp_Sk, New_Sk;
    vector<int> Sl, temp_Sl, New_Sl;
    vector<int> Table_k, temp_Table_k, New_Table_k;
    vector<int> Table_l, temp_Table_l, New_Table_l;
    // vector<vector<double>> theta;
    
    vector<vector<double> > log_sampler_Sk; // log_sampler_Sk[k][i] = log p(s_k = i)
    vector<vector<double> > log_sampler_Sl; // log_sampler_Sl[l][j] = log p(s_l = j)
    vector<vector<int> > n_kl, n_kl_bar; // n_kl(i,j) = sam_k sam_l R_kl * delta_k(i) * delta_l(j)
    vector<vector<int> > n_i, n_j; // n_i[k](i),n_j[l](j)
    vector<vector<vector<int> > > n_k, n_k_bar; // n_k[k](i,j) = sam_l R_kl * delta_k(i) * delta_l(j), n_k_bar[k](i,j) = sam_l (1-R_kl) * delta_k(i) * delta_l(j)
    //vector<vector<vector<int> > > n_k_inverse, n_k_inverse_bar;
    vector<vector<vector<int> > > n_l, n_l_bar; // n_l_inverse, n_l_inverse_bar; // n_l[l](i,j) = sam_k R_kl * delta_k(i) * delta_l(j), n_l_bar[l](i,j) = sam_k (1-R_kl) * delta_k(i) * delta_l(j)

    double temp_log_max_posterior, old_log_max_posterior;

public:
    IRM();
    void initialize();
    int cast();
    void load_relational_data_matrix();
    void load_hyper_parameter_from_text();

    void Collapsed_Gibbs_sampling();
    void Update_belonged_cluster();
    void Calc_n();
    double log_beta(double a, double b);
    void Update_table(vector<int> &S, vector<int> &Table, int &cluster_num);
    void Normalize(vector<double> &unnormalized_vector, int &length);
    double log_sum_exp(vector<double> &Vector_a, int &length);
    int Sampler(vector<double> &log_sampler, int &cluster_num);
    
    void Calc_posterior();
    void Maximize_posterior();

    void Print_progress();
    void write_data_to_text();

    void test();
    void test_show();
};

IRM::IRM():
    log_sampler_Sk(MAX_K, vector<double>(MAX_I)), log_sampler_Sl(MAX_L, vector<double>(MAX_J)),
    n_kl(MAX_I, vector<int>(MAX_J)), n_kl_bar(MAX_I, vector<int>(MAX_J)),
    n_k(MAX_K, vector<vector<int> >(MAX_I, vector<int>(MAX_J))), n_l(MAX_L, vector<vector<int> >(MAX_I, vector<int>(MAX_J))),
    n_k_bar(MAX_K, vector<vector<int> >(MAX_I, vector<int>(MAX_J))), n_l_bar(MAX_L, vector<vector<int> >(MAX_I, vector<int>(MAX_J))),
    n_i(MAX_K, vector<int>(MAX_I)), n_j(MAX_L, vector<int>(MAX_J))
{
    a = b = alpha = 1.0;
}

void IRM::initialize(){
    cluster_num_I = cluster_num_J = temp_cluster_num_I = temp_cluster_num_J = START_TABLE_NUM;
    Sk.resize(K, 0); temp_Sk.resize(K, 0); New_Sk.resize(K, 0);
    Sl.resize(L, 0); temp_Sl.resize(L, 0); New_Sl.resize(L, 0);
    for (int k=0; k < K; k++){
        Sk[k] = cast();
    }
    for (int l=0; l < L; l++){
        Sl[l] = cast();
    }
    Table_k.resize(START_TABLE_NUM); temp_Table_k.resize(START_TABLE_NUM);
    Table_l.resize(START_TABLE_NUM); temp_Table_l.resize(START_TABLE_NUM);
    Update_table(Sk, Table_k, cluster_num_I); Update_table(Sl, Table_l, cluster_num_J);
    temp_Sk = Sk; temp_Sl = Sl; New_Sk = Sk; New_Sl = Sl;
    log_sampler_Sk.resize(K); log_sampler_Sl.resize(L);
    for (int k=0; k < K; k++){
        log_sampler_Sk[k].resize(cluster_num_I);
    }
    for (int l=0; l < L; l++){
        log_sampler_Sl[l].resize(cluster_num_J);
    }
    cout << "\tComplete to initialize." << endl;
};

int IRM::cast(){
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> Uniform(0, START_TABLE_NUM-1);
    int value = Uniform(mt);
    return value;
};

void IRM::load_relational_data_matrix(){
    ifstream ifs;
    string input_file_name = "data/Simulation_data.txt";
    ifs.open(input_file_name.c_str(), ios::in);
    if(!ifs){
        cout << "Cannot open " + input_file_name << endl;
        exit(1);
    }
    char buf[1000000]; char *temp;
    ifs.getline(buf, 1000000);
    temp = strtok(buf, " ");
    K = atoi(temp);
    temp = strtok(NULL, " ");
    L = atoi(temp);

    R.resize(K);
    for (int k=0; k < K; k++){
        R[k].resize(L, 0);
        ifs.getline(buf, 1000000);
        for (int l=0; l < L; l++){
            if (l == 0) temp = strtok(buf, " ");
            else temp = strtok(NULL, " ");
            R[k][l] = atoi(temp);
        }
    }
    PRINT("K",K); PRINT("L",L);
    ifs.close();
};

void IRM::load_hyper_parameter_from_text(){
    ifstream ifs;
    string input_file_name = "data/Configuration.txt";
    ifs.open(input_file_name.c_str(), ios::in);
    if(!ifs){
        cout << "Cannot open " + input_file_name << endl;
        exit(1);
    }
    char buf[1000000];
    ifs.getline(buf, 1000000);
    a = atof(strtok(buf, " "));
    b = atof(strtok(NULL, " "));
    alpha = atof(strtok(NULL, " "));
    ifs.close();
    cout << "\tLoad hyper parameters from text." << endl;
};

void IRM::Collapsed_Gibbs_sampling(){

    cout << endl;
    cout << "Start Gibbs_sampling !!" << endl;
    initialize();
    load_hyper_parameter_from_text();
    Calc_posterior(); PRINT("initial_log_max_posterior", temp_log_max_posterior);
    old_log_max_posterior = temp_log_max_posterior;
    PRINT_VEC("Initial_Sk", Sk); PRINT_VEC("Initial_Sl", Sl); cout << endl;
    for(int iter=0; iter < 100; iter++){
        PRINT("Iter", iter);
        Update_belonged_cluster();
        PRINT("update belonged cluster.", "ok");
        Calc_posterior();
        PRINT("temp_log_max_posterior", temp_log_max_posterior);
        PRINT("old_log_max_posterior ", old_log_max_posterior);
        PRINT("calculdate posterior", "ok");
        Maximize_posterior();
        PRINT("compare max posterior", "ok");

        Print_progress();
        cout << endl;
    }
};

void IRM::Update_belonged_cluster(){
    int k,l,i,j;
    for (k=0; k < K; k++){
        temp_Sk[k] = -1;
        Update_table(temp_Sk, temp_Table_k, temp_cluster_num_I);
        log_sampler_Sk[k].clear();
        for (i=0; i < temp_cluster_num_I; i++){ 
            temp_Sk[k] = i;
            Update_table(temp_Sk, temp_Table_k, temp_cluster_num_I);
            Calc_n(); //FIXME
            double sum_log_beta = 0;
            for (j=0; j < temp_cluster_num_J; j++){
                sum_log_beta += log_beta((double) n_kl[i][j] + a, (double) n_kl_bar[i][j] + b);
                sum_log_beta -= log_beta((double) n_kl[i][j] - n_k[k][i][j] + a, (double) n_kl_bar[i][j] - n_k_bar[k][i][j] + b);
            }
            log_sampler_Sk[k].push_back(log((double) n_i[k][i]) - log((double) temp_cluster_num_I-1+alpha) + sum_log_beta);
        }
        temp_Sk[k] = temp_cluster_num_I;
        Update_table(temp_Sk, temp_Table_k, temp_cluster_num_I);
        Calc_n();
        double sum_log_beta = 0;
        for (j=0; j < temp_cluster_num_J; j++){
            sum_log_beta += log_beta((double) n_k[k][temp_cluster_num_I-1][j] + a, (double) n_k_bar[k][temp_cluster_num_I-1][j] + b);
            sum_log_beta -= log_beta(a, b);
        }
        log_sampler_Sk[k].push_back(log(alpha) - log((double) temp_cluster_num_I-1+alpha) + sum_log_beta);
        Normalize(log_sampler_Sk[k], temp_cluster_num_I);
        temp_Sk[k] = Sampler(log_sampler_Sk[k], temp_cluster_num_I);
        Update_table(temp_Sk, temp_Table_k, temp_cluster_num_I);
    }
    for (l=0; l < L; l++){
        temp_Sl[l] = -1;
        Update_table(temp_Sl, temp_Table_l, temp_cluster_num_J);
        log_sampler_Sl[l].clear();
        for (j=0; j < temp_cluster_num_J; j++){
            temp_Sl[l] = j;
            Update_table(temp_Sl, temp_Table_l, temp_cluster_num_J);
            Calc_n(); //FIXME
            double sum_log_beta = 0;
            for (i=0; i < temp_cluster_num_I; i++){
                sum_log_beta += log_beta((double) n_kl[i][j] + a, (double) n_kl_bar[i][j] + b);
                sum_log_beta -= log_beta((double) n_kl[i][j] - n_l[l][i][j] + a, (double) n_kl_bar[i][j] - n_l_bar[l][i][j] + b);
            }
            log_sampler_Sl[l].push_back(log((double) n_j[l][j]) - log((double) temp_cluster_num_J-1+alpha) + sum_log_beta);
        }
        temp_Sl[l] = temp_cluster_num_J;
        Update_table(temp_Sl, temp_Table_l, temp_cluster_num_J);
        Calc_n();
        double sum_log_beta = 0;
        for (i=0; i < temp_cluster_num_I; i++){
            sum_log_beta += log_beta((double) n_l[l][i][temp_cluster_num_J-1] + a, (double) n_l_bar[l][i][temp_cluster_num_J-1] + b);
            sum_log_beta -= log_beta(a, b);
        }
        log_sampler_Sl[l].push_back(log(alpha) - log((double) temp_cluster_num_J-1+alpha) + sum_log_beta);
        Normalize(log_sampler_Sl[l], temp_cluster_num_J);
        temp_Sl[l] = Sampler(log_sampler_Sl[l], temp_cluster_num_J);
        Update_table(temp_Sl, temp_Table_l, temp_cluster_num_J);
    }
};

void IRM::Calc_n(){
    int k,l,i,j;
    int I = temp_cluster_num_I, J = temp_cluster_num_J;

    n_kl.clear(); n_kl_bar.clear();
    n_i.clear(); n_j.clear(); n_k.clear(); n_l.clear(); n_k_bar.clear(); n_l_bar.clear();
    
    for (i=0; i < I; i++){
        vector<int> n_kl_i; vector<int> n_kl_bar_i;
        n_kl.push_back(n_kl_i); n_kl_bar.push_back(n_kl_bar_i);
        for (j=0; j < J; j++){
            n_kl[i].push_back(0); n_kl_bar[i].push_back(0);
        }
    }

    for (k=0; k < K; k++){
        vector<int> n_i_k;
        n_i.push_back(n_i_k);
        for (i=0; i < I; i++){
            n_i[k].push_back(temp_Table_k[i]);
            if(temp_Sk[k] == i) n_i[k][i] -= 1;
        }
    }

    for (l=0; l < L; l++){
        vector<int> n_j_l;
        n_j.push_back(n_j_l);
        for (j=0; j < J; j++){
            n_j[l].push_back(temp_Table_l[j]);
            if(temp_Sl[l] == j) n_j[l][j] -= 1;
        }
    }
    for (k=0; k < K; k++){
        vector<vector<int> > n_k_k; vector<vector<int> > n_k_bar_k;
        n_k.push_back(n_k_k); n_k_bar.push_back(n_k_bar_k);
        for (i=0; i < I; i++){
            vector<int> n_k_k_i; vector<int> n_k_bar_k_i;
            n_k[k].push_back(n_k_k_i); n_k_bar[k].push_back(n_k_bar_k_i);
            for (j=0; j < J; j++){
                n_k[k][i].push_back(0); n_k_bar[k][i].push_back(0);
            }
        }
    }
    
    for (l=0; l < L; l++){
        vector<vector<int> > n_l_l; vector <vector<int> > n_l_bar_l;
        n_l.push_back(n_l_l); n_l_bar.push_back(n_l_bar_l);
        for (i=0; i < I; i++){
            vector <int> n_l_l_i; vector <int> n_l_bar_l_i;
            n_l[l].push_back(n_l_l_i); n_l_bar[l].push_back(n_l_bar_l_i);
            for (j=0; j < J; j++){
                n_l[l][i].push_back(0); n_l_bar[l][i].push_back(0);
            }
        }
    }

    for (k=0; k < K; k++){
        int temp_i = temp_Sk[k];
        for (l=0; l < L; l++){
            int temp_j = temp_Sl[l];
            n_kl[temp_i][temp_j] += R[k][l]; n_kl_bar[temp_i][temp_j] += 1-R[k][l];
            n_k[k][temp_i][temp_j] += R[k][l]; n_k_bar[k][temp_i][temp_j] += 1-R[k][l];
            n_l[l][temp_i][temp_j] += R[k][l]; n_l_bar[l][temp_i][temp_j] += 1-R[k][l];
        }
    }
};

double IRM::log_beta(double a, double b){
    return(lgamma(a) + lgamma(b) - lgamma(a + b));
}

void IRM::Update_table(vector<int> &S, vector<int> &Table, int &cluster_num){
    int length = S.size();
    for (int i=0; i < length; i++){
        if(S[i] == cluster_num){
            cluster_num += 1;
        }
    }
    Table.resize(cluster_num);
    for (int i=0; i < cluster_num; i++){
        Table[i] = 0;
    }
    int k;
    for (int i=0; i < length; i++){
        if(S[i] != -1){
            Table[S[i]] += 1;
        }
        else{
            k = i;
        }
    }
    int flag=0;
    int missing_Table;
    for (int i=0; i < cluster_num; i++){
        if(Table[i] == 0){
            flag = 1;
            missing_Table = i;
        }
    }
    if (flag == 1){
        cluster_num -= 1;
        Table.erase(Table.begin() + missing_Table);
        Table.shrink_to_fit();
        for (int i=0; i < length; i++){
            if(S[i] > missing_Table){
                S[i] -= 1;
            }
        }
    }
};

void IRM::Normalize(vector<double> &unnormalized_vector, int &length){
    double sum = log_sum_exp(unnormalized_vector, length);
    int i;
    for (i=0; i < length; i++){
        unnormalized_vector[i] -= sum;
    }
};

double IRM::log_sum_exp(vector<double> &Vector_a, int &length){
    double max_iter = 0;
    for (int i=1; i < length; i++){
        if(Vector_a[i] > Vector_a[max_iter])max_iter = i;
    }
    double sum = 0;
    for (int i=0; i < length; i++){
        sum += exp(Vector_a[i] - Vector_a[max_iter]);
    }
    double return_value = Vector_a[max_iter] + log(sum);
    return(return_value);
};

int IRM::Sampler(vector<double> &log_sampler, int &cluster_num){
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> Uniform(0.0, 1.0);
    double value = Uniform(mt);
    double log_value = log(value);
    double logsum = log_sampler[0];
    int num = 0;
    while(1){
        if(log_value < logsum) break;
        else{
            num += 1;
            vector<double> part_log_sampler(num + 1);
            for (int i=0; i < num + 1; i++){
                part_log_sampler[i] = log_sampler[i];
            }
            int pass_num = num + 1;
            logsum = log_sum_exp(part_log_sampler, pass_num);
        }
        if(num == cluster_num){
            num -= 1;
            break;
        }
    }
    return num;
};

void IRM::Calc_posterior(){
    
    double log_p_sk = (double) temp_cluster_num_I * log(alpha);
    for (int i=0; i < temp_cluster_num_I; i++){
        log_p_sk += lgamma(temp_Table_k[i]);
    }
    log_p_sk += lgamma(alpha) - lgamma(alpha + K);
    log_p_sk -= lgamma(temp_cluster_num_I + 1);

    double log_p_sl = (double) temp_cluster_num_J * log(alpha);
    for (int i=0; i < temp_cluster_num_J; i++){
        log_p_sl += lgamma(temp_Table_l[i]);
    }
    log_p_sl += lgamma(alpha) - lgamma(alpha + K);
    log_p_sl -= lgamma(temp_cluster_num_J + 1);
    
    vector<vector<int> > temp_n_kl(temp_cluster_num_I, vector<int>(temp_cluster_num_J, 0));
    vector<vector<int> > temp_n_kl_bar(temp_cluster_num_I, vector<int>(temp_cluster_num_J, 0));
    for (int k=0; k < K; k++){
        int temp_i = temp_Sk[k];
        for (int l=0; l < L; l++){
            int temp_j = temp_Sl[l];
            temp_n_kl[temp_i][temp_j] += R[k][l]; temp_n_kl_bar[temp_i][temp_j] += 1-R[k][l];
        }
    }

    double marginalized_out_theta = 0;
    for (int i=0; i < temp_cluster_num_I; i++){
        for (int j=0; j < temp_cluster_num_J; j++){
            marginalized_out_theta += log_beta(temp_n_kl[i][j] + a, temp_n_kl_bar[i][j] + b);
            marginalized_out_theta -= log_beta(a, b);
        }
    }
    temp_log_max_posterior = log_p_sk + log_p_sl + marginalized_out_theta;   
};

void IRM::Maximize_posterior(){
    if(temp_log_max_posterior > old_log_max_posterior){
        Sk = temp_Sk; Update_table(Sk, Table_k, cluster_num_I);
        Sl = temp_Sl; Update_table(Sl, Table_l, cluster_num_J);
        old_log_max_posterior = temp_log_max_posterior;
    }
    else{
        temp_Sk = Sk; Update_table(temp_Sk, temp_Table_k, temp_cluster_num_I);
        temp_Sl = Sl;  Update_table(temp_Sl, temp_Table_l, temp_cluster_num_J);
    }
};

void IRM::Print_progress(){
    PRINT_VEC("Sk", Sk);
    PRINT_VEC("Sl", Sl);
    PRINT("C1", cluster_num_I);
    PRINT("C2", cluster_num_J);
    cout << endl;
    cout << endl;
}

void IRM::write_data_to_text(){
    ofstream ofs;
    string output_file_name = "result/parameter.txt";
    ofs.open(output_file_name, ios::out);
    // write log_max_posterior
    ofs << to_string(temp_log_max_posterior) << "\n";
    // write cluster num
    ofs << to_string(cluster_num_I) << " " << to_string(cluster_num_J) << "\n";
    // write clustering for Sk
    for (int k=0; k < K; k++){
        ofs << to_string(Sk[k]) << " ";
    }
    ofs << "\n";
    // write clustering for Sl
    for (int l=0; l < L; l++){
        ofs << to_string(Sl[l]) << " ";
    }
    ofs << "\n";
};

#endif //IRM_H
