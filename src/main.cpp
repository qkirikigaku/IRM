#include "IRM.h"

#include <iostream>

void run_Collapsed_Gibbs_sampler();

int main(int argc, const char * argv[]){
    run_Collapsed_Gibbs_sampler();
}

void run_Collapsed_Gibbs_sampler(){
    IRM temp_object;
    cout << "Made IRM object." << endl;
    temp_object.load_relational_data_matrix();
    cout << "Complete loading data." << endl;
    
    //temp_object.initialize();
    //temp_object.load_hyper_parameter_from_text();
    //temp_object.test();

    temp_object.Collapsed_Gibbs_sampling();
    cout << "Complete Gibbs sampling..." << endl;
    temp_object.write_data_to_text();
    cout << "Complete writing data to text." << endl;
}
