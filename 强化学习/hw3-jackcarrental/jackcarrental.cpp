#include <ctime>
#include <random>
#include <utility>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <fstream>
#include <string>
#include <numeric> // for accumulate
#include <vector>

#include "json.hpp"

#define IMPOSSIBLE -1e9 // if some state is impossible to reach, set its value to IMPOSSIBLE
using PolicyTable = std::array<std::array<int, 21>, 21>;
using namespace std;
using json = nlohmann::json;

class my_poisson_distribution{
    private:
        double lambda;
        double e_neg_lambda;
    public:
        my_poisson_distribution(double lambda): lambda(lambda), e_neg_lambda(exp(-lambda)) {}
        double operator()(int k) const {
            if (k < 0) {
                cerr << "Poisson distribution k < 0 error!" << endl;
                return 0.0;
            }
            return e_neg_lambda * pow(lambda, k) / tgamma(k + 1); // k! = tgamma(k+1
        }
};



// for the policy is deterministic, we can use an array of size 21*21 to store the policy for each state
// if action = a, move -a cars from location 1 to location 2 

class policy_iteration{
    private:
        static const double
            GAMMA,
            THETA;
        static const int
            MAX_CAR_1 = 20,
            MAX_CAR_2 = 20,
            MOVE_LIMIT = 5;


        // removed big static array on stack; use heap-allocated vectors instead
        std::vector<double> transition_prob_table;   // stores probabilities
        std::vector<double> transition_reward_table; // stores expected rewards

        // dims for indexing
        static constexpr int D0 = 21, D1 = 21, D2 = 11, D3 = 21, D4 = 21; // 21*21*11*21*21
        // D0: cars at loc1, D1: cars at loc2, D2: action+5, D3: next cars at loc1, D4: next cars at loc2
        static constexpr size_t TRANSITION_SIZE = (size_t)D0 * D1 * D2 * D3 * D4;

        inline size_t tindex(int a, int b, int c, int d, int e) const {
            // a in [0..20], b in [0..20], c in [0..10] (action+5), d in [0..20], e in [0..20]
            return (((((size_t)a * D1 + b) * D2 + c) * D3 + d) * D4 + e);
        }


        bool policy_stable;
        
        int policy[21][21]; // action for each state

        my_poisson_distribution poisson_request_1{3.0}, poisson_request_2{4.0};
        my_poisson_distribution poisson_return_1{3.0}, poisson_return_2{2.0};

        // precompute possion probabilities for requests and returns

        // note: max request and return is 20, arr[20] use 1-poisson(0~19)
        double poisson_request_1_cache[21];
        double poisson_request_2_cache[21];
        double poisson_return_1_cache[21];
        double poisson_return_2_cache[21];
        double poisson_return_1_cumsum[21]; // cumulative sum for return_1
        double poisson_return_2_cumsum[21]; // cumulative sum for return_2
        
        double calculate_transition_prob(int curr_cars_1, int curr_cars_2, int action, int next_cars_1, int next_cars_2, double &expected_reward) const;
        void precompute_poisson_cache(){
            cout << "Precomputing Poisson probabilities..." << endl;
            for (int i=0; i<=19; i++){
                poisson_request_1_cache[i] = poisson_request_1(i);
                poisson_request_2_cache[i] = poisson_request_2(i);
                poisson_return_1_cache[i] = poisson_return_1(i);
                poisson_return_2_cache[i] = poisson_return_2(i);
            }
            poisson_request_1_cache[20] = 1.0 - accumulate(poisson_request_1_cache, poisson_request_1_cache + 20, 0.0);
            poisson_request_2_cache[20] = 1.0 - accumulate(poisson_request_2_cache, poisson_request_2_cache + 20, 0.0);
            poisson_return_1_cache[20] = 1.0 - accumulate(poisson_return_1_cache, poisson_return_1_cache + 20, 0.0);
            poisson_return_2_cache[20] = 1.0 - accumulate(poisson_return_2_cache, poisson_return_2_cache + 20, 0.0);
            
            
            poisson_return_1_cumsum[0] = poisson_return_1_cache[0];
            poisson_return_2_cumsum[0] = poisson_return_2_cache[0];
            for (int k = 1; k <= 20; ++k) {
                poisson_return_1_cumsum[k] = poisson_return_1_cumsum[k-1] + poisson_return_1_cache[k];
                poisson_return_2_cumsum[k] = poisson_return_2_cumsum[k-1] + poisson_return_2_cache[k];
            }
        }

    public:
        double V[21][21]; // value function
        PolicyTable get_policy() const { // used to save policy to json
            PolicyTable policy_arr{};
            for (int i = 0; i < 21; i++) {
                for (int j = 0; j < 21; j++) {
                    policy_arr[i][j] = policy[i][j]; // policy 是 int[21][21]
                }
            }
            return policy_arr;
        }
        void precompute_transition_prob_table();
        void policy_evaluation();
        void policy_improvement();
        policy_iteration(): policy_stable(false) {
            for (int i=0; i<=20; i++){
                for (int j=0; j<=20; j++){
                    V[i][j] = 0.0;
                    policy[i][j] = 0; // initial policy: do nothing
                }
            }
        } // constructor
        bool is_policy_stable() const { return policy_stable; }
};

const double
    policy_iteration::GAMMA = 0.9,
    policy_iteration::THETA = 1e-6;



void policy_iteration::precompute_transition_prob_table(){
    precompute_poisson_cache(); // ensure poisson caches are filled

    // allocate vectors on heap BEFORE any write
    // TRANSITION_SIZE should be D0*D1*D2*D3*D4 (e.g. 21*21*11*21*21)
    transition_prob_table.assign(TRANSITION_SIZE, 0.0);
    transition_reward_table.assign(TRANSITION_SIZE, 0.0);

    for (int i = 0; i <= 20; ++i) { // state
        for (int j = 0; j <= 20; ++j) {
            for (int a = -5; a <= 5; ++a) {
                int action = a; // action
                bool valid = (action <= i) && (-action <= j); // check validity for state (i,j)
                int ai = action + 5; // 0..10
                // cout << "Precomputing for state (" << i << ", " << j << "), action " << action << endl;
                for (int i2 = 0; i2 <= 20; ++i2) { // next state
                    for (int j2 = 0; j2 <= 20; ++j2) {
                        size_t idx = tindex(i, j, ai, i2, j2);
                        if (!valid) {
                            transition_prob_table[idx] = IMPOSSIBLE;
                            transition_reward_table[idx] = IMPOSSIBLE;
                            continue;
                        }
                        double expected_reward = 0.0;
                        double prob = calculate_transition_prob(i, j, action, i2, j2, expected_reward);
                        transition_prob_table[idx] = prob;
                        transition_reward_table[idx] = expected_reward;
                    }
                }
            }
        }
    }
}

double policy_iteration::calculate_transition_prob(
    int curr_cars_1, int curr_cars_2, int action,
    int next_cars_1, int next_cars_2, double &expected_reward) const
{
    const int MAX_IDX = 20; // index 20 == ">=20" (tail)
    int early_next_cars_1 = curr_cars_1 - action;
    int early_next_cars_2 = curr_cars_2 + action;

    expected_reward = 0.0;
    double transition_prob = 0.0;
    double joint_reward_sum = 0.0; // accumulate prob*reward

    // local aliases for speed (optional)
    const auto &p_req1 = poisson_request_1_cache;
    const auto &p_req2 = poisson_request_2_cache;
    const auto &p_ret1 = poisson_return_1_cache;
    const auto &p_ret2 = poisson_return_2_cache;
    const auto &ret1_cumsum = poisson_return_1_cumsum;
    const auto &ret2_cumsum = poisson_return_2_cumsum;

    // helper to get sum of return probs from L..20 inclusive:
    auto sum_ret1_from = [&](int L)->double{
        if (L <= 0) return ret1_cumsum[20];
        if (L > 20) return 0.0;
        return ret1_cumsum[20] - ret1_cumsum[L-1];
    };
    auto sum_ret2_from = [&](int L)->double{
        if (L <= 0) return ret2_cumsum[20];
        if (L > 20) return 0.0;
        return ret2_cumsum[20] - ret2_cumsum[L-1];
    };

    // enumerate requests only (0..20). returns handled via prefix sums
    for (int req1 = 0; req1 <= MAX_IDX; ++req1) {
        // treat req1==MAX_IDX as ">=20": when computing served, just cap by early_next_cars_1
        int req1_val_for_served = (req1 == MAX_IDX) ? MAX_IDX : req1;
        for (int req2 = 0; req2 <= MAX_IDX; ++req2) {
            int req2_val_for_served = (req2 == MAX_IDX) ? MAX_IDX : req2;

            int served1 = std::min(req1_val_for_served, early_next_cars_1);
            int served2 = std::min(req2_val_for_served, early_next_cars_2);

            int after_rent_1 = early_next_cars_1 - served1; // >= 0
            int after_rent_2 = early_next_cars_2 - served2; // >= 0

            double p_requests = p_req1[req1] * p_req2[req2];
            if (p_requests == 0.0) continue; // small pruning

            double reward_for_this_request = (served1 + served2) * 10.0 - 2.0 * std::abs(action);

            // compute sum of return probabilities that lead to next_cars_1
            double sum_p_ret1 = 0.0;
            if (next_cars_1 < 20) {
                int needed_ret1 = next_cars_1 - after_rent_1;
                if (needed_ret1 >= 0 && needed_ret1 < 20) {
                    sum_p_ret1 = p_ret1[needed_ret1];
                } else {
                    sum_p_ret1 = 0.0;
                }
            } else { // next_cars_1 == 20 (tail)
                int threshold1 = 20 - after_rent_1; // need ret1 >= threshold1
                if (threshold1 <= 0) sum_p_ret1 = ret1_cumsum[20]; // all mass
                else if (threshold1 > 20) sum_p_ret1 = 0.0;
                else sum_p_ret1 = sum_ret1_from(threshold1);
            }

            if (sum_p_ret1 == 0.0) continue; // no possible returns -> skip

            // compute sum of return probabilities that lead to next_cars_2
            double sum_p_ret2 = 0.0;
            if (next_cars_2 < 20) {
                int needed_ret2 = next_cars_2 - after_rent_2;
                if (needed_ret2 >= 0 && needed_ret2 < 20) {
                    sum_p_ret2 = p_ret2[needed_ret2];
                } else {
                    sum_p_ret2 = 0.0;
                }
            } else { // next_cars_2 == 20
                int threshold2 = 20 - after_rent_2;
                if (threshold2 <= 0) sum_p_ret2 = ret2_cumsum[20];
                else if (threshold2 > 20) sum_p_ret2 = 0.0;
                else sum_p_ret2 = sum_ret2_from(threshold2);
            }

            if (sum_p_ret2 == 0.0) continue;

            // total probability for this (req1,req2) aggregated over all valid returns:
            double prob = p_requests * (sum_p_ret1 * sum_p_ret2);

            transition_prob += prob;
            joint_reward_sum += prob * reward_for_this_request;
        }
    }

    if (transition_prob > 0.0) expected_reward = joint_reward_sum / transition_prob;
    else expected_reward = 0.0;

    return transition_prob;
}

void policy_iteration::policy_improvement(){
    policy_stable = true;
    for (int i=0; i<=20; i++){ // state s
        for (int j=0; j<=20; j++){
            int old_action = policy[i][j];
            double max_action_value = -1e9;
            int best_action = 0xFFFF; // invalid action
            for (int a=-5; a<=5; a++){ // action
                int action = a;
                double action_value = 0.0;
                // compute action_value = sum_{s', r} p(s', r | s, a) [r + gamma * V(s')]
                for (int i2=0; i2<=20; i2++){ // next state s'
                    for (int j2=0; j2<=20; j2++){
                        size_t idx = tindex(i, j, action+5, i2, j2);
                        double prob = transition_prob_table[idx];
                        if (prob == IMPOSSIBLE) continue;
                        assert(action <= i && -action <= j); // action must be valid
                        double reward = transition_reward_table[idx];
                        action_value += prob * (reward + GAMMA * V[i2][j2]);
                    }
                }
                if (action_value > max_action_value){
                    max_action_value = action_value;
                    best_action = action;
                }
            }
            policy[i][j] = best_action;
            if (best_action != old_action){
                policy_stable = false;
            }
        }
    }
}

// policy_evaluation: evaluate the value function under current policy, return the max difference
void policy_iteration::policy_evaluation(){

    double delta = 0.0;
    do {
        delta = 0.0;
        for (int i=0; i<=20; i++){ // state s
            for (int j=0; j<=20; j++){
                double v = V[i][j];
                double new_v = 0.0;
                int action = policy[i][j]; // action under current policy
                // compute new_v = sum_{s', r} p(s', r | s, a) [r + gamma * V(s')]
                for (int i2=0; i2<=20; i2++){ // next state s'
                    for (int j2=0; j2<=20; j2++){
                        size_t idx = tindex(i, j, action+5, i2, j2);
                        double prob = transition_prob_table[idx];
                        if (prob == IMPOSSIBLE) continue;
                        double reward = transition_reward_table[idx];
                        new_v += prob * (reward + GAMMA * V[i2][j2]);
                    }
                }
                delta = max(delta, abs(v - new_v));
                V[i][j] = new_v;
            }
        }
    } while (delta >= THETA);
}


void save_policy_to_json(const std::string& filename,int iteration,const std::vector<PolicyTable>& all_policy){
    json data;
    data["iteration"] = iteration;
    data["policy"] = json::array();
    for (const auto& policy : all_policy) {
        json policy_json = json::array();
        for (int i = 0; i <= 20; i++) {
            std::vector<int> row;
            for (int j = 0; j <= 20; j++) {
                row.push_back(policy[i][j]);
            }
            policy_json.push_back(row);
        }
        data["policy"].push_back(policy_json);
    }
    std::ofstream fout(filename);
    fout << data.dump(2); // pretty print
}

void save_value_to_json(const std::string& filename, const policy_iteration& pi){
    json data;
    data["value"] = json::array();
    for (int i = 0; i <= 20; i++) {
        std::vector<double> row;
        for (int j = 0; j <= 20; j++) {
            row.push_back(pi.V[i][j]);
        }
        data["value"].push_back(row);
    }
    std::ofstream fout(filename);
    fout << data.dump(2); // pretty print
}


int main(){


    cout << "Jack's Car Rental Problem - Policy Iteration" << endl;
    policy_iteration pi;
    cout << "Precomputing transition probability table..." << endl;
    pi.precompute_transition_prob_table();
    int cnt = 0;
    std::vector<PolicyTable> all_policy;
    while (pi.is_policy_stable() == false){
        // save current policy to json
        all_policy.push_back(pi.get_policy());
        cnt ++;
        pi.policy_evaluation();
        pi.policy_improvement();
        if (pi.is_policy_stable()){
            cout << "Policy converged after " << cnt << " iterations." << endl;
            break;
        } else {
            cout <<"The iteration "<< cnt << ": Policy improved." << endl;
        }
    }
    // save multi policy to json
    string policy_json = "./saved_policy.json";
    save_policy_to_json(policy_json,cnt,all_policy);
    save_value_to_json("./saved_value.json", pi);
    cout << "Saved all policies to " << policy_json << endl;
    return 0;
}


// double policy_iteration::calculate_transition_prob(int curr_cars_1, int curr_cars_2, int action, int next_cars_1, int next_cars_2, double &expected_reward) const {
//     // calculate p(s'|s,a) and expected reward
//             // enumerate all possible requests and returns


//             int early_next_cars_1 = curr_cars_1 - action;
//             int early_next_cars_2 = curr_cars_2 + action;

//             // simulate the process
//             expected_reward = 0.0;
//             double transition_prob = 0.0;
//             for(int rent_1=0; rent_1<=early_next_cars_1; rent_1++){// enumerate all possible requests
//                 for(int rent_2=0; rent_2<=early_next_cars_2; rent_2++){

//                     double reward = (rent_1 + rent_2) * 10.0 - abs(action) * 2.0;

//                     int after_rent_cars_1 = early_next_cars_1 - rent_1;
//                     int after_rent_cars_2 = early_next_cars_2 - rent_2;
//                     // know next_cars_1, next_cars_2, after_rent_cars_1, after_rent_cars_2, the return must satisfy
//                     int return_1 = next_cars_1 - after_rent_cars_1;
//                     int return_2 = next_cars_2 - after_rent_cars_2;

//                     // for example, if next_cars_1 = 20, after_rent_cars_1 = 18, return_1 can be 2,3,4,...
//                     // if next_cars_1 < 20, return_1 must be exactly next_cars_1 - after_rent_cars_1
//                     // if return_1 < 0, impossible
//                     if (return_1 < 0 || return_2 < 0) continue; // impossible
//                     if (next_cars_1 == 20 || next_cars_2 == 20){// return maybe more than 20
//                         double prob = poisson_request_1_cache[rent_1] * poisson_request_2_cache[rent_2] * poisson_return_1_cache[return_1] * poisson_return_2_cache[return_2];
//                         if (next_cars_1 == 20 && next_cars_2 == 20){ // sum the extra probabilities, the return can be infinite
//                             for (int r1=return_1+1; r1<=20; r1++){
//                                 for (int r2=return_2+1; r2<=20; r2++){
//                                     prob += poisson_request_1_cache[rent_1] * poisson_request_2_cache[rent_2] * poisson_return_1_cache[r1] * poisson_return_2_cache[r2];
//                                 }
//                             }
//                         }
//                         else if (next_cars_1 == 20){ // sum the extra probabilities, the return can be infinite
//                             for (int r=return_1+1; r<=20; r++){
//                                 prob += poisson_request_1_cache[rent_1] * poisson_request_2_cache[rent_2] * poisson_return_1_cache[r] * poisson_return_2_cache[return_2];
//                             }
//                         }
//                         else if (next_cars_2 == 20){ // sum the extra probabilities
//                             for (int r=return_2+1; r<=20; r++){
//                                 prob += poisson_request_1_cache[rent_1] * poisson_request_2_cache[rent_2] * poisson_return_1_cache[return_1] * poisson_return_2_cache[r];
//                             }
//                         }
//                         transition_prob += prob;
//                         expected_reward += prob * reward;
//                     }
//                     else{// sum the probabilities and rewards, return is exactly next_cars_1 - after_rent_cars_1
//                         double prob = poisson_request_1_cache[rent_1] * poisson_request_2_cache[rent_2] * poisson_return_1_cache[return_1] * poisson_return_2_cache[return_2];
//                         transition_prob += prob;
//                         expected_reward += prob * reward;

//                     }         
//                 }
                
//             }
//             return transition_prob;
// }