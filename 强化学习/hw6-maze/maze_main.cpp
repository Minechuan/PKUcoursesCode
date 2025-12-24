#include <ctime>
#include "maze.hpp"
#include <iostream>
#include<vector>
#include<unordered_map>
#include<unordered_set>
#include<bitset>
#include<random>
#include <filesystem>
#include <fstream>
#include <iomanip>
namespace fs = std::filesystem;

using namespace std;


const int PLANNING_STEPS = 20;
const double ALPHA = 0.9;
const int N_RUNS = 5;
const double EPSILON = 0.1;
const double GAMMA = 0.95;
const double KAPPA = 0.0001; // for Dyna-Q+
std::mt19937 rng(1); 

void write_csv_row(const string &path, const vector<double> &vec) {
    ofstream ofs(path);
    if (!ofs) {
        cerr << "Failed to open " << path << " for writing\n";
        return;
    }
    ofs << fixed << setprecision(6);
    for (size_t i = 0; i < vec.size(); ++i) {
        ofs << vec[i];
        if (i + 1 < vec.size()) ofs << ",";
    }
    ofs << "\n";
    ofs.close();
}


class MazePolicyBase{
    public:
        virtual int operator()(const MazeEnv::State& state) const = 0;
    protected:
        std::unordered_map<int, std::bitset<4>> actions_seen_per_state;
        std::vector<int> seen_states_vec;  
        std::unordered_set<int> seen_states_set;

    void record_seen(int state_idx, int action) {
        auto it = actions_seen_per_state.find(state_idx);
        if (it == actions_seen_per_state.end()) {
            // first time seeing this state
            std::bitset<4> bs;
            bs.reset();
            bs.set(action);
            actions_seen_per_state.emplace(state_idx, bs);
            seen_states_set.insert(state_idx);
            seen_states_vec.push_back(state_idx);
        } else {
            it->second.set(action);
        }
    }

    std::pair<int,int> sample_state_action() { 
        if (seen_states_vec.empty()) 
            throw std::runtime_error("no seen states to sample"); 
        std::uniform_int_distribution<int> uni_state(0, (int)seen_states_vec.size() - 1); 
        int s_idx = seen_states_vec[uni_state(rng)]; 
        auto &bs = actions_seen_per_state[s_idx]; 
        std::vector<int> actions; 
        for (int a = 0; a < 4; ++a) 
            if (bs.test(a)) 
                actions.push_back(a); 
        if (actions.empty()) 
            throw std::runtime_error("no actions recorded for sampled state (should not happen)"); 
        std::uniform_int_distribution<int> uni_a(0, (int)actions.size() - 1); 
        int act = actions[uni_a(rng)]; 
        return {s_idx, act}; 
    }

};

class MazeDynaQ : public MazePolicyBase{
        public:
        int total_steps;
        double accumulated_reward = 0;
        double accumulated_array[100] = {0.0};
        int operator()(const MazeEnv::State& state) const {
            int best_action = 0;
            double best_value = q[locate(state, 0)];
            double q_s_a;
            for (int action = 1; action < 4; ++ action){
                q_s_a = q[locate(state, action)];
                if (q_s_a > best_value){
                    best_value = q_s_a;
                    best_action = action;
                }
            }
            return best_action;
        }

        MazeDynaQ(const MazeEnv& e1, const MazeEnv& e2, int change_step,int total_steps) : env1(e1), env2(e2), change_step(change_step), total_steps(total_steps) {
            epsilon = EPSILON;
            alpha = ALPHA;
            gamma = GAMMA;
            int size = e1.max_x * e1.max_y * 4;
            q = new double[size];
            n_planning = PLANNING_STEPS;
            std::uniform_real_distribution<double> uni_q(0.0, 0.1); // 小随机值
            for (int i = 0; i < size; ++i) q[i] = uni_q(rng);
        }

        ~MazeDynaQ(){
            delete []q;
        }

        void learn(bool verbose=false,double *accumulated_array=nullptr,int run_number=0){
            int action, next_action;
            double reward;
            MazeEnv* current_env_p = &env1;
            MazeEnv::State state;
            MazeEnv::StepResult step_result;
            bool done=false;
            bool if_changed = false;
            state = current_env_p->reset();
            int step = -1;
            while (true){
                step ++;
                if(step >= total_steps) break;
                if (done){
                    if (verbose) cout << "Episode finished after " << step << " steps." << endl;
                    state = current_env_p->reset();
                    done = false;
                    accumulated_reward ++; // reward for completing an episode
                }
                if (accumulated_array) accumulated_array[run_number * total_steps + step] = accumulated_reward;
                if (step >= change_step && !if_changed){
                    current_env_p = &env2;
                    if_changed = true;
                    state = current_env_p->reset();
                }
                current_env_p->set_state(state); // set env to current state

                // use epsilon-greedy to select action
                action = epsilon_greedy(state);
                // update seen states and actions
                int state_idx = state.second * env1.max_x + state.first; // idx = y * max_x + x
                record_seen(state_idx, action);
                step_result = current_env_p->step(action);
                MazeEnv::State next_state = step_result.next_state;
                reward = step_result.reward;
                done = step_result.done;

                // Q-learning update
                update_q(state, action, reward, next_state);
                state = next_state; // record state for next step

                // planning step
                MazeEnv::State planning_random_state;
                MazeEnv::State planning_next_state;
                int rand_action;
                
                for (int planning_step = 0; planning_step < n_planning; ++ planning_step){
                    // Randomly sample previously observed state-action pairs
                    // and update Q-values based on simulated experience
                    auto sample_result = sample_state_action();
                    planning_random_state = {sample_result.first % env1.max_x, sample_result.first / env1.max_x};
                    rand_action = sample_result.second;

                    if (not current_env_p->is_valid_state(planning_random_state)){
                        // After maze change, some previously valid states may become invalid
                        continue; 
                    }
                    current_env_p->set_state(planning_random_state);
                    step_result = current_env_p->step(rand_action);
                    planning_next_state = step_result.next_state;
                    // update Q-value
                    update_q(planning_random_state, rand_action, step_result.reward, planning_next_state);
                }
            }
        }

        int epsilon_greedy(MazeEnv::State state) const {
            std::uniform_real_distribution<double> uni01(0.0, 1.0);
            if (uni01(rng) < epsilon) {
                std::uniform_int_distribution<int> uni_a(0, 3);
                return uni_a(rng);
            }
            return (*this)(state);
        }

        inline int locate(MazeEnv::State state, int action) const {
            return state.second * env1.max_x * 4 + state.first * 4 + action;
        }

    private:
        MazeEnv env1;
        MazeEnv env2;
        int change_step; // the step to change maze
        double *q;
        double epsilon, alpha, gamma;
        int n_planning; // number of planning steps per real step

        void update_q(MazeEnv::State state, int action, double reward, MazeEnv::State next_state){
            int best_action = (*this)(next_state);
            q[locate(state, action)] += alpha * (reward + gamma * q[locate(next_state, best_action)] - q[locate(state, action)]);
        }
};


class MazeDynaQ_plus : public MazePolicyBase{
        public:
        int total_steps;
        double accumulated_reward = 0.0;
        int operator()(const MazeEnv::State& state) const {
            int best_action = 0;
            double best_value = q[locate(state, 0)];
            double q_s_a;
            for (int action = 1; action < 4; ++ action){
                q_s_a = q[locate(state, action)];
                if (q_s_a > best_value){
                    best_value = q_s_a;
                    best_action = action;
                }
            }
            return best_action;
        }

        MazeDynaQ_plus(const MazeEnv& e1, const MazeEnv& e2, int change_step,int total_steps) : env1(e1), env2(e2), change_step(change_step), total_steps(total_steps) {
            epsilon = EPSILON;
            alpha = ALPHA;
            gamma = GAMMA;
            kappa = KAPPA;
            int size = e1.max_x * e1.max_y * 4;
            q = new double[size];
            planning_n = PLANNING_STEPS;
            std::uniform_real_distribution<double> uni_q(0.0, 0.1); // 小随机值
            for (int i = 0; i < size; ++i) q[i] = uni_q(rng);
        }

        ~MazeDynaQ_plus(){
            delete []q;
        }

        void learn(bool verbose=false,double *accumulated_array=nullptr,int run_number=0){
            int action, next_action;
            double reward;
            MazeEnv* current_env_p = &env1;
            MazeEnv::State state;
            MazeEnv::StepResult step_result;
            bool done=false;
            long current_time = 0;
            for (int x = 0; x < env1.max_x; ++x) {
                for (int y = 0; y < env1.max_y; ++y) {
                    for (int a = 0; a < 4; ++a) {
                        int idx = y * env1.max_x * 4 + x * 4 + a;
                        last_visited_time[idx] = -1; // 等于 0
                    }
                }
            }
            bool if_changed = false;
            state = current_env_p->reset();
            int step = -1;
            while (true){
                step ++;
                if(step >= total_steps) break;
                current_time ++; // increment time
                if (done){
                    state = current_env_p->reset();
                    done = false;
                    accumulated_reward ++; // reward for completing an episode
                }
                if (accumulated_array) accumulated_array[run_number * total_steps + step] = accumulated_reward;
                if (step >= change_step && if_changed == false){
                    current_env_p = &env2;
                    state = current_env_p->reset();
                    if_changed = true;
                }
                current_env_p->set_state(state); // set env to current state
                // use epsilon-greedy to select action
                action = epsilon_greedy(state);
                // update seen states and actions
                int state_idx = state.second * env1.max_x + state.first; // idx = y * max_x + x
                record_seen(state_idx, action);
                last_visited_time[locate(state, action)] = current_time;

                step_result = current_env_p->step(action);
                MazeEnv::State next_state = step_result.next_state;
                reward = step_result.reward;
                done = step_result.done;

                // Q-learning update
                update_q(state, action, reward, next_state);
                state = next_state; // record state for next step

                // planning step
                MazeEnv::State planning_random_state;
                MazeEnv::State planning_next_state;
                int rand_action;
                for (int planning_step = 0; planning_step < planning_n; ++planning_step) {
                    // Randomly sample previously observed state-action pairs
                    // and update Q-values based on simulated experience
                    auto sample_result = sample_state_action();
                    planning_random_state = {sample_result.first % env1.max_x, sample_result.first / env1.max_x};
                    if( not current_env_p->is_valid_state(planning_random_state)){
                        continue;
                    }
                    rand_action = sample_result.second;
                    current_env_p->set_state(planning_random_state);
                    step_result = current_env_p->step(rand_action);
                    assert (last_visited_time[locate(planning_random_state, rand_action)]>=0);
                    double planning_reward = step_result.reward + kappa*sqrt(current_time - last_visited_time[locate(planning_random_state, rand_action)]); // Dyna-Q+ bonus
                    // cout << "planning_reward: " << current_time - last_visited_time[locate(planning_random_state, rand_action)]<<"  " <<planning_reward << endl;
                    planning_next_state = step_result.next_state;
                    // update Q-value
                    update_q(planning_random_state, rand_action, planning_reward, planning_next_state);
                }
            }
        }

        int epsilon_greedy(MazeEnv::State state) const {
            std::uniform_real_distribution<double> uni01(0.0, 1.0);
            if (uni01(rng) < epsilon) {
                std::uniform_int_distribution<int> uni_a(0, 3);
                return uni_a(rng);
            }
            return (*this)(state);
        }

        inline int locate(MazeEnv::State state, int action) const {
            return state.second * env1.max_x * 4 + state.first * 4 + action;
        }

    private:
        MazeEnv env1;
        MazeEnv env2;
        int change_step; // the step to change maze
        double *q;
        double epsilon, alpha, gamma;
        int planning_n; // number of planning steps per real step
        double kappa; // Dyna-Q+ bonus
        

        void update_q(MazeEnv::State state, int action, double reward, MazeEnv::State next_state){
            int best_action = (*this)(next_state);
            q[locate(state, action)] += alpha * (reward + gamma * q[locate(next_state, best_action)] - q[locate(state, action)]);
        }
        std::unordered_map<int, long> last_visited_time; // key: locate(state, action), value: last visited time
};



/*
Construct Needed Maze
*/


const int max_x = 9, max_y = 6;
const int start_x = 3, start_y = 5;
const int target_x = 8, target_y = 0;

int maze_right[max_y][max_x] = {
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0}
};


int maze_left[max_y][max_x] = {
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0},
    {0,1,1,1,1,1,1,1,1},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0}
};

int maze_both[max_y][max_x] = {
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0},
    {0,1,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0}
};

const int BLOCKING_MAZE_CHANGE_STEP = 1000;
const int BLOCKING_MAZE_TOTAL_STEPS = 3000;


const int SHORTCUT_MAZE_CHANGE_STEP = 3000;
const int SHORTCUT_MAZE_TOTAL_STEPS = 6000;


int main(){

    MazeEnv env_both(maze_both, max_x, max_y, start_x, start_y, target_x, target_y);
    MazeEnv env_right(maze_right, max_x, max_y, start_x, start_y, target_x, target_y);
    MazeEnv env_left(maze_left, max_x, max_y, start_x, start_y, target_x, target_y);

    cout << "Test policy on Blocking Maze:" << endl;


    

    /****************************************** 
        Blocking Maze Test
    ********************************************/

    cout << "-- Blocking Maze Test --" << endl;
    int n_points = BLOCKING_MAZE_TOTAL_STEPS; // 每一步都记录
    cout << "Number of sampling points: " << n_points << endl;
    // 用于累加每次 run 的数组
    vector<double> sum_dq(n_points, 0.0);
    vector<double> sum_dqp(n_points, 0.0);
    double* accumulated_array = new double[N_RUNS*n_points];

    // Test Dyna-Q
    cout << "Dyna-Q Test:" << endl;
    double total_reward = 0.0;
    for (int i = 0; i < N_RUNS; ++i) {
        for (int j = 0; j < n_points; ++j) {
            accumulated_array[i*n_points + j] = 0.0;
        }
    }
    for (int run = 0; run < N_RUNS; ++ run){
        MazeDynaQ dyna_q(env_right, env_left, BLOCKING_MAZE_CHANGE_STEP, BLOCKING_MAZE_TOTAL_STEPS);
        dyna_q.learn(false, accumulated_array, run);
        total_reward += dyna_q.accumulated_reward;
        cout << "  Run " << run + 1 << ": Accumulated reward: " << dyna_q.accumulated_reward << endl;
        for (int i = 0; i < n_points; ++i) {
            sum_dq[i] += accumulated_array[run * n_points + i];
        }
    }
    cout << "Dyna-Q Test: Mean accumulated reward: " << total_reward / N_RUNS << endl;

    // Test Dyna-Q+
    total_reward = 0.0;
    // set accumulated_array to zero
    for (int i = 0; i < N_RUNS; ++i) {
        for (int j = 0; j < n_points; ++j) {
            accumulated_array[i*n_points + j] = 0.0;
        }
    }
    for (int run = 0; run < N_RUNS; ++ run){
        MazeDynaQ_plus dyna_q_plus(env_right, env_left, BLOCKING_MAZE_CHANGE_STEP, BLOCKING_MAZE_TOTAL_STEPS);
        dyna_q_plus.learn(false, accumulated_array, run);
        total_reward += dyna_q_plus.accumulated_reward;
        cout << "  Run " << run + 1 << ": Accumulated reward: " << dyna_q_plus.accumulated_reward << endl;

        // write per-run CSV (optional)
        // string fname = "dyna_q_plus_runs/dyna_q_plus_run_" + to_string(run) + ".csv";
        // write_csv_row_from_raw(fname, dyna_q_plus.accumulated_array, n_points);

        // accumulate into sum vector
        for (int i = 0; i < n_points; ++i) {
            sum_dqp[i] += accumulated_array[run * n_points + i];
        }
    }
    cout << "Dyna-Q+ Test: Mean accumulated reward: " << total_reward / N_RUNS << endl;

    // 计算均值（平均曲线）
    vector<double> mean_dq(n_points, 0.0);
    vector<double> mean_dqp(n_points, 0.0);
    for (int i = 0; i < n_points; ++i) {
        mean_dq[i] = sum_dq[i] / static_cast<double>(N_RUNS);
        mean_dqp[i] = sum_dqp[i] / static_cast<double>(N_RUNS);
    }

    // 写出平均曲线 CSV（用于后续绘图）
    write_csv_row("dyna_q_mean.csv", mean_dq);
    write_csv_row("dyna_q_plus_mean.csv", mean_dqp);

    cout << "Saved mean curves to dyna_q_mean.csv and dyna_q_plus_mean.csv" << endl;
    cout << "-- Finished --" << endl << endl;

    /****************************************** 
        Shortcut Maze Test
    ********************************************/

    // int n_points = SHORTCUT_MAZE_TOTAL_STEPS; // 每一步都记录
    // cout << "Number of sampling points: " << n_points << endl;
    // // 用于累加每次 run 的数组
    // vector<double> sum_dq(n_points, 0.0);
    // vector<double> sum_dqp(n_points, 0.0);
    // double* accumulated_array = new double[N_RUNS*n_points];



    // // Test Dyna-Q
    // cout << "Dyna-Q Test:" << endl;
    // double total_reward = 0.0;
    // for (int i = 0; i < N_RUNS; ++i) {
    //     for (int j = 0; j < n_points; ++j) {
    //         accumulated_array[i*n_points + j] = 0.0;
    //     }
    // }
    // for (int run = 0; run < N_RUNS; ++ run){
    //     MazeDynaQ dyna_q(env_left,env_both, SHORTCUT_MAZE_CHANGE_STEP,SHORTCUT_MAZE_TOTAL_STEPS);
    //     dyna_q.learn( false, accumulated_array, run);
    //     total_reward += dyna_q.accumulated_reward;
    //     cout << "  Run " << run + 1 << ": Accumulated reward: " << dyna_q.accumulated_reward << endl;
    //     for (int i = 0; i < n_points; ++i) {
    //         sum_dq[i] += accumulated_array[run * n_points + i];
    //     }
    // }
    // cout << "Dyna-Q Test: Mean accumulated reward: " << total_reward / N_RUNS << endl;

    // // Test Dyna-Q+
    // total_reward = 0.0;
    // for (int i = 0; i < N_RUNS; ++i) {
    //     for (int j = 0; j < n_points; ++j) {
    //         accumulated_array[i*n_points + j] = 0.0;
    //     }
    // }
    // for (int run = 0; run < N_RUNS; ++ run){
    //     MazeDynaQ_plus dyna_q_plus(env_left,env_both, SHORTCUT_MAZE_CHANGE_STEP,SHORTCUT_MAZE_TOTAL_STEPS);
    //     dyna_q_plus.learn(false, accumulated_array, run);
    //     total_reward += dyna_q_plus.accumulated_reward;
    //     cout << "  Run " << run + 1 << ": Accumulated reward: " << dyna_q_plus.accumulated_reward << endl;
    //     // accumulate into sum vector
    //     for (int i = 0; i < n_points; ++i) {
    //         sum_dqp[i] += accumulated_array[run * n_points + i];
    //     }
    // }
    // cout << "Dyna-Q+ Test: Mean accumulated reward: " << total_reward / N_RUNS << endl;

    // // 计算均值（平均曲线）
    // vector<double> mean_dq(n_points, 0.0);
    // vector<double> mean_dqp(n_points, 0.0);
    // for (int i = 0; i < n_points; ++i) {
    //     mean_dq[i] = sum_dq[i] / static_cast<double>(N_RUNS);
    //     mean_dqp[i] = sum_dqp[i] / static_cast<double>(N_RUNS);
    // }

    // // 写出平均曲线 CSV（用于后续绘图）
    // write_csv_row("dyna_q_mean.csv", mean_dq);
    // write_csv_row("dyna_q_plus_mean.csv", mean_dqp);

    // cout << "Saved mean curves to dyna_q_mean.csv and dyna_q_plus_mean.csv" << endl;
    // cout << "-- Finished --" << endl << endl;    
    return 0;

}


