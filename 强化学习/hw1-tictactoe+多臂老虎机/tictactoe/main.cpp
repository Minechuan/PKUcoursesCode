#include <ctime>
#include <iostream>
#include <vector>
#include <stack>
#include "tictactoe.hpp"
#include <cstdlib>
#include <memory>
#include <random>
// Use epsilon-greedy policy


using namespace std;

/*###################################################
Used variables:
*/



std::mt19937 gen(std::random_device{}()); // 随机种子引擎
std::uniform_real_distribution<> dis(0.0, 1.0);
double epsilon = 0.1;
double lr = 0.1;

const int MAX_STATE = 1 << 18; // 2^18 = 262144, used to encode the state
vector<double> value_fn(MAX_STATE, 0.0);

// parameters
double win_reward = 1.0;
double lose_reward = -1.0;
double tie_reward = 0.0;

int outer_iteration = 5; // number of outer iterations


double predict_Q(const TicTacToe::State& state, const TicTacToe::Action& action){
    TicTacToe::State next = state; // deep copy
    next.put(action);
    return value_fn[next.board];
}

int random_action_id(int n_action) {
    std::uniform_int_distribution<> dis(0, n_action - 1); 
    return dis(gen);
}


//##################################################


class TicTacToePolicyBase{
    public:
        virtual TicTacToe::Action operator()(const TicTacToe::State& state) const = 0;
};


// randomly select a valid action for the step.
class TicTacToePolicyRandom : public TicTacToePolicyBase{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            int n_action = actions.size();
            int action_id = random_action_id(n_action);
            // state.print();
            if (state.turn == TicTacToe::PLAYER_X){
                return actions[action_id];
            } else {
                return actions[action_id];
            }
        }
        TicTacToePolicyRandom(){
            srand(time(nullptr));
        }
};

class TicTacToePolicyEpsGreedy : public TicTacToePolicyBase{
    // Used for training
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            if (state.turn == TicTacToe::PLAYER_X){
                // epsilon-greedy
                double p = dis(gen);
                // std::cout << "Epsilon: " << epsilon << " " << p << std::endl;
                if (p < epsilon){
                    // random action
                    int n_action = actions.size();
                    int action_id = rand() % n_action;
                    return actions[action_id];
                }
                else {
                    // greedy action
                    TicTacToe::Action best_action = actions[0];
                    double best_q_s_a = predict_Q(state, best_action);
                    for (const auto& action : actions){
                        double q_s_a = predict_Q(state, action);
                        if (q_s_a > best_q_s_a){
                            best_action = action;
                        }
                    }
                    return best_action;
                }
            } else { // enemy
                return actions[0];
            }
        }
        TicTacToePolicyEpsGreedy(){}
};



// select the first valid action.
class TicTacToePolicyDefault : public TicTacToePolicyBase{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            if (state.turn == TicTacToe::PLAYER_X){
                // TODO
                // search value table, and return the action with the highest value
                double best_value = -1e9;
                TicTacToe::Action best_action = actions[0];
                for (const auto& action : actions){
                    double q_s_a = predict_Q(state, action);
                    if (q_s_a > best_value){
                        best_value = q_s_a;
                        best_action = action;
                    }
                }
                return best_action;
            } else {
                return actions[0];
            }
        }
        TicTacToePolicyDefault(){}
};


void train(){

    int successive_win = 0;
    for (int it = 0; it < outer_iteration; ++ it){
        bool done = false;
        TicTacToe env(false);
        env.reset(); // reset the environment
        TicTacToePolicyEpsGreedy policy;
        stack<int> state_history;
        // play one game
        int cnt = 0;
        while (not done){
            // ++ cnt;
            TicTacToe::State state = env.get_state();
            if (state.turn == TicTacToe::PLAYER_X){// X's state
                state_history.push(state.board);
            }
            TicTacToe::Action action = policy(state);
            env.step(action);
            done = env.done();
        }
        // std::cout << "Game over after " << cnt << " steps." << std::endl;
        int winner = env.winner();
        if (winner == TicTacToe::PLAYER_X){
            int last_state = state_history.top();
            value_fn[last_state] = win_reward; // win state
            state_history.pop();
            while (not state_history.empty()){
                int state_code = state_history.top();
                state_history.pop();
                value_fn[state_code] += lr * (value_fn[last_state] - value_fn[state_code]);
            }
            successive_win += 1;

        } else if (winner == TicTacToe::PLAYER_O){
            // lose
            int last_state = state_history.top();
            value_fn[last_state] = lose_reward; // lose state
            state_history.pop();
            while (not state_history.empty()){
                int state_code = state_history.top();
                state_history.pop();
                value_fn[state_code] += lr * (value_fn[last_state] - value_fn[state_code]);
            }
        } else {
            // tie
            int last_state = state_history.top();
            value_fn[last_state] = tie_reward; // tie state
            state_history.pop();
            while (not state_history.empty()){
                int state_code = state_history.top();
                state_history.pop();
                value_fn[state_code] += lr * (value_fn[last_state] - value_fn[state_code]);
            }
        }
        if (it % 10 == 0)
            std::cout << "Episode " << it << ": " << successive_win << " successive wins, win_rate:" << (successive_win / (it + 1.0)) << std::endl;
    }
}


void test(string policy_name){
    int successive_win = 0;
    std::unique_ptr<TicTacToePolicyBase> policy;
    TicTacToe env(false);
    if(policy_name == "Random"){
        policy = std::make_unique<TicTacToePolicyRandom>();
    } else if (policy_name == "Trained"){
        policy = std::make_unique<TicTacToePolicyDefault>();
    } else {
        std::cout << "Unknown policy name: " << policy_name << std::endl;
        return;
    }
    for (int i = 0; i < 100; ++i) {
        bool done = false;
        // set verbose true 
        env.reset(); // reset the environment
        // play one game
        while (not done){
            TicTacToe::State state = env.get_state();
            TicTacToe::Action action = (*policy)(state);
            env.step(action);
            done = env.done();
            // env.step_back();
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        int winner = env.winner();
        if (winner == TicTacToe::PLAYER_X){
            successive_win += 1;
        }
    }
    std::cout << "Test over: " << successive_win << " wins in 100 games." << std::endl;
}




#include <chrono>
#include <thread>

// randomly select action
int main(){

    // Train the model
    train();
    test("Random");
    test("Trained");
    TicTacToe env(true);
    env.reset(); // reset the environment
    TicTacToePolicyDefault policy;
    bool done = false;
    // set verbose true 
    env.reset(); // reset the environment
    // play one game
    while (not done){
        TicTacToe::State state = env.get_state();
        TicTacToe::Action action = policy(state);
        env.step(action);
        done = env.done();
        // env.step_back();
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    int winner = env.winner();
    return 0;
};