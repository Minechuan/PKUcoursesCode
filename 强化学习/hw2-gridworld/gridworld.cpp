#include <utility>
#include <cstdlib>
#include <iostream>
#include <cmath>


#define SIZE 5
#define ACTION_NUM 4

using namespace std;
class GridWorld{
    public:
        static const int 
            NORTH=0, SOUTH=1, EAST=2, WEST=3;
        static const char ACTION_NAME[][16];
        typedef pair<int, int> State;
        bool verbose;
        State state(){
            return make_pair(x, y);
        }
        void set_state(int x, int y){
            this->x = x;
            this->y = y;
            if (verbose){
                cout << "State reset: (" << x << "," << y << ")" << endl;
            }
        }
        void reset(){
            set_state(0, 0);
        }
        pair<State, double> step(int action){
            State old_state = state();
            double reward = state_transition(action);
            if (verbose){
                cout << "State: (" << old_state.first << "," << old_state.second << ")" << endl;
                cout << "Action: " << ACTION_NAME[action] << endl;
                cout << "Reward: " << reward << endl;
                cout << "New State: (" << x << "," << y << ")" << endl << endl;
            }
            return make_pair(state(), reward);
        }
        int sample_action(){
            return rand() % 4;
        }
        GridWorld(int x=0, int y=0, bool verbose=false){
            set_state(x, y);
            this->verbose = verbose;
        }
        
    private:
        int x, y;
        double state_transition(int action){
            if (state() == make_pair(1, 0)){
                x = 1;
                y = 4;
                return 10.0;
            }
            if (state() == make_pair(3, 0)){
                x = 3;
                y = 2;
                return 5.0;
            }
            if (action == NORTH and y == 0 or
                action == SOUTH and y == 4 or
                action == EAST and x == 4 or
                action == WEST and x == 0){
                return -1.0; 
            }
            switch (action){
                case NORTH:
                    y --; break;
                case SOUTH:
                    y ++; break;
                case EAST:
                    x ++; break;
                case WEST:
                    x --; break;
            }
            return 0.0;
        }
};
const char GridWorld::ACTION_NAME[][16] = {"NORTH(0,-1)", "SOUTH(0,1)", "EAST:(1,0)", "WEST:(-1,0)"};



void print_values(double V[SIZE][SIZE]) {
    // round to 2 decimal places
    cout << fixed;
    cout.precision(2);
    for (int y = 0; y < SIZE; y++) {
        for (int x = 0; x < SIZE; x++) {
            cout << V[x][y] << "\t";
        }
        cout << endl;
    }
}



void random_policy(int iterations=10000, double gamma=0.9, double threshold=1e-4){
    GridWorld env = GridWorld(0, 0, false);

    double value_table[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    double old_value_table[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };
    double policy_prob[4];
    
    int it = 0;
    for (; it<iterations; it++){
        env.reset();
        double delta = 0; // Check convergence
        for(int i=0;i<SIZE;i++){
            for(int j=0;j<SIZE;j++){
                double v = 0;
                
                for(int a=0;a<ACTION_NUM;a++){
                    env.set_state(i, j);
                    policy_prob[a] = 0.25; // equal probability for each action
                    auto state_reward = env.step(a);
                    // cout << "From (" << i << "," << j << ") taking action " << a << " to (" << state_reward.first.first << "," << state_reward.first.second << ") with reward " << state_reward.second << endl;
                    v += policy_prob[a] * (state_reward.second + gamma * old_value_table[state_reward.first.first][state_reward.first.second]);
                }
                // cout << "Value at (" << i << "," << j << "): " << v << endl;
                delta = max(delta, abs(v - old_value_table[i][j]));
                value_table[i][j] = v;
            }
        }
        if (delta < threshold){
            cout << "Converged at iteration " << it << endl;
            break;
        }
        // Update old_value_table
        for(int i=0;i<SIZE;i++){
            for(int j=0;j<SIZE;j++){
                old_value_table[i][j] = value_table[i][j];
            }
        }
    }
    cout << "----------------------------------------" << endl;
    cout << "Value function of a random policy:" << endl;
    cout << "Value Table after " << it << " iterations:" << endl;
    print_values(value_table);
}



void optimal_policy(int iterations=10000, double gamma=0.9, double threshold=1e-4){
    GridWorld env = GridWorld(0, 0, false);

    double value_table[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    double old_value_table[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };
    
    int it = 0;
    for (; it<iterations; it++){
        env.reset();
        double delta = 0; // Check convergence
        for(int i=0;i<SIZE;i++){
            for(int j=0;j<SIZE;j++){
                double v = old_value_table[i][j];
                
                for(int a=0;a<ACTION_NUM;a++){
                    env.set_state(i, j);
                    auto state_reward = env.step(a);
                    // cout << "From (" << i << "," << j << ") taking action " << a << " to (" << state_reward.first.first << "," << state_reward.first.second << ") with reward " << state_reward.second << endl;
                    v = max(v, state_reward.second + gamma * old_value_table[state_reward.first.first][state_reward.first.second]);
                }
                // cout << "Value at (" << i << "," << j << "): " << v << endl;
                delta = max(delta, abs(v - old_value_table[i][j]));
                value_table[i][j] = v;
            }
        }
        if (delta < threshold){
            cout << "Converged at iteration " << it << endl;
            break;
        }
        // Update old_value_table
        for(int i=0;i<SIZE;i++){
            for(int j=0;j<SIZE;j++){
                old_value_table[i][j] = value_table[i][j];
            }
        }
    }
    cout << "----------------------------------------" << endl;
    cout << "Value function of a optimal policy:" << endl;
    cout << "Value Table after " << it << " iterations:" << endl;
    print_values(value_table);
}


#include <chrono>
#include <thread>
// int main(){
//     GridWorld env = GridWorld(0, 0, true);
//     int update_counter = 0;
//     while (true){
//         update_counter ++;
//         int action = env.sample_action();
//         auto state_reward = env.step(action);
//         this_thread::sleep_for(chrono::milliseconds(1000));
//     }
//     return 0;
// }

int main(){
    random_policy(10000, 0.9, 1e-4);
    this_thread::sleep_for(chrono::milliseconds(1000));
    cout << endl << endl;
    optimal_policy(10000, 0.9, 1e-4);
    return 0;
}