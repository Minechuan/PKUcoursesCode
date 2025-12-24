#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <utility>
#include <vector>
#include "blackjack.hpp"

class BlackjackPolicyBase{
    public:
        virtual int operator() (const Blackjack::State& state)=0;
};

class BlackjackPolicyDefault : public BlackjackPolicyBase{
    public:
        int operator() (const Blackjack::State& state){
            if (state.turn == Blackjack::PLAYER){
                return state.player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
            } else {
                return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
            }
        }
};

class BlackjackPolicyLearnableDefault : public BlackjackPolicyBase{
    static constexpr const char *ACTION_NAME = "HS";
    public:
        int operator() (const Blackjack::State& state){
            if (state.turn == Blackjack::DEALER){
                return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
            } else {
                return policy[state.dealer_shown][state.player_ace][state.player_sum];
            }
        }
        void update_policy(int dealer_shown, bool player_ace, int player_sum){
            // TODO
            // simply take argmax
            if (value[dealer_shown][player_ace][player_sum][0] >= value[dealer_shown][player_ace][player_sum][1])
                policy[dealer_shown][player_ace][player_sum] = 0;
            else 
                policy[dealer_shown][player_ace][player_sum] = 1;
        }
        // n iterations for each start (initial state and first action), no need to modify.
        void update_value(Blackjack& env, int n=10000){
            // TODO

            // REMEMBER to call set_value_initial() at the beginning
            // simulate from every possible initial state:(dealer_shown, player_ace, player_sum) \
                (call Blackjack::reset(,,) to do this) and player's every possible first action
            // BE AWARE only use player's steps (rather than dealer's) to update value estimation.
            set_value_initial(); 
            for (int it=0; it<n; it++){ // outer loop for iterations
                // cout << "\rIteration " << it+1 << endl;
                // sample an episode
                int first_visit_MC_count[11][2][22][2];
                memset(first_visit_MC_count, 0, sizeof(first_visit_MC_count));
                episode.clear();
                // randomly choose an initial state
                int dealer_shown = rand()%10 + 1; // [1~10]
                bool player_ace = rand()%2; // [0,1]
                int player_sum = rand()%10 + 11; // [11~20]
                // Judge if natural blackjack                
                if (player_sum == 21 && player_ace) { // No need to consider dealer's natural, because if dealer also has natural, it's a tie
                    // Natural blackjack
                    continue;
                }
                env.reset(dealer_shown, player_ace, player_sum);

                // Explore one episode: Use policy: pi
                Blackjack::State state = env.state();
                Blackjack::StepResult result;
                // choose first action : explore start
                int action;
                action = rand() % 2; // 0: HIT, 1: STICK
                episode.push_back(EpisodeStep(state, action));
                first_visit_MC_count[state.dealer_shown][state.player_ace][state.player_sum][action] ++;
                result = env.step(action);
                state = env.state();
                while (state.turn == Blackjack::PLAYER && !result.done){ // player's turn
                    int action = (*this)(state);
                    episode.push_back(EpisodeStep(state, action));
                    first_visit_MC_count[state.dealer_shown][state.player_ace][state.player_sum][action] ++;
                    result = env.step(action);
                    state = env.state();
                    if (result.done) break; // game over
                }
                if (!result.done){ // dealer's turn
                    // dealer's turn
                    bool done = result.done; // not done yet
                    while (not done){
                        int action = (*this)(state);
                        result = env.step(action);
                        state = env.state();
                        done = result.done;
                    }
                }
                int reward = result.player_reward; // final reward
                // from the end of the episode to the beginning, update value function
                for (int i=episode.size()-1; i>=0; i--){
                    EpisodeStep step = episode[i];
                    if (first_visit_MC_count[step.dealer_shown][step.player_ace][step.player_sum][step.action] > 1) {
                        // Exist more than one visit,
                        first_visit_MC_count[step.dealer_shown][step.player_ace][step.player_sum][step.action] --;
                        continue;
                    }
                    // Incremental Mean
                    state_action_count[step.dealer_shown][step.player_ace][step.player_sum][step.action] ++;
                    value[step.dealer_shown][step.player_ace][step.player_sum][step.action] += 
                        (reward - value[step.dealer_shown][step.player_ace][step.player_sum][step.action]) /
                        state_action_count[step.dealer_shown][step.player_ace][step.player_sum][step.action];
                    // (r_{k+1} - V_k)/(K+1)
                    first_visit_MC_count[step.dealer_shown][step.player_ace][step.player_sum][step.action] = 0; // no need to consider again
                    update_policy(step.dealer_shown, step.player_ace, step.player_sum); // update policy after each value update
                }


            }
        }
        void print_policy() const {
            cout << setw(10) << "Player Without Ace" << "\t\t" << "Player With Usable Ace." << endl;
            for (int player_sum = 21; player_sum >= 11; -- player_sum){
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << ACTION_NAME[policy[dealer_shown][0][player_sum]];
                }
                cout << "\t\t\t\t" ;
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << ACTION_NAME[policy[dealer_shown][1][player_sum]];
                }
                cout << endl;
            }
            cout << endl;
        }
        void print_value() const {
            cout << setw(40) << "Player Without Ace" << setw(20) <<"\t\t\t" << "Player With Usable Ace." << endl;
            for (int player_sum = 21; player_sum >= 11; -- player_sum){
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][0][player_sum][
                        policy[dealer_shown][0][player_sum]];
                }
                cout << "\t";
                for (int dealer_shown = 1; dealer_shown <= 10 ; ++ dealer_shown){
                    cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][1][player_sum][
                        policy[dealer_shown][1][player_sum]];
                }
                cout << endl;
            }
            cout << endl;
        }
        
        BlackjackPolicyLearnableDefault(){ // initialize policy, Constructor
            for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                for (int player_ace = 0; player_ace <= 1; ++ player_ace){
                    for (int player_sum = 0; player_sum <= 21; ++ player_sum){
                        int& action = policy[dealer_shown][player_ace][player_sum];
                        action = player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
                    }
                }
            }
        }
    private:
        // 11: dealer_shown (A-10); 
        // 2: player_usable_ace (true/false);
        // 22: player_sum (only need to consider 11-20, because HIT when sum<11 and STICK when sum=21 are always best action.)
        int policy[11][2][22];
        // 11:dealer_shown; 2:player_usable_ace; 22:player_sum; 2:action (0:HIT, 1:STICK)
        double value[11][2][22][2];
        int state_action_count[11][2][22][2];

        // record a episode sampled (only player's steps).
        struct EpisodeStep{
            // state: (dealer_shown, player_ace, player_sum)
            int dealer_shown;
            int player_ace;
            int player_sum;
            // the action taken at state
            int action;
            EpisodeStep(const Blackjack::State& state, int action){
                dealer_shown = state.dealer_shown;
                player_ace = int(state.player_ace);
                player_sum = state.player_sum;
                this->action = action;
            }
            EpisodeStep(){}
        };
        vector<EpisodeStep> episode; 

        void set_value_initial(){
            memset(value, 0, sizeof(value));
            memset(state_action_count, 0, sizeof(state_action_count));
        }
};

// Demonstrative play of player & dealer with default policy 
// int main(){
//     Blackjack env(true);
//     BlackjackPolicyDefault policy;
//     bool done;
//     while (true) {
//         done = false;    
//         env.reset();
//         while (not done){
//             Blackjack::State state = env.state();
//             int action = policy(state);
//             Blackjack::StepResult result = env.step(action);
//             done = result.done;
//         }
//         cout << endl;
//         this_thread::sleep_for(chrono::milliseconds(1000));
//     }
//     return 0;
// }

int main(){
    Blackjack env(false);
    BlackjackPolicyLearnableDefault policy;// initialize policy

    policy.update_value(env, 10000000);
    policy.print_policy();
    policy.print_value();
    return 0;
}