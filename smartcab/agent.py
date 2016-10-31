import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.Qtable = dict()
        self.init_Qtable()
        self.alpha = 0. # Q-Learning alpha initialization
        self.gamma = 0. # Q-Learning gamma initialization
        self.epsilon = 0.   # Q-Learning epsilon-greedy initialization
        self.old_state = None   # Q-Learning, save previous state 
        self.old_reward = None  # Q-Learning, save previous reward
        self.old_action = None  # Q-Learning, save previous action
        self.succ_trials = 0    # Count number of trials where destination is reached
        self.trials = 0         # Count total number of trials
        self.rew_tot = 0        # Sum total reward for a specific trial
        self.mistakes = 0       # Count number of mistakes (reward < 0)
        self.time_start = 0     # Save starting deadline

        self.register = dict()  # Save each trials data, to export to CSV
      

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.epsilon = 1./((self.trials//2)+1)

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.old_state = None
        self.old_reward = None
        self.old_action = None
        self.rew_tot = 0
        self.mistakes = 0
        self.time_start = self.env.get_deadline(self)
    

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        if self.trials == 0:    # records starting deadline
            self.time_start == deadline

        if deadline <= 0:       # detects if deadline is 0 or less, meaning trial has ended without success
            self.trials += 1
            self.register[self.trials] = {'succ': 0,'time': (self.time_start-deadline),'rew': self.rew_tot,
                                          'mistakes': self.mistakes, 'alpha': self.alpha, 'gamma': self.gamma}
            
        
        # if deadline is bigger than 12, taxicab has time to take optimal action
        # if deadline is 12 or less, taxicab may benefit to take some penalties,
        # in order to reach destination before deadline
        deadline_rush = 0   
        if deadline <= 12:                            
            deadline_rush = 1
            
        # TODO: Update state
        state = (inputs, self.next_waypoint, deadline_rush)
        self.state = self.encode_state(state) # encode state as a string, for easier recording in Q-table
        
        # TODO: Select action according to your policy
        actions = self.choose_action()
        if random.random() < self.epsilon:
            action = random.choice([None, 'forward', 'left', 'right'])
        else:
            action = random.choice(actions) #random.choice([None, 'forward', 'left', 'right'])

        # Execute action and get reward
        reward = self.env.act(self, action)

        if reward < 0:
            self.mistakes += 1
        self.rew_tot += reward
        

        if self.env.done == True:
            self.succ_trials += 1
            self.trials += 1
            self.register[self.trials] = {'succ': 1,'time': (self.time_start-deadline),'rew': self.rew_tot,
                                          'mistakes': self.mistakes, 'alpha': self.alpha, 'gamma': self.gamma}
            

        # TODO: Learn policy based on state, action, reward
        # Q-Learning algorithm
        if self.old_state != None:
            Qsa_old = self.Qtable[(self.old_state, self.old_action)]
            Qsa_current = self.Qtable[(self.state, action)]
            Qsa_old = (1-self.alpha) * Qsa_old + self.alpha * (self.old_reward + self.gamma*Qsa_current)
            self.Qtable[(self.old_state, self.old_action)] = Qsa_old

        # Update old_states as the current states for next iteration
        self.old_state = self.state
        self.old_reward = reward
        self.old_action = action
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


    def encode_state(self, state):
        """ Encodes state into a string with 6 chars

        Returns a string with following format: lortwd

        Args:
            state: a triple composed by input from environment, next waypoint and deadline simplified (0 or 1)

        Returns:
            A string with 6 chars that maps the current state as follows: "lortwd"            
                l - light value from inputs: {r = red, g = green}
                o - oncoming value from inputs: {n = none, r = right, l = left, f = forward}
                r - right value from inputs: {n = none, r = right, l = left, f = forward}
                t - left value from inputs: {n = none, r = right, l = left, f = forward}
                w - next waypoint value: {r = right, l = left, f = forward}
                d - deadline value: {0, 1}
            Example string: rnnrn0
        """
        state_str = str()

        # Obtains the first letter of each input/state, since these are unique
        for key, value in state[0].iteritems():
            state_str += str(value).lower()[0]
        state_str += state[1][0]
        state_str += str(state[2])

 
        return state_str 


    def init_Qtable(self):
        """ Initializes Q-table to 0, for all pairs (state, action)

        Goes through all possible states and actions and sets this (state,action)
        pair in the Q-table, with Q- value initialized as zero (0)
        
        Args:
            none

        Returns:
            none        
        """
        for light in ('r','g'):
            for oncoming in ('n','r','l','f'):
                for right in ('n','r','l','f'):
                    for left in ('n','r','l','f'):
                        for waypoint in ('r','l','f'):
                            for deadline in ('0','1'):
                                Qstate = str(light + oncoming + left + right +
                                             waypoint + deadline)
                                for Qaction in (None, 'right','left','forward'):
                                    self.Qtable[(Qstate, Qaction)] = 0.


    def choose_action(self):
        """ Chooses the best action from the Q-table

        Taking into account current state, searches for the action with highest Q-value
            
        Args:
            none
            
        Returns:
            A list of possible actions, to account when multiple actions
            have same highest Q-value.

        """
        actions = []
        reward = None

        for action in (None, 'right','left','forward'):
            if self.Qtable[(self.state, action)] > reward:
                actions = [action]
                reward = self.Qtable[(self.state, action)]
            elif self.Qtable[(self.state, action)] == reward:
                actions.append(action)

        return actions
    

def run():
    """Run the agent for a finite number of trials."""

    for alpha in [0.25, 0.5, 0.75]:
        for gamma in [0.25, 0.5, 0.75]:
            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Set Q-learning parameters
            a.alpha = alpha
            a.gamma = gamma

            # Now simulate it
            sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=100)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

            # Create dataframe with registered trials in order to dump
            # to CSV file, for analysis in a spreadsheet            
            df = pandas.DataFrame(a.register)         
            print df.transpose().to_csv(path_or_buf="results.csv", header=True, mode='a')

if __name__ == '__main__':
    run()
