 
# External imports
import torch
import numpy as np
import torch.nn.functional as F
import math

from torch.autograd import Variable

from preprocessing.parser_code_to_codeType import getBitmapVector
from src.neural_code2task_syn.task_synthesizer import obtain_karel_saturation_score_for_code
from src.karel_data_converters.converter_format_iclr18_to_karelgym import \
    iclr18_codejson_to_karelgym_codejson
from src.karel_emulator.code import Code

class Rolls(object):

    def __init__(self, action, proba, multiplicity, depth):
        self.successor = {}
        # The action that this node in the tree corresponds to
        self.action = action  # -> what was the sample taken here
        self.proba = proba  # -> Variable containing the log proba of
                            # taking this action
        self.multi_of_this = multiplicity  # -> How many times was this
                                              # prefix (until this point of
                                              # the sequence) seen
        self.depth = depth  # -> How far along are we in the sequence

        # Has no successor to this sample
        self.is_final = True

        # This is the reward that would be obtained by doing this prefix once.
        # This is only to use for bookkeeping.
        self.own_reward = 0
        # The one to use to compute gradients is the following.

        # This contains `self.own_reward * self.multi_of_this` + sum of all the
        # dependents self.dep_reward
        self.dep_reward = 0

    ## Stores the trajectories(program tokens) as well as successor to each trajectory token
    def expand_samples(self, trajectory, end_multiplicity, end_proba):
        '''
        The assumption here is that all but the last steps of the trajectory
        have already been created.
        '''
        assert(len(trajectory) > 0)

        pick = trajectory[0]
        if pick in self.successor:
         #   print('pick', pick)
            self.successor[pick].expand_samples(trajectory[1:],
                                                end_multiplicity,
                                                end_proba)
        else:
            # We add a successor so we are necessarily not final anymore
            self.is_final = False
            # We don't expand the samples by several steps at a time so verify
            # that we are done
            assert(len(trajectory) == 1)
            self.successor[pick] = Rolls(pick, end_proba,
                                         end_multiplicity,
                                         self.depth + 1)

        ##TODEBUG
        #print('self.successor', self.successor)

    def yield_var_and_grad(self):
        '''
        Yields 2-tuples:
        -> Proba: Variable correponding to the proba of this last choice
        -> Grad: Gradients for each of those variables
        '''
        for succ in self.successor.values():
            for var, grad in succ.yield_var_and_grad():
                yield var, grad
        yield self.proba, self.reinforce_gradient()

    def assign_rewards(self, reward_assigner, trace, entropy_weight):
        '''
        Using the `reward_assigner` scorer, go depth first to assign the
        reward at each timestep, and then collect back all the "depending
        rewards"
        '''
        if self.depth == -1:
            # This is the root from which all the samples come from, ignore
            pass
        else:
            # Assign to this step its own reward
            self.own_reward = reward_assigner.step_reward(trace,
                                                          self.is_final)
            self.own_reward = self.own_reward - (entropy_weight * math.log(self.proba.data))
            

        # Assign their own score to each of the successor
        for next_step, succ in self.successor.items():
            new_trace = trace + [next_step.item()]

            ##TODEBUG
            #print('new_trace',new_trace) 
            succ.assign_rewards(reward_assigner, new_trace, entropy_weight)

        # If this is a final node, there is no successor, so I can already
        # compute the dep-reward.
        if self.is_final:
            self.dep_reward = self.multi_of_this * self.own_reward
        else:
            # On the other hand, all my child nodes have already computed their
            # dep_reward so I can collect them to compute mine
            self.dep_reward = self.multi_of_this * self.own_reward
            for succ in self.successor.values():
                self.dep_reward += succ.dep_reward

    def reinforce_gradient(self):
        '''
        At each decision, compute a reinforce gradient estimate to the
        parameter of the probability that was sampled from.
        '''
        if self.depth == -1:
            return None
        else:
            # We haven't put in a baseline so just ignore this
            baselined_reward = self.dep_reward
            grad_value = baselined_reward / (1e-6 + self.proba.data)

            # We return a negative here because we want to maximize the rewards
            # And the pytorch optimizers try to minimize them, so this put them
            # in agreement
            return -grad_value



class Environment(object):

    def __init__(self, reward_norm, environment_data):
        '''
        reward_norm: float -> Value of the reward for correct answer
        environment_data: anything -> Data/Ground Truth to use for the reward evaluation
        To create different types of reward, subclass it and modify the
        `should_skip_reward` and `reward_value` function.
        '''
        self.reward_norm = reward_norm
        self.environment_data = environment_data

    def step_reward(self, trace, is_final):
        '''
        trace: List[int] -> all prediction of the sample to score.
        is_final: bool -> Is the sample finished.
        '''
        if self.should_skip_reward(trace, is_final):
            return 0
        else:
            return self.reward_value(trace, is_final)

    def should_skip_reward(self, trace, is_final):
        raise NotImplementedError

    def reward_value(self, trace, is_final):
        raise NotImplementedError

##State: Input and Output input_worlds
##Action: Program
##Reward: +1 or -1

class MultiIOGrid(Environment):
    '''
    This only gives rewards at the end of the prediction.
    +1 if the two programs lead to the same final state.
    -1 if the two programs lead to different outputs
    '''

    def __init__(self, reward_norm, tgt_bmpVector, simulator, vocab, num_tasks_iter, all_bmpVector):
        '''
        reward_norm: float
        input_grids, output_grids: Reference IO for the synthesis
        '''
        super(MultiIOGrid, self).__init__(reward_norm,
                                        (tgt_bmpVector, simulator, vocab, num_tasks_iter, all_bmpVector))
        self.tgt_bmpVector = tgt_bmpVector
        self.simulator = simulator
        self.vocab = vocab
        self.num_tasks_iter = num_tasks_iter
        self.all_bmpVector = all_bmpVector


    def should_skip_reward(self, trace, is_final):
        return (not is_final)

    def reward_value(self, trace, is_final):
        
        rew = 0
        
        parse_success, cand_prog, cand_prog_json = self.simulator.get_prog_ast(trace)
        if not parse_success:
            # Program is not syntactically correct
            rew = 0
        else:
            cand_prog_tkns = [self.vocab["idx2tkn"][tkn_idx] for tkn_idx in trace]
            cand_bmpVector, _, _ = getBitmapVector(cand_prog_tkns)
            if((cand_bmpVector in self.all_bmpVector) and (self.all_bmpVector.index(self.tgt_bmpVector) == self.all_bmpVector.index(cand_bmpVector))):
            
                    cand_prog_json_karelgym = iclr18_codejson_to_karelgym_codejson(cand_prog_json)
                    code = Code('karel', cand_prog_json_karelgym)
                    scores = []
                    score = obtain_karel_saturation_score_for_code(code, self.num_tasks_iter)
                    scores.append(score)
                    rew = np.mean(scores) * self.reward_norm
            else:
                rew = 0
                
        return rew

EnvironmentClasses = {
    "BlackBoxGeneralization": MultiIOGrid,
    "BlackBoxConsistency": MultiIOGrid,
}
