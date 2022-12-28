import torch
from torch.autograd import Variable

from src.neural_task2code_syn.utils.actions import ACTION_MAP
from src.neural_task2code_syn.karelgym.karel_gym import KarelGym


# Teacher class to train RL agents
class KarelGymTeacher(KarelGym):
    def __init__(self, nb_rollouts, data_path, device=None):
        self.device = torch.device(
            device if torch.cuda.is_available() and 'cuda' in device else "cpu")
        super(KarelGymTeacher, self).__init__(data_path=data_path)
        self.nb_rollouts = nb_rollouts

    # Grids are represented as tensors
    def _get_grid_representation(self, task_id=None):
        if task_id is None:
            task_id = self.KState_id
        sample_inp_grids = []
        sample_post_grids = []
        for pregrid, postgrid in zip(self.TASKS[task_id].TRAIN_TASK.pregrids,
                                     self.TASKS[task_id].TRAIN_TASK.postgrids):
            inp_grid = pregrid.toPytorchTensor(padding=self.max_pad)
            post_grid = postgrid.toPytorchTensor(padding=self.max_pad)

            # Do the inp_grid
            sample_inp_grids.append(inp_grid)
            sample_post_grids.append(post_grid)

        sample_inp_grids = Variable(torch.stack(sample_inp_grids, 0))
        sample_post_grids = Variable(torch.stack(sample_post_grids, 0))

        # Duplicate the tensor of the grid representation over the first dimension
        # For examples with one I/O pair
        if sample_inp_grids.size(dim=0) < 2:
            sample_inp_grids = sample_inp_grids.expand(2, -1, -1, -1)
            sample_post_grids = sample_post_grids.expand(2, -1, -1, -1)

        return sample_inp_grids, sample_post_grids

    def reset_task(self, task_id):
        # Reset selected task to initial state
        self.steps_taken = 0
        self.blocks_used = set()
        self.done = False
        self.true_success = False
        # Choose a new task and update KState variable
        self.KState_id = task_id
        self.KState = self.TASKS[self.KState_id]
        self.KState.reset()
        return {"in_grids": self.KState.TRAIN_TASK.pregrids,
                "out_grids": self.KState.TRAIN_TASK.postgrids,
                "code": self.KState.current_code.getJson()}

    def multistep(self, action_trace, task_id=None):

        info = {"semantic_success": False, "true_success": False}
        if task_id is None:
            task_id = self.KState_id

        self.reset_task(task_id)

        if not isinstance(action_trace, list):
            action_trace = [action_trace]
        # Transform list of tensor actions to list of actions if it is not empty.
        if torch.is_tensor(action_trace[0]) and len(action_trace) > 1:
            action_trace = [action.item() for action in action_trace if
                            action.item() < len(ACTION_MAP)]
        else:  # When trace is a tensor with multiple actions
            action_trace = action_trace[0].tolist()
            if not isinstance(action_trace, list):
                action_trace = [action_trace]
            action_trace = [action for action in action_trace if
                            action < len(ACTION_MAP)]

        if action_trace:  # action_trace can be empty

            self.steps_taken += len(action_trace)
            self.blocks_used.update(action_trace)
            cur_code, cur_world, cur_execution_info, body_left_empty, size_constraint_met, type_constraint_met, \
            self.done, self.true_success, semantic_success, solved = \
                self.KState.step(ls_actions=action_trace)

            info = {"cur_code": cur_code, "cur_world": cur_world,
                    "cur_exec_info": cur_execution_info,
                    "body_left_empty": body_left_empty,
                    "size_constraint": size_constraint_met,
                    "type_constraint": type_constraint_met,
                    "code_done": self.done, "true_success": self.true_success,
                    "semantic_success": semantic_success, "solved": solved}

        rew = self.get_reward()

        if self.steps_taken > self.H:  # Horizon condition
            self.done = True

        return {"in_grids": self.TASKS[task_id].TRAIN_TASK.pregrids,
                "out_grids": self.TASKS[task_id].TRAIN_TASK.postgrids,
                "code": self.TASKS[
                    task_id].current_code.getJson()}, rew, self.done, info

    def batch_reset(self, batch_ids=None):
        batch = []
        for task_id in batch_ids:
            pregrids, postgrids = self._get_grid_representation(task_id)
            batch.append({"id": task_id, "in_grids": pregrids, "out_grids": postgrids})

        return batch

    # Given a batch of ids returns the target codes as dictionaries
    def get_expert_trajectories(self, batch_of_ids):
        return [self.TASKS[i].TARGET_CODE for i in batch_of_ids]

    def get_reward(self):
        return super(KarelGymTeacher, self)._get_reward() / self.nb_rollouts

    def _save_all_grids(self):
        all_grids = []
        for i in range(len(self.TASKS)):
            self.KState_id = i
            grid = self._get_grid_representation()
            all_grids.append(grid)
        torch.save(all_grids, "IOGrids.pt")
