import os
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from ObjectFactory import ObjectFactory
from Utilities import Utilities
from Visualizer import Visualizer


def agent_reached_goal(environment, goal_index):
    target_goal_layer = 0 if goal_index == environment.object_type_num else goal_index.item() + 1
    same_elements_on_grid_num = torch.logical_and(environment.env_map[0, 0, :, :],
                                                  environment.env_map[0, target_goal_layer, :, :]).sum()
    if same_elements_on_grid_num > 0:
        return True
    return False


def update_pre_located_objects(object_locations, agent_location, goal_reached):
    pre_located_objects = []

    if goal_reached:
        for obj_type in object_locations:
            temp = []
            for loc in obj_type:
                if any(~torch.eq(loc, agent_location[0])):
                    temp.append(loc.tolist())
                else:
                    temp.append([-1, -1])
            pre_located_objects.append(temp)
    return torch.tensor(pre_located_objects)


def create_tensors(params):
    environments = torch.zeros((params.EPISODE_NUM,
                                params.STEPS_NUM,
                                params.OBJECT_TYPE_NUM + 1,
                                params.HEIGHT,
                                params.WIDTH), dtype=torch.float32)
    needs = torch.zeros((params.EPISODE_NUM,
                         params.STEPS_NUM,
                         params.OBJECT_TYPE_NUM), dtype=torch.float32)
    actions = torch.zeros((params.EPISODE_NUM,
                           params.STEPS_NUM), dtype=torch.int32)
    selected_goals = torch.zeros((params.EPISODE_NUM,
                                  params.STEPS_NUM), dtype=torch.int32)
    goal_reached = torch.zeros((params.EPISODE_NUM,
                                params.STEPS_NUM), dtype=torch.bool)

    return environments, needs, actions, selected_goals, goal_reached


def get_data_on_episode(episode, rand_indices, goal_selection_time, orig_needs):
    for step in range(goal_selection_time.shape[0]):
        rand_step = rand_indices[step]
        rand_need = orig_needs[episode, rand_step, :].unsqueeze(0).tolist()
        yield rand_need


def generate_action():
    original_data_dir = './Original Data'
    res_dir = './Random Data'
    if not os.path.exists(res_dir):
        os.path.exists(res_dir)

    utility = Utilities()
    params = utility.get_params()
    factory = ObjectFactory(utility)
    res_folder = utility.make_res_folder()

    original_environments = torch.load(os.path.join(original_data_dir, 'environments.pt'))
    original_needs = torch.load(os.path.join(original_data_dir, 'needs.pt'))
    original_goal_reached = torch.load(os.path.join(original_data_dir, 'goal_reached.pt'))
    meta_controller = factory.get_meta_controller()
    controller = factory.get_controller()

    # episode_begin = True
    # environment = None
    print_threshold = 3
    visualizer = Visualizer(utility)
    environments, needs, actions, selected_goals, goal_reached = create_tensors(params)
    for episode in range(params.EPISODE_NUM):
        batch_environments_ll = []
        batch_actions_ll = []
        batch_needs_ll = []
        batch_selected_goals_ll = []

        goal_selection_time = torch.argwhere(original_goal_reached[episode, :]) + 1
        goal_selection_time = torch.cat([torch.zeros(1, 1, dtype=torch.int32), goal_selection_time])

        back_up_indices = torch.randperm(goal_selection_time.shape[0])[:math.ceil(goal_selection_time.shape[0]*.1)]
        rand_indices = torch.cat([torch.randperm(goal_selection_time.shape[0]), back_up_indices])

        environment_objects = torch.argwhere(original_environments[episode, 0, 1:, :, :])
        each_type_object_num = [torch.eq(environment_objects[:, 0], e).sum().item() for e in
                                environment_objects[:, 0].unique()]
        pre_located_objects = -1 * torch.ones(len(each_type_object_num), max(each_type_object_num), 2,
                                              dtype=torch.int32)
        last_ind = [0] * len(each_type_object_num)
        for obj in environment_objects:
            obj_type = obj[0]
            pre_located_objects[obj_type, last_ind[obj_type], :] = obj[1:]
            last_ind[obj_type] += 1
        agent_location = torch.argwhere(original_environments[episode, 0, 0, :, :])
        agent_needs = original_needs[episode, rand_indices[0], :].unsqueeze(0)
        agent = factory.get_agent(pre_location=agent_location.tolist(),
                                  preassigned_needs=agent_needs.tolist())
        environment = factory.get_environment(None,
                                              None,
                                              each_type_object_num,
                                              pre_located_objects,
                                              random_new_object_type=None,
                                              random_new_object=False)

        n_step = 0
        n_goal = 0
        agent_location = agent.location
        for preassigned_needs in get_data_on_episode(episode=episode,
                                                     rand_indices=rand_indices,
                                                     goal_selection_time=goal_selection_time,
                                                     orig_needs=original_needs):
            agent = factory.get_agent(pre_location=agent_location.tolist(),
                                      preassigned_needs=preassigned_needs)

            goal_map, goal_type = meta_controller.get_goal_map(environment,
                                                               agent,
                                                               controller=controller)  # goal type is either 0 or 1
            n_goal += 1
            while True:
                batch_environments_ll.append(environment.env_map.clone())
                batch_needs_ll.append(agent.need.clone())
                batch_selected_goals_ll.append(goal_type.cpu().clone())

                if episode < print_threshold:
                    fig, ax = visualizer.map_to_image(agent, environment)
                    fig.savefig('{0}/episode_{1}_goal_{2}_step_{3}.png'.format(res_folder, episode, n_goal, n_step))
                    plt.close()

                agent_goal_map = torch.stack([environment.env_map[:, 0, :, :], goal_map], dim=1)
                action_id = controller.get_action(agent_goal_map).clone()
                agent.take_action(environment, action_id)

                step_goal_reached = agent_reached_goal(environment, goal_type)
                goal_reached[episode, n_step] = step_goal_reached

                batch_actions_ll.append(action_id.clone())
                # all_actions += 1
                n_step += 1

                if step_goal_reached or n_step == params.STEPS_NUM:
                    if step_goal_reached:
                        pre_located_objects_location = update_pre_located_objects(environment.object_locations,
                                                                                  agent.location,
                                                                                  step_goal_reached)
                        pre_located_objects_num = environment.each_type_object_num
                        # pre_located_agent = agent.location.tolist()
                        # pre_assigned_needs = agent.need.tolist()

                        environment = factory.get_environment(few_many=None,
                                                              probability_map=None,
                                                              pre_located_objects_num=pre_located_objects_num,
                                                              pre_located_objects_location=pre_located_objects_location,
                                                              random_new_object_type=None,
                                                              random_new_object=False)
                    break

            if n_step == params.STEPS_NUM:
                break

        environments[episode, :, :, :, :] = torch.cat(batch_environments_ll, dim=0)
        needs[episode, :, :] = torch.cat(batch_needs_ll, dim=0)
        selected_goals[episode, :] = torch.cat(batch_selected_goals_ll, dim=0)
        actions[episode, :] = torch.cat(batch_actions_ll, dim=0)

        if episode % 100 == 0:
            print(episode)

    # Saving to memory
    if not os.path.exists('./Data_{0}'.format(params.AGENT_TYPE)):
        os.mkdir('./Data_{0}'.format(params.AGENT_TYPE))

    torch.save(environments, './Data_{0}/environments.pt'.format(params.AGENT_TYPE))
    torch.save(needs, './Data_{0}/needs.pt'.format(params.AGENT_TYPE))
    torch.save(selected_goals, './Data_{0}/selected_goals.pt'.format(params.AGENT_TYPE))
    torch.save(goal_reached, './Data_{0}/goal_reached.pt'.format(params.AGENT_TYPE))
    torch.save(actions, './Data_{0}/actions.pt'.format(params.AGENT_TYPE))
