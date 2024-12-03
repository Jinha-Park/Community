import math
import traceback
from collections import defaultdict
from ortools.linear_solver import pywraplp
import heapq


ENERGY_THRESH = 3
NUM_TASK_OPTIONS = 5
NUM_PAIR_OPTIONS = 1

def get_energy_cost(player_abilities, task):
    return sum(max(task[i]- player_abilities[i],0) for i in range(len(task)))

def get_zero_energy_partners(player, community):
    preferences  = []
    for id, task in enumerate(community.tasks):
        ind_cost = get_energy_cost(player.abilities, task)
        if ind_cost == 0:
            continue
        for partner in community.members:
            if partner.id == player.id or partner.incapacitated:
                continue  # Skip self, incapacitated
            if partner.id == player.id:
                continue

            partner_ind_cost = get_energy_cost(partner.abilities, task)
            #Don't partner if they can do it alone
            if partner_ind_cost == 0:
                continue

            partnerhip_cost = get_energy_cost([max(player.abilities[i], partner.abilities[i]) for i in range(len(player.abilities))], task)
            if partnerhip_cost == 0:
                preferences.append([id, partner.id])
    return preferences

def get_k_best_for_task(community, secondary_energy_limit, k=5):
    best = {}
    
    for taskid, task in enumerate(community.tasks):
        heap = []

        for playerid in range(len(community.members)):
            player = community.members[playerid]
            cost = get_energy_cost(player.abilities, task)
            remaining_energy = player.energy - cost

            if remaining_energy > secondary_energy_limit:
                heapq.heappush(heap, (remaining_energy, f"{playerid}"))

            for partnerid in range(playerid + 1, len(community.members)):
                partner = community.members[partnerid]
                partnerhip_cost = get_energy_cost(
                    [max(player.abilities[i], partner.abilities[i]) for i in range(len(player.abilities))],
                    task
                )
                
                remaining_energy_1 = player.energy - (partnerhip_cost / 2)
                remaining_energy_2 = partner.energy - (partnerhip_cost / 2)

                if remaining_energy_1 > secondary_energy_limit and remaining_energy_2 > secondary_energy_limit:
                    heapq.heappush(heap, (min(remaining_energy_1, remaining_energy_2), f"{playerid}-{partnerid}"))

            if len(heap) > k:
                heapq.heappop(heap)  # Pop the smallest item to maintain size k

        best[f"T_{taskid}"] = heap
    return best

def get_tasks_that_need_sacrifices(community):
    sacrifices = []
    for taskid, task in enumerate(community.tasks):
        sacrfice = True
        for playerid in range(len(community.members)):
            player = community.members[playerid]
            cost = get_energy_cost(player.abilities, task)
            if cost < 10:
                sacrfice = False
            for partnerid in range(playerid + 1, len(community.members)):
                partner = community.members[partnerid]
                partnerhip_cost = get_energy_cost(
                    [max(player.abilities[i], partner.abilities[i]) for i in range(len(player.abilities))],
                    task
                )
                if partnerhip_cost < 20:
                    sacrfice = False
        if sacrfice:
            sacrifices.append(taskid)
    return sacrifices



def solve_assignment(best, alpha=1, beta=0.5):

    solver = pywraplp.Solver.CreateSolver('SCIP')
    min_energy = min(energy for t, assignments in best.items() for energy, _ in assignments)
    shift = abs(min_energy) + 1 if min_energy < 0 else 0

    # Update energies in the input data
    shifted_best = {
        t: [(energy + shift, player_set) for energy, player_set in assignments]
        for t, assignments in best.items()
    }

    best = shifted_best
    
    # Variables
    x = {}
    for t, assignments in best.items():
        for energy, player_set in assignments:
            x[(t, player_set)] = solver.BoolVar(name=f'x_{t}_{player_set}')

    # Objective: Weighted sum of tasks completed and energy left
    objective = solver.Objective()

    # Add terms for the number of tasks completed
    for (t, player_set), var in x.items():
        objective.SetCoefficient(var, alpha)

    # Add terms for the total energy left
    for (t, player_set), var in x.items():
        energy = next(e for e, p in best[t] if p == player_set)
        objective.SetCoefficient(var, beta * energy)

    objective.SetMaximization()

    # Constraints
    # Each task can only have one assignment
    for t in best.keys():
        solver.Add(sum(x[(t, player_set)] for energy, player_set in best[t]) <= 1)

    # Each player can only participate in one task
    players = {str(p) for t in best for _, p_set in best[t] for p in p_set.split('-')}
    for player in players:
        solver.Add(
            sum(
                x[(t, player_set)]
                for t, assignments in best.items()
                for energy, player_set in assignments
                if player in player_set.split('-')
            ) <= 1
        )

    # Solve
    status = solver.Solve()
    assignments = {}
    # Results
    if status == pywraplp.Solver.OPTIMAL:
        print('Optimal solution found!')
        for (t, player_set), var in x.items():
            if var.solution_value() > 0:
                assignments[player_set] = t
                print(f'Task {t} assigned to {player_set}')
        print('Weighted objective value:', objective.Value())
        return assignments
    else:
        print('No optimal solution found.')
        return None


def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player.'''
    try:
        preferences = []

        task_avg_difficulty = sum(sum(task) for task in community.tasks) / len(community.tasks)
        player_avg_skill = sum(sum(member.abilities) for member in community.members) / len(community.members)

        primary_energy_limit = 0  # Allow energy to drop but not too far into the negatives

        # Calculate the ratio of difficulty to skill
        difficulty_ratio = task_avg_difficulty / max(player_avg_skill, 1e-6)  # Avoid division by zero

        # Map difficulty ratio to the range [-10, primary_energy_limit]
        if difficulty_ratio <= 1:
            secondary_energy_limit = primary_energy_limit  # Tasks are manageable
        else:
            # The higher the difficulty_ratio, the closer the limit gets to -10
            # secondary_energy_limit = primary_energy_limit - (difficulty_ratio - 1) * 10
            secondary_energy_limit = primary_energy_limit - (10 * (1 - math.exp(-(difficulty_ratio - 1))))
            secondary_energy_limit = max(secondary_energy_limit, -10)  # Ensure it doesn't drop below -10

        # Rest if negative energy
        if player.energy <= primary_energy_limit:
            return []
        
        # Only do tasks that cost 0 energy or rest if low energy
        if player.energy < 3:
            return get_zero_energy_partners(player, community)
        
        sacrifices = get_tasks_that_need_sacrifices(community)

        best_for_task= get_k_best_for_task(community, secondary_energy_limit)
        print(best_for_task)

        # All tasks left need sacrifices!
        if len(sacrifices) == len(community.tasks):
            #TODO: Handle sacrifices
            pass
        
        assignment = solve_assignment(best_for_task)
        if assignment == None:
            #TODO: Why even will this be none. But maybe just return the top k choices for this player then
            print("Bruh why?", best_for_task, sacrifice)

        for p in assignment:
            taskid = int(assignment[p].split('_')[1])
            if '-' in p:
                p1,p2 = p.split('-')
                if int(p1)== player.id:
                    preferences.append([taskid, int(p2)])
                elif int(p2) == player.id:
                    preferences.append([taskid, int(p1)])
        return preferences
    
    except Exception as e:
        print(e)
        traceback.print_exc()


def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually.'''
    preferences = []

    task_avg_difficulty = sum(sum(task) for task in community.tasks) / max(len(community.tasks), 1e-6) 
    player_avg_skill = sum(sum(member.abilities) for member in community.members) / max(len(community.members), 1e-6)

    primary_energy_limit = 0  # Allow energy to drop but not too far into the negatives

    # Calculate the ratio of difficulty to skill
    difficulty_ratio = task_avg_difficulty / max(player_avg_skill, 1e-6)  # Avoid division by zero

    # Map difficulty ratio to the range [-10, primary_energy_limit]
    if difficulty_ratio <= 1:
        secondary_energy_limit = primary_energy_limit  # Tasks are manageable
    else:
        # The higher the difficulty_ratio, the closer the limit gets to -10
        # secondary_energy_limit = primary_energy_limit - (difficulty_ratio - 1) * 10
        secondary_energy_limit = primary_energy_limit - (10 * (1 - math.exp(-(difficulty_ratio - 1))))
        secondary_energy_limit = max(secondary_energy_limit, -10)  # Ensure it doesn't drop below -10
    
    #TODO: Take the assignments and use it!!

    # Evaluate tasks for individual completion
    for task_id, task in enumerate(community.tasks):
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        remaining_energy = player.energy - energy_cost

        # Consider tasks that leave the player with energy above the secondary limit
        if remaining_energy > secondary_energy_limit:
            preferences.append((task_id, energy_cost, remaining_energy))

    # Sort tasks by a combination of low energy cost and high remaining energy
    preferences.sort(key=lambda x: (x[1], -x[2]))  # Sort by energy cost, then remaining energy

    # Return task IDs in preferred order
    return [task_id for task_id, _, _ in preferences]

'''
Prevent overlap in volunteering
Avoid volunteering to partner on easy tasks
Player energy can fall below -10 but they become incapacitated
Factor in player energy to determine limit

python3.11 community.py --num_members 20 --num_turns 1000 --num_abilities 5 --group_abilities_distribution 4 --abilities_distribution_difficulty hard --group_task_distribution 4 --task_distribution_difficulty hard --g9 20

python3.11 community.py --num_members 20 --num_turns 1000 --num_abilities 5 --g9 20
'''
