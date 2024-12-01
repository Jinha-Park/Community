import traceback
import networkx as nx
import matplotlib.pyplot as plt

def get_energy_cost(player_abilities, task):
    return max(sum(max(task[i]- player_abilities[i],0) for i in range(len(task))),0)

def computeAbilities(community):
    pairAbilities = {}
    for i, player in enumerate(community.members):
        pairAbilities[f"P_{player.id}"] = player.abilities
        for partner in community.members[i+1:]:
            pairAbilities[f"P_{partner.id}"] = partner.abilities
            total_abilities = [max(player.abilities[k], partner.abilities[k]) for k in range(len(player.abilities))]
            pairAbilities[f"P_{player.id}-P_{partner.id}"] = total_abilities
    return pairAbilities

def buildGraph(community, abilities):
    tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)
    G = nx.DiGraph()

    source = "source"
    sink = "sink"
    G.add_node(source)
    G.add_node(sink)

    for tid,task in tasks:
        # Source to tasks - weight 0 TODO:Will changing capacity mean a task can be done by multiple players?
        task_id = f"T_{tid}"
        G.add_edge(source, task_id, weight=0, capacity=1)
        for player in abilities:
            
            ability = abilities[player]
            cost = get_energy_cost(ability, task)
            
            #TODO:
            #Right now the weight is just energy cost. But have to add in complex cost function. Also have to 
            #Make sure players don't get incapicated and rest.
            
            if "-" in player:  # Pair
                G.add_edge(task_id, player, weight=cost/2, capacity=1)    
                player1, player2 = player.split("-")

                 # Tasks to conflict nodes
                G.add_edge(player1, f"C_{player1}", weight=0, capacity=1)
                G.add_edge(player2, f"C_{player2}", weight=0, capacity=1)
            else:  # Single player
                G.add_edge(task_id, player, weight=cost, capacity=1)
                G.add_edge(player, f"C_{player}", weight=0, capacity=1)

    # Conflict nodes to sink
    for player in community.members:
        conflict_node = f"C_P_{player.id}"
        G.add_edge(conflict_node, sink, weight=0, capacity=1)  # Conflict nodes can have at most one assignment

    return G

def optimalAssignments(community, abilities):
    G = buildGraph(community, abilities)
    flow_dict = nx.max_flow_min_cost(G, "source", "sink")

    assignments = {}
    total_cost = 0
    print(flow_dict["source"])

    for player in flow_dict["source"]:
        if flow_dict["source"][player] > 0:  # Player or pair assigned
            for task in flow_dict[player]:
                if flow_dict[player][task] > 0:  # Task assigned
                    assignments[player] = task
                    cost = G.edges[player, task]['weight']
                    total_cost += cost
                    print(f"Assigned {player} -> {task}, Cost: {cost}")

    print("\nOptimal Assignments:")
    for player, task in assignments.items():
        cost = G.edges[player, task]['weight']
        print(f"{player} -> {task} (Cost: {cost})")

    print(f"\nTotal Cost: {total_cost}")
    return assignments      

def get_id(s):
    return int(''.join([char for char in s if char.isdigit()]))

def assignPlayers(community):
    abilities = computeAbilities(community)
    assignments = optimalAssignments(community, abilities)
    
    return {
        get_id(player1) if '-' in assignment else get_id(assignment): [get_id(task), get_id(player2) if '-' in assignment else get_id(assignment)]
        for task, assignment in assignments.items()
        for player1, player2 in [(assignment.split('-') + [None])[:2]] 
    }


def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player.'''

    preferences = []
    try:
        assignments = assignPlayers(community)
        if player.id in assignments:
            print("Im assigned")
            task, pair = assignments[player.id]
            
            #Need to do task alone
            if pair == player.id:
                print("Doing it alone")
                return []
            
            return [assignments[player.id]]

        else:
            print("I am free ", player.id)
            return []
    except Exception as e:
        print(e)
        traceback.print_exc()


    # TODO: Move this logic before weights!!
    task_avg_difficulty = sum(sum(task) for task in community.tasks) / len(community.tasks)
    player_avg_skill = sum(sum(member.abilities) for member in community.members) / len(community.members)

    primary_energy_limit = 0  # Allow energy to drop but not too far into the negatives

    # Calculate the ratio of difficulty to skill
    difficulty_ratio = task_avg_difficulty / max(player_avg_skill, 1e-6)  # Avoid division by zero

    # Map difficulty ratio to the range [-10, primary_energy_limit]
    if difficulty_ratio <= 1:
        secondary_energy_limit = primary_energy_limit  # Tasks are manageable
    else:
        secondary_energy_limit = primary_energy_limit - (difficulty_ratio - 1) * 10
        secondary_energy_limit = max(secondary_energy_limit, -10)  # Ensure it doesn't drop below -10

    # Sort tasks by total difficulty (descending)
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_id, task in sorted_tasks:
        best_partner = None
        best_remaining_energy = -float('inf')  # Track the best post-task energy state

        # Check if the player can complete the task alone
        if all(task[i] <= player.abilities[i] for i in range(len(task))):
            continue  # Skip partnering for tasks the player can complete alone

        for partner in community.members:
            if partner.id == player.id or partner.incapacitated:
                continue  # Skip self or incapacitated players

            # Check if the partner can complete the task alone
            if all(task[i] <= partner.abilities[i] for i in range(len(task))):
                continue  # Skip partnering for tasks the partner can complete alone

            # Calculate energy cost for both players
            energy_cost = sum(max(task[i] - max(player.abilities[i], partner.abilities[i]), 0) for i in range(len(task))) / 2
            player_remaining_energy = player.energy - energy_cost
            partner_remaining_energy = partner.energy - energy_cost

            # Allow partnering even if energy dips below the primary limit
            if (
                player_remaining_energy > secondary_energy_limit
                and partner_remaining_energy > secondary_energy_limit
            ):
                # Choose the partner that maximizes the minimum remaining energy
                if min(player_remaining_energy, partner_remaining_energy) > best_remaining_energy:
                    best_partner = partner.id
                    best_remaining_energy = min(player_remaining_energy, partner_remaining_energy)

        # Add the task and partner to preferences if a valid partner is found
        if best_partner is not None:
            preferences.append([task_id, best_partner])

    return preferences


def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually.'''
    preferences = []
    try:
        assignments = assignPlayers(community)
        if player.id in assignments:
            # print("Solo assigned", player.id, assignments[player.id])
            return [assignments[player.id][0]]
    except:
        traceback.print_exc()

    #TODO: Move logic before weight setting!
    task_avg_difficulty = sum(sum(task) for task in community.tasks) / max(len(community.tasks), 1e-6) 
    player_avg_skill = sum(sum(member.abilities) for member in community.members) / max(len(community.members), 1e-6)

    primary_energy_limit = 0  # Allow energy to drop but not too far into the negatives

    # Calculate the ratio of difficulty to skill
    difficulty_ratio = task_avg_difficulty / max(player_avg_skill, 1e-6)  # Avoid division by zero

    # Map difficulty ratio to the range [-10, primary_energy_limit]
    if difficulty_ratio <= 1:
        secondary_energy_limit = primary_energy_limit  # Tasks are manageable
    else:
        secondary_energy_limit = primary_energy_limit - (difficulty_ratio - 1) * 10
        secondary_energy_limit = max(secondary_energy_limit, -10)  # Ensure it doesn't drop below -10

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
How to reduce time - We don't have to recompute everytime since the matching is per round?
'''