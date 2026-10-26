from collections import deque
from heapq import heappop, heappush
from typing import NamedTuple


class SearchResult:
    def __init__(self, path, expanded):
        self.path = path
        self.expanded = expanded


class AStarNode(NamedTuple):
    f: float
    insertion_index: int
    g: float
    state: tuple


def reconstruct_path(parents, goal):
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = parents[state]
    path.reverse()
    return path


def bfs(campus, start, goal):
    frontier = deque([start])
    parents = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)
        expanded += 1
        for action, successor in campus.neighbors(state):
            if successor not in parents:
                parents[successor] = state
                frontier.append(successor)
    return SearchResult(None, expanded)


def manhattan(state, goal):
    # Add the absolute row difference and the absolute column difference.
    state_row, state_col = state
    goal_row, goal_col = goal
    return abs(state_row - goal_row) + abs(state_col - goal_col)


def astar(campus, start, goal):
    # Use heapq as a priority queue and manhattan() as the heuristic.
    # Track best known g-costs and parent states in dictionaries.
    insertion_index = 0
    start_node = AStarNode(f=0, insertion_index=insertion_index, g=0, state=start)
    frontier = [start_node]
    parents = {start: None}
    best_g = {start: 0.0}
    expanded = 0

    while frontier:
        expanded_node = heappop(frontier)

        if expanded_node.g != best_g[expanded_node.state]:
            continue  # better path already discovered

        if expanded_node.state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)

        expanded = expanded + 1
        for _, neighbor in campus.neighbors(expanded_node.state):
            g = expanded_node.g + 1
            if g < best_g.get(neighbor, float('inf')):
                insertion_index += 1
                best_g[neighbor] = g
                parents[neighbor] = expanded_node.state

                h = manhattan(neighbor, goal)
                f = g + h
                neighbor_node = AStarNode(f=f, insertion_index=insertion_index, g=g, state=neighbor)

                heappush(frontier, neighbor_node)
    return SearchResult(None, expanded)


def epsilon_greedy(q_values, state, actions, epsilon, rng):
    # q_values is a dictionary. Its keys are (state, action) pairs.
    # Example key: ((1, 1), "RIGHT"). Unseen entries have value 0.0.
    if rng.random() < epsilon:
        return rng.choice(actions)

    best_q = max(q_values.get((state, a), 0.0) for a in actions)
    best_actions = [
        a for a in actions
        if q_values.get((state, a), 0.0) == best_q  # default 0.0
    ]
    return rng.choice(best_actions)  # tie breaks or just selects best_q


def q_learning_update(q_values, state, action, reward, next_state,
                      actions, alpha, gamma, terminal):
    # Get Q(state, action), defaulting to 0.0.
    # If terminal is True, the target is reward with no continuation term.
    # Otherwise target = reward + gamma * max Q(next_state, next_action).
    # Consider every action, including unseen entries with value 0.0.
    # Apply Q <- Q + alpha * (target - Q).
    current_q = q_values.get((state, action), 0.0)
    if terminal:
        target = reward
    else:
        target = reward + gamma * max(
            q_values.get((next_state, next_action), 0.0)  # default if unseen
            for next_action in actions
        )

    q_values[(state, action)] = current_q + alpha * (target - current_q)
