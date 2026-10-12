from collections import deque, OrderedDict, namedtuple
import heapq
from heapq import heappop, heappush
from itertools import count
from typing import NamedTuple, Tuple


class SearchResult:
    def __init__(self, path, expanded):
        self.path = path
        self.expanded = expanded


class AStarNode(NamedTuple):
    f: float
    insertion_number: int
    g: float
    h: float
    state: tuple
    parent: AStarNode | None


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
    return abs(state_row - goal_row + state_col - goal_col)


def astar(campus, start, goal):
    # Use heapq as a priority queue and manhattan() as the heuristic.
    # Track best known g-costs and parent states in dictionaries.
    insertion_number = 0
    start_node = AStarNode(f=0, g=0, h=0, state=start, parent=None, insertion_number=insertion_number)
    insertion_number += 1
    frontier = [start_node]
    expansions: OrderedDict[Tuple, AStarNode] = OrderedDict()

    while goal not in expansions and len(frontier) != 0:
        expanded = heappop(frontier)
        previous = expansions.get(expanded.state)
        if previous is None or expanded < previous:
            expansions[expanded.state] = expanded
            for _, neighbor in campus.neighbors(expanded.state):
                insertion_number += 1
                g = expanded.g + 1
                h = manhattan(neighbor, goal)
                f = g + h
                neighbor_node = AStarNode(f=f, g=g, h=h, insertion_number=insertion_number, state=neighbor,
                                          parent=expanded)
                heappush(frontier, neighbor_node)

    expansion_count = len(expansions) - 1  # don't count start node
    if goal not in expansions:  # goal not found
        return SearchResult(None, expansion_count)
    trace_node = expansions[goal]
    path = [trace_node.state]
    while trace_node.parent is not None:
        trace_node = trace_node.parent
        path.append(trace_node.state)
    return SearchResult(path[::-1], expansion_count)


def epsilon_greedy(q_values, state, actions, epsilon, rng):
    """TODO 3: Choose an action with epsilon-greedy exploration."""
    # q_values is a dictionary. Its keys are (state, action) pairs.
    # Example key: ((1, 1), "RIGHT"). Unseen entries have value 0.0.
    raise NotImplementedError("Complete TODO 3: epsilon_greedy")


def q_learning_update(q_values, state, action, reward, next_state,
                      actions, alpha, gamma, terminal):
    """TODO 4: Update exactly one Q-table entry in place; return nothing."""
    # Get Q(state, action), defaulting to 0.0.
    # If terminal is True, the target is reward with no continuation term.
    # Otherwise target = reward + gamma * max Q(next_state, next_action).
    # Consider every action, including unseen entries with value 0.0.
    # Apply Q <- Q + alpha * (target - Q).
    raise NotImplementedError("Complete TODO 4: q_learning_update")
