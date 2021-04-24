import pandas as pd
from timeit import timeit
from functools import total_ordering
from more_itertools import grouper
from typing import NewType, Tuple
from itertools import *
from heapq import *

test_case = (int, tuple)
Action = list
State = NewType("State", Tuple[Tuple[bool], bool])


def read_test_cases(filename: str) -> [test_case]:
    """Reads a file filename and returns a list of test_cases"""
    test_cases = []
    with open(filename) as file:
        for k, costs in grouper(2, file.readlines()[1:]):
            k, costs = int(k), tuple(int(n) for n in costs.split(", "))
            test_cases.append((k, costs))
    return test_cases


@total_ordering
class BridgeNode:
    """ Represents the state of game at any one point """

    def __init__(self, state: State, depth: int = None, path_cost: int = None, parent=None, action=""):
        self.parent = parent
        self.state = state
        self.depth = depth
        self.path_cost = path_cost
        self.action = action

    def __le__(self, other):
        return self.path_cost <= other.path_cost

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def __repr__(self):
        return f"People:{self.state[0]}, Torch:{self.state[1]}, " \
               f"Depth:{self.depth}, Path Cost:{self.path_cost}, Action:{self.action}"

    __str__ = __repr__


class BridgeProblem:
    """ Includes functions and attributes defining a problem """
    def __init__(self, inputs: test_case):
        self.min_num_ppl_crossing = 1
        self.max_num_ppl_crossing = 2
        k, costs = inputs
        self._costs = costs
        self.k = k
        self.initial_node = BridgeNode(State((tuple(False for _ in costs), False)), depth=0, path_cost=0,
                                       action="Start")
        self._to_cost = {i: cost for i, cost in enumerate(costs)}

    def successor(self, node: BridgeNode) -> [(Action, State)]:
        m = self.min_num_ppl_crossing
        n = self.max_num_ppl_crossing

        if not node.state[1]:
            m = 2

        positions, flash_is_west = node.state
        res = []
        legal_positions = []

        for i, isWest in enumerate(positions):
            if isWest == flash_is_west:
                legal_positions.append(i)

        for indices in chain.from_iterable(combinations(legal_positions, r) for r in range(m, n + 1)):
            newPositions = tuple(not pos if i in indices else pos for i, pos in enumerate(positions))
            res.append(([*indices], State((newPositions, not flash_is_west))))
        return res

    def expand(self, node: BridgeNode) -> {BridgeNode}:
        successors = []
        for action, state in self.successor(node):
            bn = BridgeNode(state, node.depth + 1, parent=node)
            bn.path_cost = node.path_cost + max(self._to_cost[pos] for pos in action)
            bn.action = action
            successors.append(bn)
        return successors

    @staticmethod
    def goal_test(node: BridgeNode) -> bool:
        return all(node.state[0])

    @staticmethod
    def get_ancestors(node: BridgeNode) -> [BridgeNode]:
        res = []
        while node:
            res.append(node)
            node = node.parent
        return res[::-1]

    def h(self, state: State) -> int:
        costs = [self._costs[i] for i, b in enumerate(state[0]) if not b]
        if not costs:
            return 0

        # New h
        costs = sorted(costs, reverse=True)
        if len(costs) % 2 != 0:
            costs.append(0)
        cost = 0
        for val, _ in grouper(2, costs):
            cost += val

        return cost


class Metrics:
    """ Used in algorithms to account different metrics """
    def __init__(self, solution_cost=0, search_cost=0, space_requirement=0):
        self.solution_cost = solution_cost
        self.search_cost = search_cost
        self._space_requirement = space_requirement
        self.max_space_requirement = space_requirement

    def __repr__(self):
        return "Metrics({!r})".format({var: self.__dict__.get(var) for var in ["solution_cost",
                                                                               "search_cost",
                                                                               "max_space_requirement",
                                                                               "depth"]})

    @property
    def space_requirement(self):
        return self._space_requirement

    @space_requirement.setter
    def space_requirement(self, value):
        self.max_space_requirement = max(self.max_space_requirement, value)
        self._space_requirement = value

    @space_requirement.getter
    def space_requirement(self):
        return self._space_requirement


def UCS(problem: BridgeProblem) -> ([BridgeNode], Metrics):
    """Return Uniform Cost Search solution (list of nodes)"""
    metrics = Metrics()
    pq = [problem.initial_node]
    explored = set()

    while pq:
        node = heappop(pq)

        if problem.goal_test(node):
            metrics.solution_cost, metrics.depth = node.path_cost, node.depth
            return problem.get_ancestors(node), metrics

        explored.add(node)

        children = problem.expand(node)
        metrics.search_cost += len(children)

        for bn in children:
            if bn not in explored and bn not in pq:
                heappush(pq, bn)
            elif bn in pq and bn.path_cost < pq[pq.index(bn)].path_cost:
                pq.remove(bn)  # Removes node with a similar state but different cost
                heappush(pq, bn) # Add less cost node

        metrics.space_requirement = max(metrics.space_requirement, len(pq))

    raise ValueError(f"Priority Queue is empty\nMetrics: {metrics}")


def DLS(problem: BridgeProblem, limit: int, metrics) -> ([BridgeNode], Metrics):
    """Return Depth Limited Search solution (list of nodes) or an empty list when cutoff happens"""

    return RDLS(problem.initial_node, problem, limit, metrics, explored=set())


def RDLS(node: BridgeNode, problem: BridgeProblem, limit: int, metrics: Metrics, explored: set = None):
    """Result is either a list of nodes, and empty list (cutoff) or a failure None"""
    metrics.space_requirement += 1
    metrics.search_cost += 1
    cutoff_occurred = False

    # Optimization step
    if explored is not None:
        if node.state in explored:
            return [], metrics
        explored.add(node.state)

    if problem.goal_test(node):
        metrics.solution_cost, metrics.depth = node.path_cost, node.depth
        return problem.get_ancestors(node), metrics

    # Cut off step
    if node.depth == limit:
        return [], metrics

    for bn in problem.expand(node):
        # Recursive step
        result, result_metrics = RDLS(bn, problem, limit, metrics, explored)
        metrics.space_requirement -= 1
        if isinstance(result, list) and not result:
            cutoff_occurred = True
        else:
            return result, metrics
    return ([], metrics) if cutoff_occurred else (None, metrics)


def IDS(problem: BridgeProblem) -> ([BridgeNode], Metrics):
    """Return Iterated Deepening Search solution along with metrics"""
    metrics = Metrics()
    for depth in count():
        result, metrics = DLS(problem, depth, metrics)
        if isinstance(result, list) and result:
            return result, metrics


def AStar(problem: BridgeProblem) -> ([BridgeNode], Metrics):
    """Return A* solution along with metrics"""
    metrics = Metrics(space_requirement=1)
    pq = [(problem.initial_node.path_cost + problem.h(problem.initial_node.state), problem.initial_node)]
    explored = set()

    while pq:
        priority, node = heappop(pq)

        if problem.goal_test(node):
            metrics.solution_cost, metrics.depth = node.path_cost, node.depth
            return problem.get_ancestors(node), metrics

        explored.add(node)

        children = problem.expand(node)
        metrics.search_cost += len(children)

        for node in children:
            priority = node.path_cost + problem.h(node.state)
            bn = (priority, node)

            in_pq = None
            for i, (_, pq_node) in enumerate(pq):
                if node == pq_node: in_pq = i

            if node not in explored and in_pq is None:
                heappush(pq, bn)

            elif in_pq is not None and bn < pq[in_pq]:
                pq.pop(in_pq)
                heappush(pq, bn)

        metrics.space_requirement = max(metrics.space_requirement, len(pq))

    raise ValueError(f"Priority Queue is empty\nMetrics: {metrics}")


def generate_report(test_cases, df_only=True):
    """Returns a pandas DataFrame report of executing each algorithm on each test case (used for report)"""
    solutions = {}
    evaluation = {}

    strategies = [UCS, IDS, AStar]
    strategy_to_name = {strategy: name for strategy, name in zip(strategies, ["UCS", "IDS", "A*"])}

    test_cases_name = [" ".join(str(i) for i in ["(", *costs, ")"]) for k, costs in test_cases]
    test_cases_to_name = {case: name for case, name in zip(test_cases, test_cases_name)}

    for case in test_cases:
        prob = BridgeProblem(case)
        for strategy in [UCS, IDS, AStar]:
            time = timeit(lambda: strategy(prob), number=1)
            sol, metric = strategy(prob)
            evaluation[(test_cases_to_name[case], strategy_to_name[strategy])] = (metric, time)
            solutions[(test_cases_to_name[case], strategy_to_name[strategy])] = sol
            print(f"Finish {case}")

    test_cases_x_strategies = product(test_cases_name, ["Solution Cost", "Search Cost", "Space Requirement",
                                                        "Time (s)"])
    df = pd.DataFrame(columns=["UCS", "IDS", "A*"],
                      index=pd.MultiIndex.from_tuples(test_cases_x_strategies, names=["Test Case", "Metric"]))

    for (case, strategy), (metric, time) in evaluation.items():
        df[strategy][df.index.get_level_values('Test Case') == case] = [metric.solution_cost,
                                                                        metric.search_cost,
                                                                        metric.space_requirement,
                                                                        time]

    if df_only:
        return df

    move_length = 20

    def format_solution(solution):
        moves = []
        action = "Move"
        for step in solution[1:]:
            moves.append(f"{action} a{', a'.join(str(k) for k in step.action)}")
            action = "Move" if action != "Move" else "Return"
        for i in range(move_length - len(moves)):
            moves.append("")
        return moves

    sol_dfs = {}

    for (case, strategy), sol in solutions.items():
        if strategy not in sol_dfs:
            sol_dfs[strategy] = sol_df = pd.DataFrame(columns=test_cases_name, index=list(range(move_length)))
        sol_dfs[strategy][case] = format_solution(sol)

    return df, sol_dfs


def print_report(hard=False) -> None:
    """ Loads up test_cases.txt and performance & solutions of all algorithms in latex"""
    if hard:
        test_cases = read_test_cases("test_cases_hard.txt")
    else:
        test_cases = read_test_cases("test_cases.txt")
    df, solutions = generate_report(test_cases)
    df = generate_report(test_cases)
    print(df.to_latex())
    for n, d in solutions.items():
        print(n, d.to_latex())


def input_and_run() -> None:
    """Asks for test case inputs, prints solution and metrics of every test case for every strategy"""
    num_cases = int(input("Input the number of test cases: "))
    test_cases = []
    for _ in range(num_cases):
        k = int(input("Input K (> 1): "))
        if k < 2:
            raise ValueError("can't have k <= 1")
        costs = tuple(int(c) for c in input("Input comma separated costs: ").split(","))
        test_cases.append((k, costs))

    sep = "_ _" * 10
    for case in test_cases:
        print(f"\n\n\n{sep}\nK:\t{case[0]}\nCosts:\t{case[1]}\n\n")  # Prints info for each test case
        problem = BridgeProblem(case)
        for strategy in [UCS, IDS, AStar]:
            solution, metrics = strategy(problem)

            sol_formatted = []
            for node in solution[1:]:
                sol_formatted.append("\n\t")
                sol_formatted.append(", ".join(f"Move a{mv}" for mv in node.action))
            sol_formatted = "".join(sol_formatted)

            print(f"Strategy:\t{strategy.__name__}\nSolution:{sol_formatted}\nMetrics:\t{metrics}\n")


if __name__ == "__main__":
    # case_example = (10, (5, 5, 5, 5, 5, 5, 5, 5, 5, 5))
    # case_example = (5, (1, 3, 2, 5, 1))
    # case_example = (3, (3, 2, 5))
    # prob = BridgeProblem(case_example)

    # ## Solve problem
    # sol, metr = UCS(prob)
    # sol, metr = IDS(prob)
    # sol, metr = AStar(prob)

    # print(sol, metr)

    input_and_run()
