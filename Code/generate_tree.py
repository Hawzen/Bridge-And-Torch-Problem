from Solution.bridge_problem import *
import pydotplus


def _UCS_graph(problem: BridgeProblem, optimize_repeated=True) -> (list, list):
    """ Used for graphing, returns solution as well as all nodes in a list """
    all_nodes = [problem.initial_node]
    pq = [problem.initial_node]
    closed = set()

    while True:
        assert pq
        node = heappop(pq)

        if problem.goal_test(node):
            return problem.get_ancestors(node), all_nodes

        closed.add(node)

        children = problem.expand(node)

        for bn in children:
            if bn not in closed and bn not in pq:
                heappush(pq, bn)
            elif bn in pq and bn.path_cost < pq[pq.index(bn)].path_cost:
                pq.remove(bn)  # Removes node with similar state but different cost
                pq.append(bn)  # Add less cost node
                heapify(pq)

        all_nodes.extend(children)


def _DLS_graph(problem: BridgeProblem, limit: int, allnodes: list) -> ([BridgeNode], [BridgeNode]):
    """Return Depth Limited Search solution (list of nodes) or an empty list when cutoff happens"""
    return _RDLS_graph(problem.initial_node, problem, limit, allnodes)


def _RDLS_graph(node: BridgeNode, problem: BridgeProblem, limit: int, allnodes: list):
    """Result is either a list of nodes, and empty list (cutoff) or a failure None"""
    cutoff_occurred = False
    if problem.goal_test(node):
        return problem.get_ancestors(node), allnodes
    if node.depth == limit:
        return [], allnodes
    children = problem.expand(node)
    allnodes.extend(children)
    for bn in children:
        result, allnodes = _RDLS_graph(bn, problem, limit, allnodes)
        if isinstance(result, list) and not result:
            cutoff_occurred = True
        elif result is not None:
            return result, allnodes
    return ([], allnodes) if cutoff_occurred else (None, allnodes)


def _IDS_graph(problem: BridgeProblem) -> ([BridgeNode], [BridgeNode]):
    """Return Iterated Deepening Search solution along with metrics"""
    allnodes = [problem.initial_node]
    for depth in count():
        result, allnodes = _DLS_graph(problem, depth, allnodes)
        if isinstance(result, list) and result:
            return result, allnodes


def _AStar_graph(problem: BridgeProblem) -> (list, list):
    """Used for graphing, returns solution as well as all nodes in a list"""
    all_nodes = [problem.initial_node]
    pq = [(problem.initial_node.path_cost + problem.h(problem.initial_node.state), problem.initial_node)]
    closed = set()

    while True:
        assert pq
        priority, node = heappop(pq)

        if problem.goal_test(node):
            return problem.get_ancestors(node), all_nodes

        closed.add(node)

        children = problem.expand(node)

        for node in children:
            priority = node.path_cost + problem.h(node.state)
            bn = (priority, node)

            inpq = None
            for i, (_, pq_node) in enumerate(pq):
                if node == pq_node: inpq = i

            if node not in closed and inpq is None:
                heappush(pq, bn)
            elif inpq is not None and bn < pq[inpq]:
                pq.pop(inpq)
                pq.append(bn)
                heapify(pq)

        all_nodes.extend(children)


case = (3, (3, 2, 5))
# case = (4, (2, 3, 2, 5))
# case = (2, (1, 2))
k, _ = case

prob = BridgeProblem(case)
# sol, allnodes = _UCS_graph(prob)
# sol, allnodes = _IDS_graph(prob)
sol, allnodes = _AStar_graph(prob)

solution_color = "color me"
for node in sol:
    node.__setattr__("color", solution_color)

merge, show_heuristic = False, True
pydot_nodes = []
node_to_nd = {}
k_to_colorpallet = {
    n: f"blues{n}" for n in range(3, 9) # purd, blues
}

for i, node in enumerate(allnodes):
    state = "".join(["1" if b else "0" for b in node.state[0]])
    node_name = f"{str(i)}_{state}" if not merge else state
    s = (node_name.replace(":", "_").replace(" ", "_"))
    nd = pydotplus.graphviz.Node(name=s)

    nd.set("fillcolor", sum(int(n) for n in state) + 1)
    nd.set("colorscheme", k_to_colorpallet[k+1])
    nd.add_style('filled')
    node_to_nd[i] = nd
    pydot_nodes.append(nd)

edge_done = set()
graph = pydotplus.Dot(graph_type="graph")
for i, nd in enumerate(pydot_nodes):
    graph.add_node(nd)
    node = allnodes[i]
    if node.parent is None:
        continue

    for index, nodep in enumerate(allnodes):
        if id(nodep) == id(node.parent): j = index
    parent_nd = node_to_nd[j]

    if merge:
        pair = frozenset((nd.get_name(), parent_nd.get_name()))
        if pair in edge_done:
            continue
        edge_done.add(pair)
        edge = pydotplus.Edge(nd, parent_nd)
    else:
        edge = pydotplus.Edge(nd, parent_nd)
        label = f"{node.path_cost}, {prob.h(node.state)}" if show_heuristic else node.path_cost
        edge.set("label", label)

    if getattr(allnodes[j], "color", None) == getattr(node, "color", None) == solution_color:
        edge.set("color", "red")
        edge.set("penwidth", "3.0")

    graph.add_edge(edge)

graph.write("image.dot")
