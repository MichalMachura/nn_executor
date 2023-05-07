from dataclasses import dataclass, field
from queue import Queue
from typing import Literal, Tuple, List, Dict, Union
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from nn_executor.graph_analyzer import NodeDescription, BranchDescription, GraphDescription


@dataclass
class NodeVisualDescription:
    name: str = ''
    node_idx: int = None
    row: int = None
    col: int = None

    generated_branches: List['BranchVisualDescription'] = field(default_factory=list)
    source_branches: List['BranchVisualDescription'] = field(default_factory=list)

    __x: float = None
    __y: float = None

    @property
    def x(self):
        return self.__x if self.__x is not None else self.row

    @x.setter
    def x(self, x: float):
        self.__x = x

    @property
    def y(self):
        return self.__y if self.__y is not None else self.col

    @y.setter
    def y(self, y: float):
        self.__y = y

    @property
    def coords(self) -> List[float]:
        return [self.x, self.y]

    def add_generated_branch(self, branch: 'BranchVisualDescription'):
        self.generated_branches.append(branch)

    def add_source_branch(self, branch: 'BranchVisualDescription'):
        self.source_branches.append(branch)


@dataclass
class BranchVisualDescription:
    branch_idx: int = None
    nodes: List[NodeVisualDescription] = field(default_factory=list)
    branch_col: int = None

    @property
    def first_row(self):
        return self.src_node.row

    @property
    def last_row(self):
        return self.dst_node.row

    @property
    def src_node(self) -> NodeVisualDescription:
         return self.nodes[0]

    @property
    def dst_node(self) -> NodeVisualDescription:
         return self.nodes[-1]

    def __len__(self) -> int:
        """returns number of edges in branch"""
        return len(self.nodes) - 1

    def add_to_branch_ends(self):
        self.src_node.add_generated_branch(self)
        self.dst_node.add_source_branch(self)


@dataclass
class GraphVisualDescription:
    nodes: List[NodeVisualDescription] = field(default_factory=list)
    branches: List[BranchVisualDescription] = field(default_factory=list)
    branches_splitting_nodes: List[NodeVisualDescription] = field(default_factory=list)

    __COLS: int = None
    __ROWS: int = None

    def __make_nodes(self, nodes: List[NodeDescription]):
        visual_nodes = []
        for node in nodes:
            name = "Input" if node.is_input_node \
                           else "Output" if node.is_output_node \
                                         else str(node.node_module)
            visual_node = NodeVisualDescription(name=name, node_idx=node.node_idx)
            visual_nodes.append(visual_node)

        self.nodes = visual_nodes

    def __make_branches(self, branches: List[BranchDescription]):
        vis_branches = []
        for branch in branches:
            vis_nodes = [self.nodes[node.node_idx] for node in branch.nodes]
            vis_branch = BranchVisualDescription(branch_idx=branch.branch_idx,
                                                 nodes=vis_nodes)
            vis_branch.add_to_branch_ends()

            vis_branches.append(vis_branch)
        self.branches = vis_branches

    def __make_splitting_nodes(self, splitting_nodes: List[NodeDescription]):
        vis_splitting_nodes = []
        for node in splitting_nodes:
            vis_node = self.nodes[node.node_idx]
            vis_splitting_nodes.append(vis_node)

        self.branches_splitting_nodes = vis_splitting_nodes

    def __determinate_abs_dist(self):
        # without zero node - input node
        unknown_len_nodes = list(self.branches_splitting_nodes[1:])
        self.nodes[0].row = 0

        while len(unknown_len_nodes) != 0:
            node = unknown_len_nodes.pop(0)
            #                branch with known len
            available_sources = [b for b in node.source_branches if b.src_node.node_idx not in unknown_len_nodes]
            # all sources are available
            if len(available_sources) == len(node.source_branches):
                # get max path from beginning
                lengths = []
                for branch in available_sources:
                    #   src dist from beg + num of branch's edges
                    L = branch.src_node.row + len(branch)
                    lengths.append(L)
                node.row = max(lengths)
            else:
                unknown_len_nodes.append(node)

        # number of representation rows
        self.__ROWS = max([node.row for node in self.branches_splitting_nodes if node.row is not None]) + 1

    def __assign_branch_column(self):
        sorted_branches = sorted(self.branches.copy(), key=lambda b: len(b))
        # iterate over rows
        for row in range(self.__ROWS):
            occupied_cols = []
            # get occupied columns
            for branch in sorted_branches:
                if branch.src_node.row < row < branch.dst_node.row \
                        and branch.branch_col is not None:
                    occupied_cols.append(branch.branch_col)
            # assign col for branch
            for branch in sorted_branches:
                # if branch covers row and has not assigned column
                if branch.src_node.row < row < branch.dst_node.row \
                        and branch.branch_col is None:
                    # find first free column
                    branch.branch_col = 0
                    while branch.branch_col in occupied_cols:
                        branch.branch_col += 1

                    occupied_cols.append(branch.branch_col)

        # the widest row size
        self.__COLS = max([b.branch_col for b in self.branches if b.branch_col is not None]) + 1

    def __plan_nodes_locations(self):
        # TODO use numpy
        NODES_LOCATION = ([[None for _ in  range(self.__COLS)] for _ in range(self.__ROWS)])
        # reserve column's range for branches
        for branch in self.branches:
            for row in range(branch.first_row+1, branch.last_row):
                NODES_LOCATION[row][branch.branch_col] = -1

        # place internal branches' nodes
        for branch in self.branches:
            step = (branch.last_row - branch.first_row) / len(branch) if len(branch) > 0 else 0
            # without first and last node
            for i, node in enumerate(branch.nodes[1:-1]):
                node.col = branch.branch_col
                node.y = branch.branch_col
                node.row = branch.first_row + i + 1
                node.x = branch.first_row + (i + 1) * step

        # place multi-branch' nodes
        for node in self.branches_splitting_nodes:
            ROW = NODES_LOCATION[node.row]
            # find free place
            for col in range(len(ROW)):
                if ROW[col] is None:
                    ROW[col] = node.node_idx
                    node.col = col
                    break

    @staticmethod
    def make_from_graph_description(gd: GraphDescription) -> 'GraphVisualDescription':
        gvd = GraphVisualDescription()
        gvd.__make_nodes(gd.nodes)
        gvd.__make_branches(gd.branches)
        gvd.__make_splitting_nodes(gd.branches_splitting_nodes)

        gvd.__determinate_abs_dist()
        gvd.__assign_branch_column()
        gvd.__plan_nodes_locations()

        return gvd


HORIZONTAL = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=np.float32)
VERTICAL =   np.array([[-1, 0, 0],
                       [0, -1, 0]], dtype=np.float32)

def ROTATION(degree: float) -> np.ndarray:
    sin = np.sin(np.deg2rad(degree))
    cos = np.cos(np.deg2rad(degree))
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0]], dtype=np.float32)


def draw_graph(gvd: GraphVisualDescription, scale: Tuple[int, int] = (1, 1), transformation: np.ndarray = HORIZONTAL):
    scale_xy = np.array(scale)
    nodes_pos = {}
    for node in gvd.nodes:
        xy = np.array(node.coords)
        s_xy = scale_xy * xy
        t_xy = np.matmul(transformation, np.array(s_xy.tolist() + [1]).reshape(3, 1))
        XY = t_xy.flatten().tolist()
        nodes_pos[node.node_idx] = XY

    fig, ax = plt.subplots()
    connections = []
    for branch in gvd.branches:
        connections.extend([(n1.node_idx, n2.node_idx) for n1, n2 in zip(branch.nodes[:-1], branch.nodes[1:])])
        XYs = [nodes_pos[node.node_idx] for node in branch.nodes]
        XYs = np.array(XYs)
        ax.plot(XYs[:,0], XYs[:,1], '-')
    G = nx.DiGraph(set(connections))

    coords = []
    for node in gvd.nodes:
        xy = nodes_pos[node.node_idx]
        coords.append(xy)

    coords = np.array(coords)
    sc = ax.scatter(coords[:,0], coords[:,1])

    annot = ax.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                annot.set_text(gvd.nodes[ind["ind"][0]].name)
                annot.get_bbox_patch().set_facecolor('g')
                annot.get_bbox_patch().set_alpha(0.9)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    G = nx.DiGraph(connections)

    nx.draw(G, nodes_pos, with_labels=True)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show(block=True)



##### OLD VERSION

def find_abs_len(nodes: Dict[int, Tuple[List[int], List[int]]],
                 branches: List[Tuple[int, int, int]]):
    nodes_abs_len = {node_idx: 0 for node_idx in nodes.keys()}
    # without zero node - input node
    unknown_len_nodes = list(set(nodes.keys()) - set([0]))

    if 0 not in nodes.keys():
        raise RuntimeError("zero not in keys!")

    while len(unknown_len_nodes) != 0:
        node = unknown_len_nodes.pop(0)
        is_src_of_branches, is_dst_of_branches = nodes[node]
        #                beg of source branch has known len
        src_available = [branches[b][0] not in unknown_len_nodes for b in is_dst_of_branches]
        # all sources are available
        if sum(src_available) == len(src_available):
            # get max path from beginning
            lengths = []
            for b_id in is_dst_of_branches:
                src_node, _, branch_len = branches[b_id]
                #   src dist from beg + branch_len
                L = nodes_abs_len[src_node] + branch_len
                lengths.append(L)

            nodes_abs_len[node] = max(lengths)

        else:
            unknown_len_nodes.append(node)

    return nodes_abs_len


def assign_branch_columns(b_desc: List[Tuple[int, int, int]], node_pos:Dict[int, int], rows_num: int):
    branches_col = {i: -1 for i in range(len(b_desc))}
    for row in range(rows_num):
        occupied_cols = []
        # get occupied columns
        for b_idx, (node_beg, node_end, b_len) in enumerate(b_desc):
            if node_pos[node_beg] <= row <= node_pos[node_end] and branches_col[b_idx] >= 0:
                occupied_cols.append(branches_col[b_idx])
        # assign col for branch
        for b_idx, (node_beg, node_end, b_len) in enumerate(b_desc):
            if node_pos[node_beg] < row < node_pos[node_end] and branches_col[b_idx] < 0:
                col = 0
                while col in occupied_cols:
                    col += 1
                branches_col[b_idx] = col
                occupied_cols.append(col)

    return branches_col


def plan_a_graph(nodes_names: List[str], branches: List[List[Tuple[int, ...]]], scale_YX=(0.5, 2)):
    import numpy as np
    np.set_printoptions(edgeitems=50, linewidth=1500)

    b_desc: List[Tuple[int, int, int]] = [] # (beg, end, len)
    # {node_idx : (is_source_of_these_branches, is_dst_of_these_branches)}
    nodes: Dict[int, Tuple[List[int], List[int]]] = {}

    for i, b in enumerate(branches):
        branch_beg = b[0][0]
        branch_end = b[-1][0]
        b_desc += [(branch_beg, branch_end, len(b)-1)]

        for col in [branch_beg, branch_end]:
            if col not in nodes.keys():
                nodes[col] = ([], [])

        nodes[branch_end] = (nodes[branch_end][0],       nodes[branch_end][1] + [i])
        nodes[branch_beg] = (nodes[branch_beg][0] + [i], nodes[branch_beg][1])

    node_pos = find_abs_len(nodes, b_desc)

    TOT_LEN = max(node_pos.values())
    LEVEL_PARALLEL_BRANCHES = [0] + [0] * TOT_LEN

    for node_idx, pos in node_pos.items():
        source_branches = nodes[node_idx][1]
        for branch_idx in source_branches:
            src_node, dst_node, branch_len = b_desc[branch_idx]
            src_pos = node_pos[src_node]
            # increment levels covered by this branch
            for i in range(src_pos + 1, # exclude src node incrementation
                           pos):
                LEVEL_PARALLEL_BRANCHES[i] += 1
        # add this node
        LEVEL_PARALLEL_BRANCHES[pos] += 1

    WIDTH = max(LEVEL_PARALLEL_BRANCHES)

    b_cols = assign_branch_columns(b_desc, node_pos, TOT_LEN)

    NODES_LOCATION = [[None for _ in  range(WIDTH)] for _ in range(TOT_LEN+1)]
    for b_idx, b_col in b_cols.items():
        b_beg, b_end, b_len = b_desc[b_idx]
        b_beg, b_end = node_pos[b_beg], node_pos[b_end]
        # reserve column range for branch - it could be longer than num of nodes
        for row in range(b_beg+1, b_end):
            NODES_LOCATION[row][b_col] = -1

        # assign network's nodes without branches' nodes, so first and last node of branch
        for i, (node_idx, node_output_idx) in enumerate(branches[b_idx][1:-1]):
            row = b_beg + i + 1
            NODES_LOCATION[row][b_col] = node_idx
    # assign branches' shared nodes
    for shared_node_idx, node_row in node_pos.items():
        COLS = NODES_LOCATION[node_row]
        for col, node_idx in enumerate(COLS):
            if node_idx is None:
                NODES_LOCATION[node_row][col] = shared_node_idx
                break

    pos = {}
    for row, ROW in enumerate(NODES_LOCATION):
        for col, node_idx in enumerate(ROW):
            if node_idx is not None and node_idx >= 0:
                pos[node_idx] = (row * scale_YX[1], col * scale_YX[0])

    for branch in branches:
        points = [(node_idx,) + pos[node_idx] for (node_idx, _) in branch]
        points = np.array(points)
        plt.plot(points[:,1], points[:,2], '-o',)
        for p in points:
            plt.text(p[1], p[2], str(int(np.round(p[0]))))

    plt.show()


    connections = []
    multi_connections = []
    for b in branches:
        local_connections = []
        for (n1, _), (n2, _) in zip(b[:-1], b[1:]):
            local_connections.append((n1, n2))

        connections.extend(local_connections)
        multi_connections.append(local_connections)

    connections = list(set(connections))

    G = nx.DiGraph(connections)
    for i, name in enumerate(nodes_names):
        G.nodes[i]['name'] = name

    nx.draw(G, pos, with_labels=True)
    plt.show(block=True)

    return

