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

    generates_branches: List['BranchVisualDescription'] = field(default_factory=list)
    finishes_branches: List['BranchVisualDescription'] = field(default_factory=list)

    _x: float = None
    _y: float = None

    @property
    def x(self):
        return self._x if self._x is not None else self.row

    @x.setter
    def x(self, x: float):
        self._x = x

    @property
    def y(self):
        return self._y if self._y is not None else self.col

    @y.setter
    def y(self, y: float):
        self._y = y

    @property
    def coords(self) -> List[float]:
        return [self.x, self.y]

    def add_generated_branch(self, branch: 'BranchVisualDescription'):
        self.generates_branches.append(branch)

    def add_finished_branch(self, branch: 'BranchVisualDescription'):
        self.finishes_branches.append(branch)


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
        self.dst_node.add_finished_branch(self)


@dataclass
class GraphVisualDescription:
    nodes: List[NodeVisualDescription] = field(default_factory=list)
    branches: List[BranchVisualDescription] = field(default_factory=list)
    splitting_nodes: List[NodeVisualDescription] = field(default_factory=list)

    _COLS: int = None
    _ROWS: int = None

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

        self.splitting_nodes = vis_splitting_nodes

    def __determinate_abs_dist(self):
        # without zero node - input node
        unknown_len_nodes = list(self.splitting_nodes[1:])
        self.nodes[0].row = 0

        while len(unknown_len_nodes) != 0:
            node = unknown_len_nodes.pop(0)
            #                branch with known len
            available_sources = [b for b in node.finishes_branches if b.src_node.node_idx not in unknown_len_nodes]
            # all sources are available
            if len(available_sources) == len(node.finishes_branches):
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
        self._ROWS = max([node.row for node in self.splitting_nodes if node.row is not None]) + 1

    def __assign_branch_column(self):
        # sorted_branches = sorted(self.branches.copy(), key=lambda b: len(b))[::-1]
        sorted_branches = self.branches
        # iterate over rows
        for row in range(self._ROWS):
            occupied_cols = []
            # get occupied columns
            for branch in sorted_branches:
                if branch.src_node.row <= row <= branch.dst_node.row \
                        and branch.branch_col is not None:
                    occupied_cols.append(branch.branch_col)
            # assign col for branch
            for branch in sorted_branches:
                # if branch covers row and has not assigned column
                if branch.src_node.row <= row <= branch.dst_node.row \
                        and branch.branch_col is None:
                    # find first free column
                    branch.branch_col = 0
                    while branch.branch_col in occupied_cols:
                        branch.branch_col += 1

                    occupied_cols.append(branch.branch_col)

        # the widest row size
        self._COLS = max([b.branch_col for b in self.branches if b.branch_col is not None]) + 1

    def __plan_nodes_locations(self):
        RESERVED = -1
        FREE = None
        OCCUPIED = 1

        # TODO use numpy
        NODES_LOCATION = ([[FREE for _ in  range(self._COLS)] for _ in range(self._ROWS)])
        # reserve column's range for branches
        for branch in self.branches:
            for row in range(branch.first_row+1, branch.last_row):
                NODES_LOCATION[row][branch.branch_col] = RESERVED

        # place internal branches' nodes
        for branch in self.branches:
            step = (branch.last_row - branch.first_row) / len(branch) if len(branch) > 0 else 0
            # without first and last node
            for i, node in enumerate(branch.nodes[1:-1]):
                node.col = branch.branch_col
                node.y = branch.branch_col
                node.row = branch.first_row + i + 1
                node.x = branch.first_row + (i + 1) * step

        # place splitting nodes
        for node in self.splitting_nodes:
            ROW = NODES_LOCATION[node.row]
            branches_cols = [b.branch_col for b in node.finishes_branches + node.generates_branches if b.branch_col is not None]
            if len(branches_cols) == 0:
                branches_cols = [0]
            proposed_col = np.median(branches_cols)
            dists = [abs(i - proposed_col) for i in range(len(ROW))]
            # indices = np.argsort(dists)
            indices = range(len(ROW))
            # find free place
            for col in indices:
                if ROW[col] is FREE or ROW[col] is RESERVED:
                    ROW[col] = OCCUPIED
                    node.col = col
                    break

    @staticmethod
    def make_from_graph_description(gd: GraphDescription) -> 'GraphVisualDescription':
        gvd = GraphVisualDescription()
        gvd.__make_nodes(gd.nodes)
        gvd.__make_branches(gd.branches)
        gvd.__make_splitting_nodes(gd.splitting_nodes)

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
    fig, ax = plt.subplots()

    scale_xy = np.array(scale)
    nodes_pos = {}
    for node in gvd.nodes:
        s_xy = scale_xy * np.array(node.coords)
        t_xy = np.matmul(transformation, np.array(s_xy.tolist() + [1]).reshape(3, 1))
        XY = t_xy.flatten().tolist()
        nodes_pos[node.node_idx] = XY

    # graph connections
    connections = []
    for branch in gvd.branches:
        connections.extend([(n1.node_idx, n2.node_idx) for n1, n2 in zip(branch.nodes[:-1], branch.nodes[1:])])

    # branches plotting
    for branch in gvd.branches:
        XYs = np.array([nodes_pos[node.node_idx] for node in branch.nodes])
        ax.plot(XYs[:,0], XYs[:,1], '-')

    # make a scatter plot of points
    coords = []
    for node in gvd.nodes:
        xy = nodes_pos[node.node_idx]
        coords.append(xy)
    coords = np.array(coords)
    sc = ax.scatter(coords[:,0], coords[:,1])

    # annotation box
    annot = ax.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # on hover event reaction: show proper annotation box
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
    fig.canvas.mpl_connect("motion_notify_event", hover)

    # build a graph and display it
    G = nx.DiGraph(set(connections))
    nx.draw(G, nodes_pos, with_labels=True)
    plt.show(block=True)
