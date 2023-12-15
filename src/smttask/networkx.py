from warnings import warn
from collections.abc import Hashable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import smttask
from . import DataFile
from . import base
from . import task_types
from .config import config

from numbers import Number
PlainArg = (Number, str, np.ndarray)

# This may be spun-off at some point
class PropertyMap(dict):
    """
    Keep a collection of property values|attributes for different keys
    (usu. object types|names)

    Assigns a ParameterSet to each key.
    You can specify a default ParameterSet, for when a key is not found.

    Examples
    --------
    >>> pmap = PropertyMap(Alice={'colour': 'orange'}, Bob={'colour': 'yellow'})
    >>> pmap.default = {'colour': 'blue'}
    >>> pmap['Alice'].colour
    >>> 'orange'
    >>> pmap['Charlie'].colour
    >>> 'blue'
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize as you would a dictionary. The ParameterSet constructor
        is called on each value.
        """
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            self[k] = config.ParameterSet(v)
        self._default = None
    @property
    def default(self):
        return self._default
    @default.setter
    def default(self, attrs):
        """Set the default attribute list"""
        self._default = config.ParameterSet(attrs)
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if self.default is not None:
                return self.default
            else:
                raise

class TaskInput:
    """
    A wrapper for unhashable inputs.
    Can be used for hashable inputs as well.
    """
    def __init__(self, task, name, value):
        self.task = task
        self.name = name
        self.value = value
    def __hash__(self):
        return hash((self.task, repr(self.value)))
    def __str__(self):
        if isinstance(self.value, type):
            return self.value.__qualname__
        return str(self.value)

class TaskGraph(nx.DiGraph):
    display_params = config.ParameterSet({
        'nodetypes': [task_types.RecordedTask, task_types.MemoizedTask,
                      #smttask.StatelessFunction,
                      #smttask.File,
                      DataFile,
                      PlainArg],
        'nodesizes': [1000,        1000,
                      700,
                      700,         700,
                      300],
        'nodecolors': mpl.cm.Pastel1,
        'nodedefaults': {'color': None,
                         'font_color': '#888888',
                         'nodesize': 300},
        'netprops': {'edge_color': '#888888',
                     'edgefont_color': 'k'}
    })
    def __init__(self, stem, depth=-1):
        """
        Parameters
        ----------
        stem: Task
            The task for which we want to graph dependencies.
        depth: int (default: -1)
            How deep to recurse into task dependencies. Negative values
            indicate to recurse indefinitely.
        """
        super().__init__()
        self.stem = stem
        self.add_node(stem)
        self.pos = None  # Place to store node positions
        nx.set_node_attributes(self, {stem: str(stem)}, name='value')
        for name, inp in stem.taskinputs:
            if isinstance(inp, base.Task):
                if depth == 0:
                    node = inp
                    self.add_node(node)
                    nx.set_node_attributes(self, {node: str(node)}, name='value')
                else:
                    node = TaskGraph(inp, depth=depth-1)
                    self.add_node(node)
            else:
                node = inp
                if not isinstance(node, Hashable):
                    node = TaskInput(stem, name, node)
                if node is None:  # networkx doesn't allow None as a node
                    node = "None"
                self.add_node(node)
                nx.set_node_attributes(self, {node: str(node)}, name='value')
                #f not isinstance(inp, smttask.InputTypes)
                #   warn("Input {} is of unrecognized type {}."
                #        .format(inp, type(inp)))
            # self.add_node(node)
            self.add_edge(stem, node)
            nx.set_edge_attributes(self, {(stem, node): name}, name='param name')

    def draw(self, figsize=(12,10), return_fig=False,
             stempos=None, pos=None, ax=None, legend=True):
        """
        A default function for displaying the task graph.

        Parameters
        ----------
        figsize: tuple (default: 12, 10)
            Passed on to `plt.figure(figsize=<figsize>)`.
        return_fig: bool (default: False)
            Set to true if you want `draw` to return the produced figure.
            Default is False because in a typical notebook with
            `%matplotlib inline`, the figure will already be displayed, and
            returning may lead to it being displayed twice.
        stempos: tuple (default: None)
            Position of the stem node. Shifts all others accordingly.
            Ignored if `pos` is provided.
        pos: NetworkX position dict (default: None)
            Override calulated positions. Leave `None` to have them computed
            automatically.
        ax: matplotlib Axes (default: None)
            Axes on which to draw the graph
        legend: bool (default: True)
            Set to False to prevent drawing the legend.

        Returns
        -------
        matplotlib figure (only if `return_fig=True`)
        """
        dp = self.display_params

        colorlst = dp.nodecolors
        if isinstance(colorlst, mpl.colors.Colormap):
            colorlst = [mpl.colors.to_hex(c) for c in colorlst.colors]
        nodeprops = PropertyMap(
            {T: {'color': c, 'size': s, 'font_color': '#888888'}
             for T,s,c in zip(dp.nodetypes, dp.nodesizes, colorlst)})
        nodeprops['others'] = config.ParameterSet(
            {'color': colorlst[len(dp.nodetypes)],
             'font_color': dp.nodedefaults.font_color,
             'size': dp.nodedefaults.nodesize})
        netprops = dp.netprops

        nodelists = {T: [] for T in dp.nodetypes}; nodelists['others'] = []
        for node in self:
            assigned=False
            for T in dp.nodetypes:
                if isinstance(node, T):
                    nodelists[T].append(node)
                    assigned=True
                    continue
            if not assigned:
                nodelists['others'].append(node)

        labels = nx.get_node_attributes(self, 'value')
        edge_labels = nx.get_edge_attributes(self, 'param name')

        # Compute node positions
        if pos is None:
            #pos = nx.bipartite_layout(self, nodes=[node for node in self if isinstance(node, Task)])
            # pos = nx.shell_layout(self)
            # pos = nx.kamada_kawai_layout(self)
            # pos = nx.spring_layout(self)
            # pos = nx.spiral_layout(self)
            # pos = nx.spectral_layout(self)
            pos = nx.circular_layout(self)
            # pos = nx.bipartite_layout(self, nodes=[self.stem], align='horizontal')
            if stempos is not None:
                offset = stempos - pos[self.stem]
                for p in pos.values():
                    p += offset
                assert np.all(np.isclose(pos[self.stem], stempos))
        self.pos = pos

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        else:
            plt.sca(ax)

        for T, nodes in nodelists.items():
            props = nodeprops[T]
            nx.draw_networkx_nodes(nodes, pos, ax=ax, node_color=props.color,
                                   node_size=props.size);
            nx.draw_networkx_labels(nodes, pos, labels, font_size=10, ax=ax,
                                    font_color=props.font_color);
        nx.draw_networkx_edge_labels(self, pos, edge_labels, font_size=10, ax=ax,
                                     font_color=netprops.edgefont_color);
        # TODO: split into target types, and change node_size accordingly
        nx.draw_networkx_edges(self, pos, self.edges, ax=ax,
                               edge_color=netprops.edge_color,
                               node_size=600, arrowsize=30);

        # Recurse and draw nested graphs
        for node in self:
            if isinstance(node, TaskGraph):
                node.draw(stempos=pos[node], ax=ax, legend=False,
                          return_fig=False, figsize=figsize)

        # Create legend
        if legend:
            legendhandles = []
            legendlabels = []
            for T, p in nodeprops.items():
                legendhandles.append(mpl.patches.Circle((0,0), color=p.color, radius=5))
                if T == 'others':
                    legendlabels.append('Unrecognized type')
                elif T is PlainArg:
                    # HACK
                    legendlabels.append('PlainArg')
                else:
                    legendlabels.append(T.__name__)
            # legendhandles.append(mpl.patches.Circle((0,0),
            #                      color=nodeprops['default'].color, radius=5))
            # legendlabels.append('Unrecognized type')
            ax.legend(legendhandles, legendlabels);

        # Axis formatting
        ax.set_axis_off()

        if return_fig:
            return fig
