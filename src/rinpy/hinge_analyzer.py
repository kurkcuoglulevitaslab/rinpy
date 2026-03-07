# -*- coding: utf-8 -*-
"""
    __author__ = 'Zehra Sarica'
    __email__ = ['sarica16@itu.edu.tr','zehraacar559@gmail.com']
"""

import logging
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
from scipy.sparse.linalg import eigsh

from rinpy import log_util
from rinpy import utils
from rinpy.constants import RESIDUE_NUMBER, MODE, CHAIN_ID, INSERTION, HIGH_PERCENTAGE_TEMPLATE, RESIDUE_NAME, \
    CENTRALITY_SCORE, TXT_EXT, PDB_EXT, PML_EXT, CSV_EXT, HTML_EXT, PNG_EXT, X, Y, Z, RESIDUE_INDEX, SEGMENT_ID
from rinpy.log_util import log_details
from rinpy.pymol_utils import PymolUtils
from rinpy.style_config import FONT_STYLES, FONT_FAMILY, EDGE_COLOR, COLOR_PALETTE
from rinpy.utils import CentralityType

_HINGE_MODES_DIR = "hinge_modes"


class HingeAnalyzer:
    """
       Performing graph spectral analysis for the generated network of a protein structure to obtain hinge residues
       of each mode (each eigenvector is referred to as a mode). The eigenvector is obtained from Laplacian matrix.

       Parameters
       ----------
       graph
           The protein structure represented as a network, where nodes are residues
           and edges represent local strength interaction or affinity between residues.
       pdb_name
           The identifier of the PDB file being analyzed.
       destination_output_path
           The path where analysis results, plots, or output files will be saved.
       actual_residue_number_map
           contains defined parameters of a node, such as Chain ID, Residue Number and so on.
    """

    def __init__(self, graph: nx.Graph, pdb_name: str, destination_output_path: str,
                 actual_residue_number_map: dict[int, tuple[str, str, int, str, str, str]]):
        self.graph = graph
        self.pdb_name = pdb_name
        self.destination_output_path = Path(destination_output_path)
        self.actual_residue_number_map = actual_residue_number_map
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.size'] = 12
        self.pymol_utils = PymolUtils(pdb_name=pdb_name,
                                      destination_output_path=destination_output_path,
                                      actual_residue_number_map=actual_residue_number_map)

        utils.create_folder_not_exists(self.destination_output_path / self.pdb_name / _HINGE_MODES_DIR)

    def _compute_laplacian_modes(self, num_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """computes the eigenvalues and eigenvectors of the graph laplacian, then computes spectral nodes
            excluding trivial eigenvector
        """
        nodes = sorted(self.graph.nodes())
        laplacian = nx.laplacian_matrix(self.graph, nodelist=nodes).astype(float)

        if not nx.is_connected(self.graph):
            logging.error("Graph must be connected for spectral partitioning.")
            logging.info("Graph must be connected for spectral partitioning.")
            return

        k = num_modes + 1
        eigen_values, eigen_vectors = eigsh(laplacian, k=k, which="SM")

        idx = np.argsort(eigen_values)
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        fiedler_vector = eigen_vectors[:, 1].real
        spectral_modes = eigen_vectors[:, 1:k]  # Fiedler +  {num_modes-1} more num_modes included

        start = time.time()
        self._plot_fiedler(fiedler_vector=fiedler_vector)
        log_util.log_elapsed_time_detail("plot_fiedler: ", start, time.time())

        return eigen_values, eigen_vectors, fiedler_vector, spectral_modes

    def _is_same_chain(self, current_node_id: int, next_node_id: int) -> bool:
        current_node_data = self.actual_residue_number_map.get(current_node_id)
        next_node_data = self.actual_residue_number_map.get(next_node_id)
        if current_node_data is None or next_node_data is None:
            return False
        return current_node_data[1] == next_node_data[1]  # [1] = chain_id

    def _find_hinge_residues_with_sign(self, mode_data: np.ndarray) -> list[int]:
        node_ids_map = dict(enumerate(sorted(self.graph.nodes())))

        mode_signs = np.sign(mode_data)
        abs_magnitude_diff_signs = np.sign(np.diff(np.abs(mode_data)))

        has_sign_change = np.zeros(len(mode_signs), dtype=bool)

        for i in range(len(mode_signs) - 1):
            if mode_signs[i] != mode_signs[i + 1]:
                if self._is_same_chain(node_ids_map[i], node_ids_map[i + 1]):
                    has_sign_change[i] = True

        for idx, abs_diff_sign in enumerate(abs_magnitude_diff_signs):
            if has_sign_change[idx] and abs_diff_sign < 0:
                has_sign_change[idx] = False
                is_same_chain: bool = self._is_same_chain(node_ids_map[idx], node_ids_map[idx + 1])
                if is_same_chain:
                    has_sign_change[idx + 1] = True

        hinge_nodes: set = {node_ids_map[i] for i, is_hinge in enumerate(has_sign_change) if is_hinge}
        return sorted(hinge_nodes)

    def _save_and_plot_mode_to_file(self, mode_data: np.ndarray, mode_name: str, hinge_residues: list[int]) -> None:
        mode_output_path = os.path.join(self.destination_output_path, self.pdb_name, _HINGE_MODES_DIR)
        utils.create_folder_not_exists(mode_output_path)
        nodes = self.graph.nodes()
        actual_hinge_residues_tuple = self.get_actual_hinge_residues_tuple(nodes=nodes,
                                                                           hinge_residues=hinge_residues)

        full_path = os.path.join(str(mode_output_path), f"{mode_name}_hinge_residues{TXT_EXT}")
        self.write_to_file(actual_hinge_residues=actual_hinge_residues_tuple, full_path=full_path)

        residue_keys = [(nodes[n][RESIDUE_NAME], nodes[n][CHAIN_ID], nodes[n][RESIDUE_NUMBER], nodes[n][INSERTION],
                         nodes[n][SEGMENT_ID]) for
                        n in nodes]

        mode_df = pd.DataFrame({
            RESIDUE_INDEX: residue_keys,
            'mode': mode_data
        })

        residue_to_mode = mode_df.set_index(RESIDUE_INDEX)[MODE].to_dict()

        base_pdb_df = utils.get_base_pdb_df(residue_to=residue_to_mode,
                                            destination_output_path=self.destination_output_path,
                                            pdb_name=self.pdb_name)

        out_filename = os.path.join(str(mode_output_path), f"{mode_name}{PDB_EXT}")
        utils.write_ppdb_to_pdb_file_atom_and_hetatom(ppdb=base_pdb_df, out_filename=out_filename)

        full_path = os.path.join(str(mode_output_path), f"{mode_name}{PML_EXT}")

        high_percentage_residues_df = pd.read_csv(os.path.join(str(self.destination_output_path),
                                                               self.pdb_name,
                                                               f'{self.pdb_name}_{HIGH_PERCENTAGE_TEMPLATE.format(type=CentralityType.BET.display_name)}'),
                                                  sep=";",
                                                  header=None,
                                                  names=[RESIDUE_INDEX, CENTRALITY_SCORE, RESIDUE_NAME, CHAIN_ID,
                                                         RESIDUE_NUMBER, INSERTION, SEGMENT_ID])
        self.pymol_utils.export_pymol_script_hinge(
            full_path_to_pdb=out_filename,
            residues=hinge_residues,
            full_path=full_path,
            high_percentage_residues=list(
                zip(
                    high_percentage_residues_df[RESIDUE_NAME],
                    high_percentage_residues_df[CHAIN_ID],
                    high_percentage_residues_df[RESIDUE_NUMBER],
                    high_percentage_residues_df[INSERTION],
                    high_percentage_residues_df[SEGMENT_ID]
                )
            )
        )

    def _save_selected_eigen_vectors(self, spectral_modes: np.ndarray) -> None:
        filtered_data_df = pd.DataFrame(spectral_modes)
        num_columns = filtered_data_df.shape[1]
        headers = [f"mode{i}" for i in range(1, num_columns + 1)]
        filtered_data_df.columns = headers

        residue_ids = [
            f"{self.actual_residue_number_map[i + 1][2]}"
            if (i + 1) in self.actual_residue_number_map else str(i + 1)
            for i in range(filtered_data_df.shape[0])
        ]

        chain_ids = [
            f"{self.actual_residue_number_map[i + 1][1]}"
            if (i + 1) in self.actual_residue_number_map else str(i + 1)
            for i in range(filtered_data_df.shape[0])
        ]

        residue_names = [
            f"{self.actual_residue_number_map[i + 1][0]}"
            if (i + 1) in self.actual_residue_number_map else str(i + 1)
            for i in range(filtered_data_df.shape[0])
        ]

        insertions = [
            f"{self.actual_residue_number_map[i + 1][3]}"
            if (i + 1) in self.actual_residue_number_map and self.actual_residue_number_map[i + 1][3] else "''"
            for i in range(filtered_data_df.shape[0])
        ]

        segment_ids = [
            f"{self.actual_residue_number_map[i + 1][4]}"
            if (i + 1) in self.actual_residue_number_map and self.actual_residue_number_map[i + 1][4] else "''"
            for i in range(filtered_data_df.shape[0])
        ]

        filtered_data_df.insert(0, "index", range(1, filtered_data_df.shape[0] + 1))
        filtered_data_df.insert(1, RESIDUE_NAME, residue_names)
        filtered_data_df.insert(2, CHAIN_ID, chain_ids)
        filtered_data_df.insert(3, RESIDUE_NUMBER, residue_ids)
        filtered_data_df.insert(4, INSERTION, insertions)
        filtered_data_df.insert(5, SEGMENT_ID, segment_ids)

        filtered_data_path = self.destination_output_path / self.pdb_name / _HINGE_MODES_DIR / f'{self.pdb_name}_eigenvectors{CSV_EXT}'
        filtered_data_df.to_csv(filtered_data_path, index=False)

    def compute_hinge_residues_with_sign(self, num_modes: int = None) -> None:
        if num_modes is None:
            num_modes = 4

        eigen_values, eigen_vectors, fiedler_vector, spectral_modes = self._compute_laplacian_modes(
            num_modes=num_modes)

        self._save_selected_eigen_vectors(spectral_modes=spectral_modes)

        for i in range(spectral_modes.shape[1]):
            mode_data = spectral_modes[:, i]
            hinge_residues = self._find_hinge_residues_with_sign(mode_data=mode_data)
            if hinge_residues and len(hinge_residues) > 0:
                mode_data = np.where(mode_data < 0, -1, 1)
                mode_name: str = f'{self.pdb_name}_laplacian_mode_{(i + 1)}'
                self._save_and_plot_mode_to_file(mode_data=mode_data,
                                                 mode_name=mode_name,
                                                 hinge_residues=hinge_residues)

                node_ids = sorted(list(self.graph.nodes()))
                cluster_labels = {node_ids[i]: int(mode_data[i]) for i in range(len(node_ids))}
                full_path = os.path.join(self.destination_output_path, self.pdb_name, _HINGE_MODES_DIR,
                                         f'{mode_name}_hinge_interactive_clusters_3d{HTML_EXT}')
                self.plot_graph_interactive_clusters_3d(self.graph, cluster_labels, hinge_residues, full_path)
                start = time.time()

                full_path = os.path.join(self.destination_output_path, self.pdb_name, _HINGE_MODES_DIR,
                                         f"{mode_name}_graph_clusters_2d{PNG_EXT}")
                self.plot_graph_clusters_2d(self.graph, cluster_labels, hinge_residues, full_path=full_path)
                log_util.log_elapsed_time_detail("plot_graph_clusters_2d: ", start, time.time())

    @staticmethod
    def get_actual_hinge_residues_tuple(nodes, hinge_residues: list[int]) -> list[str]:
        return [(
            f"{nodes[hr][RESIDUE_NAME]};{nodes[hr][CHAIN_ID]};{nodes[hr][RESIDUE_NUMBER]};'{nodes[hr][INSERTION]}';'{nodes[hr][SEGMENT_ID]}'")
            for hr in hinge_residues
        ]

    def _get_full_save_path(self, filename: str, extension: str = "png") -> str:
        return os.path.join(self.destination_output_path, self.pdb_name, f"{self.pdb_name}_{filename}.{extension}")

    def write_to_file(self, actual_hinge_residues, full_path=None):
        if full_path is None:
            full_path = self._get_full_save_path(filename="hinge_residues", extension="txt")

        with open(full_path, 'w') as f:
            for i, hinge_residue in enumerate(actual_hinge_residues):
                if i < len(actual_hinge_residues) - 1:
                    f.write(f"{hinge_residue}\n")
                else:
                    f.write(f"{hinge_residue}")

    @log_details("Plotting Fiedler vector")
    def _plot_fiedler(self, fiedler_vector: np.ndarray, sort: bool = False) -> None:
        """ Plots and optionally saves the Fiedler vector.

        Parameters:
        - fiedler_vector (np.ndarray): The Fiedler vector to plot.
        - save_path (str, optional): Path to save the figure (without extension). If None, just shows the plot.
        - sort (bool): Whether to plot the sorted Fiedler vector (for smoother visualization).
        - file_format (str): Image format for saving (e.g., 'png', 'pdf').
        - dpi (int): Resolution of the saved image.
        """
        full_path = self._get_full_save_path(filename="fiedler_vector")
        vector = fiedler_vector.copy()
        if sort:
            vector = np.sort(vector)

        fig, ax = plt.subplots(figsize=(10, 6))

        next_round = int(np.ceil(len(vector) / 100.0)) * 100
        ax.set_xlim(0, next_round)

        residue_labels = [
            f"{i + 1} ({self.actual_residue_number_map[i + 1][1]}, {utils.get_residue_id_by_tuple(self.actual_residue_number_map[i + 1])})"
            for i in range(len(vector))
            if (i + 1) in self.actual_residue_number_map
        ]
        x_ticks = ax.get_xticks().astype(int)
        x_ticks = x_ticks[x_ticks < len(vector)]

        if (len(vector) - 1) not in x_ticks:
            x_ticks = np.append(x_ticks, len(vector) - 1)

        ax.plot(vector, marker='o', linestyle='-', color='blue', label='Fiedler Vector')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
        ax.set_title(f"Fiedler Vector - {self.pdb_name}", fontdict=FONT_STYLES["title"])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{residue_labels[i]}" for i in x_ticks], rotation=45, **{
            "fontsize": FONT_STYLES["xtick"]["labelsize"],
            "fontfamily": FONT_STYLES["xtick"]["fontfamily"]
        })
        ax.set_xlabel("Residue Index (Chain ID, Residue Number)", fontdict=FONT_STYLES["xlabel"])

        for label in ax.get_yticklabels():
            label.set_fontsize(FONT_STYLES["ytick"]["labelsize"])
            label.set_fontfamily(FONT_STYLES["ytick"]["fontfamily"])

        ax.set_ylabel("Value", fontdict=FONT_STYLES["ylabel"])
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(prop={"family": FONT_FAMILY, "size": FONT_STYLES["legend"]["fontsize"]})

        next_round = int(np.ceil(len(vector) / 100.0)) * 100
        ax.set_xlim(0, next_round)

        fig.tight_layout()

        fig.savefig(full_path, dpi=300)
        logging.info(f"Saving Fiedler vector to: {os.path.abspath(full_path)}")
        plt.close(fig)

    def plot_graph_clusters_2d(self, graph, cluster_labels, hinge_residues, full_path=None):
        if full_path is None:
            full_path = self._get_full_save_path(filename="graph_clusters_2d")
        unique_clusters = sorted(set(cluster_labels.values()))
        color_map = {cluster_id: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, cluster_id in
                     enumerate(unique_clusters)}

        fig, ax = plt.subplots()

        for cluster_id in unique_clusters:
            cluster_nodes = [n for n in graph.nodes() if cluster_labels[n] == cluster_id]
            xs = [graph.nodes[n]['x'] for n in cluster_nodes]
            ys = [graph.nodes[n]['y'] for n in cluster_nodes]
            ax.scatter(xs, ys, s=100, label=f"Cluster ({cluster_id})", color=color_map[cluster_id],
                       edgecolor=color_map[cluster_id], linewidth=1, alpha=0.6)

            for n in cluster_nodes:
                x, y = graph.nodes[n]['x'], graph.nodes[n]['y']
                ax.text(x, y, str(n), fontsize=6, ha='center', va='center', color='black', alpha=0.6)

        for u, v in graph.edges():
            x = [graph.nodes[u]['x'], graph.nodes[v]['x']]
            y = [graph.nodes[u]['y'], graph.nodes[v]['y']]
            ax.plot(x, y, color=EDGE_COLOR, alpha=0.5, linewidth=0.5)

        hinge_x = [graph.nodes[n]['x'] for n in hinge_residues]
        hinge_y = [graph.nodes[n]['y'] for n in hinge_residues]
        ax.scatter(hinge_x, hinge_y, s=120, c='yellow', edgecolors='black', marker='*',
                   label='Hinge Residues', zorder=10)

        for n in hinge_residues:
            x, y = graph.nodes[n]['x'], graph.nodes[n]['y']
            ax.text(x, y, str(n), fontsize=7, ha='center', va='center', color='black', fontweight='bold')

        ax.set_xlabel("X Coordinate", fontdict=FONT_STYLES["xlabel"])
        ax.set_ylabel("Y Coordinate", fontdict=FONT_STYLES["ylabel"])
        ax.set_title("2D Spectral Clustering of Residue Interaction Network", fontdict=FONT_STYLES["title"])
        ax.legend(prop={"family": FONT_FAMILY, "size": FONT_STYLES["legend"]["fontsize"]})
        ax.axis('equal')
        ax.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        self.update_tick_label_style(ax.get_xticklabels())
        self.update_tick_label_style(ax.get_yticklabels())

        fig.tight_layout()
        fig.savefig(full_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def update_tick_label_style(labels):
        for label in labels:
            label.set_fontsize(FONT_STYLES["ytick"]["labelsize"])
            label.set_fontfamily(FONT_STYLES["ytick"]["fontfamily"])

    def plot_graph_clusters_3d(self, graph, cluster_labels, hinge_residues):
        """ 3D plot (X-Y-Z) with manually assigned high-contrast colors."""
        full_path = self._get_full_save_path(filename="graph_clusters_3d")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        unique_clusters = sorted(set(cluster_labels.values()))
        color_map = {cluster_id: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, cluster_id in
                     enumerate(unique_clusters)}

        legend_handles = []

        for cluster_id in unique_clusters:
            cluster_nodes = [n for n in graph.nodes() if cluster_labels[n] == cluster_id]
            xs = [graph.nodes[n]['x'] for n in cluster_nodes]
            ys = [graph.nodes[n]['y'] for n in cluster_nodes]
            zs = [graph.nodes[n]['z'] for n in cluster_nodes]
            scatter = ax.scatter(xs, ys, zs, s=100, label=f"Cluster ({cluster_id})", color=color_map[cluster_id])
            legend_handles.append(scatter)

        for u, v in graph.edges():
            x = [graph.nodes[u][X], graph.nodes[v][X]]
            y = [graph.nodes[u][Y], graph.nodes[v][Y]]
            z = [graph.nodes[u][Z], graph.nodes[v][Z]]
            ax.plot(x, y, z, color=EDGE_COLOR, alpha=0.5, linewidth=0.5)

        hinge_scatter = []
        for n in hinge_residues:
            if n in graph.nodes():
                x, y, z = graph.nodes[n][X], graph.nodes[n][Y], graph.nodes[n][Z]
                ax.scatter(x, y, z, s=150, c='yellow', edgecolors='black', marker='*')
                ax.text(x, y, z, str(n), fontsize=7, ha='center', va='center', color='black', fontweight='bold')
                hinge_scatter.append(ax.scatter([], [], s=150, c='yellow', edgecolors='black', marker='*'))

        legend_handles.append(hinge_scatter[0])
        ax.legend(handles=legend_handles,
                  labels=[f"Cluster ({i})" for i in unique_clusters] + ["Hinge Residues"],
                  prop={"family": FONT_FAMILY, "size": FONT_STYLES["legend"]["fontsize"]}, loc='upper right',
                  bbox_to_anchor=(1.2, 1.01))

        ax.set_xlabel("X Coordinate", fontdict=FONT_STYLES["xlabel"])
        ax.set_ylabel("Y Coordinate", fontdict=FONT_STYLES["ylabel"])
        ax.set_zlabel("Z Coordinate", fontdict=FONT_STYLES["zlabel"])
        plt.title("3D Spectral Clustering of Residue Interaction Network", fontdict=FONT_STYLES["title"])

        self.update_tick_label_style(ax.get_xticklabels())
        self.update_tick_label_style(ax.get_yticklabels())
        self.update_tick_label_style(ax.get_zticklabels())

        plt.tight_layout()
        fig.savefig(full_path, dpi=300)
        plt.close(fig)

    # @staticmethod
    @staticmethod
    def plot_graph_interactive_clusters_3d(graph, cluster_labels, hinge_residues, full_path):
        unique_clusters = sorted(set(cluster_labels.values()))
        color_map = {
            cluster_id: COLOR_PALETTE[1] if cluster_id < 0 else COLOR_PALETTE[0]
            for cluster_id in unique_clusters
        }
        node_traces = []
        for cluster_id in unique_clusters:
            cluster_nodes = [n for n in graph.nodes if cluster_labels[n] == cluster_id]
            x = [graph.nodes[n][X] for n in cluster_nodes]
            y = [graph.nodes[n][Y] for n in cluster_nodes]
            z = [graph.nodes[n][Z] for n in cluster_nodes]

            labels = []
            for n in cluster_nodes:
                node = graph.nodes[n]
                seg_id = node.get(SEGMENT_ID)
                seg_text = f", Segment ID: {seg_id}" if seg_id not in [None, '', "''"] else ''
                label = (
                    f"{node[RESIDUE_NAME]} "
                    f"(Residue Number: {utils.get_residue_id(node)}, Chain: {node[CHAIN_ID]}{seg_text}), "
                    f"Cluster ({cluster_id})"
                )
                labels.append(label)

            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(size=5, color=color_map[cluster_id]),
                text=labels,
                hoverinfo='text',
                name=f'Cluster {cluster_id}',
                hoverlabel=dict(
                    font=dict(color='white')
                )
            )
            node_traces.append(trace)

        edge_x, edge_y, edge_z = [], [], []
        for u, v in graph.edges():
            edge_x += [graph.nodes[u][X], graph.nodes[v][X], None]
            edge_y += [graph.nodes[u][Y], graph.nodes[v][Y], None]
            edge_z += [graph.nodes[u][Z], graph.nodes[v][Z], None]

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='#1f4e79', width=1),
            hoverinfo='none',
            name='Edges'
        )

        hinge_trace = go.Scatter3d(
            x=[graph.nodes[n][X] for n in hinge_residues],
            y=[graph.nodes[n][Y] for n in hinge_residues],
            z=[graph.nodes[n][Z] for n in hinge_residues],
            mode='markers+text',
            marker=dict(size=7, color='yellow', symbol='diamond', line=dict(color='black', width=1)),
            name='Hinge Residues',
            hoverinfo='text',
            hovertext=[
                f"<b>Residue Name: {graph.nodes[n][RESIDUE_NAME]}</b><br>"
                f"<b>Chain: {graph.nodes[n][CHAIN_ID]}</b><br>"
                f"<b>Hinge Residue Number: {utils.get_residue_id(graph.nodes[n])}</b>"
                + (f"<b>Segment ID</b>: {graph.nodes[n][SEGMENT_ID]}<br>" if graph.nodes[n][SEGMENT_ID] not in
                                                                             [None, '', "''"] else "")
                + f"<br>────────────────────<br>"
                  f"x={graph.nodes[n]['x']:.3f}<br>"
                  f"y={graph.nodes[n]['y']:.3f}<br>"
                  f"z={graph.nodes[n]['z']:.3f}"
                for n in hinge_residues],
            hoverlabel=dict(
                bgcolor='yellow',
                font=dict(color='black')
            )
        )

        fig = go.Figure(data=[edge_trace, hinge_trace] + node_traces)

        axis_style = dict(
            title_font=dict(family=FONT_FAMILY, size=18),
            tickfont=dict(family=FONT_FAMILY, size=16)
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X Coordinate", **axis_style),
                yaxis=dict(title="Y Coordinate", **axis_style),
                zaxis=dict(title="Z Coordinate", **axis_style),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title=dict(
                text="Interactive 3D Spectral Clustering of Residue Interaction Network",
                font=dict(family="Times New Roman", size=24),
                x=0.5,
                xanchor="center",
                y=0.95,
                yanchor="top"
            ),
            template='plotly',
            legend=dict(
                font=dict(family=FONT_FAMILY, size=16),
                title=dict(
                    font=dict(family=FONT_FAMILY, size=16)
                )
            ),
            scene_camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            )
        )

        pyo.plot(fig, filename=full_path, auto_open=False)
        logging.info(f"Interactive plot saved to: {full_path}")


def main():
    pass


if __name__ == '__main__':
    main()
