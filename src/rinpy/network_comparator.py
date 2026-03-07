# -*- coding: utf-8 -*-
"""
    __author__ = 'Zehra Sarica'
    __email__ = ['sarica16@itu.edu.tr','zehraacar559@gmail.com']
"""

import argparse
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh

from rinpy import utils, logging_config
from rinpy.communication_path_efficiency import CommunicationPathEfficiency, _RESIDUE_PAIR_TYPE
from rinpy.constants import HIGH_PERCENTAGE_TEMPLATE, ATOM_NUMBER, CENTRALITY_SCORE, RESIDUE_NAME, CHAIN_ID, \
    RESIDUE_NUMBER, INSERTION, SEGMENT_ID, CENTRALITY_CSV_TEMPLATE, X, Y, Z
from rinpy.font_utils import load_fonts_once
from rinpy.log_util import log_time, log_with_stars
from rinpy.style_config import FONT_STYLES, FONT_FAMILY
from rinpy.utils import CentralityType

_NODE_KEY_TYPE = tuple[str, str, int, str, str]  # (residue_name, chain_id, residue_number, insertion, segment_id)
_EDGE_KEY_TYPE = tuple[_NODE_KEY_TYPE, _NODE_KEY_TYPE]
_RESIDUE_KEY_TYPE = tuple[str, int, str, str]  # (chain_id, residue_number, insertion, segment_id)

_RESIDUE_NAME_TARGET = 'residue_name_target'
_RESIDUE_NAME_SOURCE = 'residue_name_source'
_UNCHANGED_LABEL = 'unchanged'
_COUNT = "count"
CENTRALITY_CHANGES_CSV = "centrality_changes.csv"
_VALIDATION_VALUES = [None, '', "''", '""', ""]
_DELTA_MEAN = "delta_mean"
_HTML_3D_SIZE = 18
_WEIGHT = 'weight'
_CHANGE_TYPE = "change_type"
_WEIGHT_CHANGE_PERCENTAGE = "weight_change_percentage"
_WEIGHT_CHANGE = "weight_change"
_TARGET_WEIGHT = "target_weight"
_SOURCE_WEIGHT = "source_weight"
_EDGE_STATUS = "edge_status"
_TARGET_RESIDUE = "target_residue"
_SOURCE_RESIDUE = "source_residue"
_TARGET_NODE_ID = 'target_node_id'
_SOURCE_NODE_ID = 'source_node_id'
_NODE_KEY = 'node_key'
_DELTA = 'delta'
_SCORE = 'score'
_TARGET_SUFFIX = '_target'
_SOURCE_SUFFIX = '_source'
_RESULT_FOLDER_POSTFIX: str = "compare_results"
_CENTRALITY_SCORE_TYPE: str = "{type}_centrality_score"
_CENTRALITY_DELTA: str = "{type}_centrality_delta"

_CENTRALITY_SCORE_COLUMNS = [ATOM_NUMBER, CENTRALITY_SCORE, RESIDUE_NAME, CHAIN_ID, RESIDUE_NUMBER, INSERTION,
                             SEGMENT_ID]
_CENTRALITY_SCORE_DTYPES = {ATOM_NUMBER: int, CENTRALITY_SCORE: float, RESIDUE_NAME: str, CHAIN_ID: str,
                            RESIDUE_NUMBER: int,
                            INSERTION: str, SEGMENT_ID: str}
_CENTRALITY_SCORE_SORT_KEYS = [ATOM_NUMBER]
_CENTRALITY_SCORE_COMMON_COLUMNS = [RESIDUE_NAME, CHAIN_ID, RESIDUE_NUMBER, INSERTION, SEGMENT_ID]

_ZOOM_CONFIG = {'scrollZoom': True, 'displayModeBar': True}


class ChangeType(Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"

    def __str__(self):
        return self.value


class EdgeStatusType(Enum):
    COMMON = "common"
    ADDED = "added"
    REMOVED = "removed"
    ALL = "all"

    def __str__(self):
        return self.value


_EDGE_WEIGHT_COLOR_MAP = {
    ChangeType.INCREASE: '#b91c1c',
    ChangeType.DECREASE: '#0761e8',
    ChangeType.UNCHANGED: '#9ca3af',
    ChangeType.ADDED: '#5eead4',
    ChangeType.REMOVED: '#e879f9'
}


class NetworkComparator:
    """ NetworkComparator performs comparative analysis between two protein residue interaction networks (RINs).
        It enables structural comparison and shortest path analysis between specified residue pairs across
        source and target protein structures.

        Parameters
        ----------
        source_input_path : Path
            Directory to the source protein network.

        target_input_path : Path
            Directory to the target protein network.

        output_path : Path
            Directory where analysis results will be saved.

        residue_pairs : list of residue pairs, optional
            List of (source_residue, target_residue) pairs
            used for shortest path analysis.

            Each residue is defined as:
                (chain_id, residue_number, insertion_code, segment_id)

            Format:
                [
                    (('A', 12, '', ''), ('A', 144, '', '')),
                    ...
                ]
        """
    _fonts_initialized = False

    def __init__(self, source_input_path: str,
                 target_input_path: str,
                 output_path: str,
                 residue_pairs: list[_RESIDUE_PAIR_TYPE] | None = None):

        if not NetworkComparator._fonts_initialized:
            load_fonts_once()
            NetworkComparator._fonts_initialized = True

        self.source_input_path = Path(source_input_path)
        self.target_input_path = Path(target_input_path)

        self.source_pdb_name = self.source_input_path.name
        self.source_pdb_name_save = f"{self.source_input_path.name}"
        self.target_pdb_name = self.target_input_path.name
        self.target_pdb_name_save = f"{self.target_input_path.name}"

        self.output_path = Path(
            output_path) / f"{self.source_pdb_name_save}_{self.target_pdb_name_save}_{_RESULT_FOLDER_POSTFIX}"
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.residue_pairs = residue_pairs or []

        self.source_network = self._get_network(self._find_graphml(self.source_pdb_name, self.source_input_path))
        self.target_network = self._get_network(self._find_graphml(self.target_pdb_name, self.target_input_path))

        self.source_centrality_df = self._get_centrality_score_df(self.source_pdb_name, self.source_input_path)
        self.target_centrality_df = self._get_centrality_score_df(self.target_pdb_name, self.target_input_path)

        # node map: node_id -> key(chain_id, residue_number, insertion, segment_id) # residue_name not included
        self.source_node_ids_to_key_map = self._build_node_id_to_key(self.source_network, False)
        self.target_node_ids_to_key_map = self._build_node_id_to_key(self.target_network, False)

        # Reverse map: key -> node_id
        self.source_node_keys_to_id_map = {v: k for k, v in self.source_node_ids_to_key_map.items()}
        self.target_node_keys_to_id_map = {v: k for k, v in self.target_node_ids_to_key_map.items()}

        self.common_node_keys, self.common_node_ids = self._build_common_residue_nodes(self.source_node_keys_to_id_map,
                                                                                       self.target_node_keys_to_id_map)

    @log_with_stars("Network Comparator Analyzer")
    @log_time("Network Comparator Analyzer")
    def run(self, num_modes: int = 20) -> None:
        logging_config.clear_logs()

        self._process_quantify_changes(num_modes=num_modes)

        self._process_graph_signal_processing()

        if self.residue_pairs:
            self._process_communication_path_efficiency()

    @log_with_stars("Communication Path Efficiency Analyzer")
    @log_time("Communication Path Efficiency Analyzer")
    def _process_communication_path_efficiency(self) -> None:
        """ Computes communication path efficiency for the given networks"""
        cpe = CommunicationPathEfficiency(source_network=self.source_network,
                                          target_network=self.target_network,
                                          output_path=self.output_path,
                                          residue_pairs=self.residue_pairs,
                                          source_pdb_name=self.source_pdb_name,
                                          target_pdb_name=self.target_pdb_name)
        cpe.compute()

    @log_with_stars("Compute Difference Networks & Quantify Changes")
    @log_time("Compute Difference Networks & Quantify Changes")
    def _process_quantify_changes(self, num_modes: int = 20) -> None:
        """ Computes difference networks between two states and quantify changes in edge weights, centrality measures,
            and community structures.

        Parameters
        ----------
        num_modes : int, default is 20
            specifies the number of non-zero modes used for community structure analysis.
        """
        self._compare_edges()
        self._compare_centrality()
        self._community_structure_analysis(num_modes=num_modes)

    @log_with_stars("Graph Signal Processing")
    @log_time("Graph Signal Processing")
    def _process_graph_signal_processing(self, num_modes: int = 20) -> None:
        """ Performs graph signal processing to analyze how local perturbations propagate
        through the heterogeneous network.

        Parameters
        ----------
        num_modes : int, default is 20
            specifies the number of non-zero modes used for community structure analysis.
        """
        self._compute_graph_signal_processing(num_modes=num_modes)
        self._plot_hub_residues()

    @staticmethod
    def _build_common_residue_nodes(source_keys_to_id: dict[_RESIDUE_KEY_TYPE, int],
                                    target_keys_to_id: dict[_RESIDUE_KEY_TYPE, int]) -> tuple[
        list[_RESIDUE_KEY_TYPE], list[int]]:
        """ builds the common residue node keys and ids. This helps to identify a node either
            using node key or using node id.
        """
        common_keys = set(source_keys_to_id.keys()) & set(target_keys_to_id.keys())
        common_mapping = []
        for key in common_keys:
            common_mapping.append({
                _NODE_KEY: key,  # (chain_id, residue_number, insertion, segment_id)
                _SOURCE_NODE_ID: source_keys_to_id[key],
                _TARGET_NODE_ID: target_keys_to_id[key]
            })
        common_mapping.sort(key=lambda x: (x[_NODE_KEY][0], x[_NODE_KEY][1], x[_NODE_KEY][2], x[_NODE_KEY][3]))

        common_source_ids = [m[_SOURCE_NODE_ID] for m in common_mapping]
        common_target_ids = [m[_TARGET_NODE_ID] for m in common_mapping]

        common_node_keys = [m[_NODE_KEY] for m in common_mapping]
        common_node_ids = list(set(common_source_ids) & set(common_target_ids))

        return common_node_keys, common_node_ids

    @staticmethod
    def _get_network(pdb_network: Path) -> nx.Graph:
        """ load network from the graphml file and return it as a networkx"""
        network = nx.read_graphml(pdb_network)
        node_map = {node: int(node) for node in network.nodes()}
        network = nx.relabel_nodes(network, node_map)
        return network

    @staticmethod
    def _get_high_percentage_score(pdb_name: str, path: Path,
                                   centrality_type: CentralityType = CentralityType.BET) -> pd.DataFrame:
        """ converts the given high percentage file to the dataframe"""
        filename = f"{pdb_name}_{HIGH_PERCENTAGE_TEMPLATE.format(type=centrality_type.display_name)}"
        high_centrality_score_path = next(path.glob(filename), None)
        centrality_score_df = utils.get_df(output_file_path=high_centrality_score_path,
                                           columns=_CENTRALITY_SCORE_COLUMNS,
                                           dtypes=_CENTRALITY_SCORE_DTYPES,
                                           sort_keys=_CENTRALITY_SCORE_SORT_KEYS,
                                           sep=";")

        return centrality_score_df

    @staticmethod
    def _get_centrality_df(pdb_name: str, path: Path,
                           centrality_type: CentralityType = CentralityType.BET) -> pd.DataFrame:
        """ converts centrality scores file to the dataframe"""
        df = utils.get_df(output_file_path=next(
            path.glob(f"{pdb_name}_{CENTRALITY_CSV_TEMPLATE.format(type=centrality_type.display_name)}"), None),
            columns=_CENTRALITY_SCORE_COLUMNS,
            dtypes=_CENTRALITY_SCORE_DTYPES,
            sort_keys=_CENTRALITY_SCORE_SORT_KEYS)
        df = df.rename(columns={CENTRALITY_SCORE: _CENTRALITY_SCORE_TYPE.format(type=centrality_type.display_name)})
        return df

    def _get_centrality_score_df(self, pdb_name: str, path: Path) -> pd.DataFrame:
        """ get centrality scores dataframe based on pdb name and the given path directory.
            And also merge all scores into dataframe.
        """
        bet_df = self._get_centrality_df(pdb_name=pdb_name, path=path, centrality_type=CentralityType.BET)
        clos_df = self._get_centrality_df(pdb_name=pdb_name, path=path, centrality_type=CentralityType.CLOS)
        deg_df = self._get_centrality_df(pdb_name=pdb_name, path=path, centrality_type=CentralityType.DEG)

        df = bet_df[[ATOM_NUMBER, _CENTRALITY_SCORE_TYPE.format(
            type=CentralityType.BET.display_name)] + _CENTRALITY_SCORE_COMMON_COLUMNS].merge(
            clos_df[[ATOM_NUMBER, _CENTRALITY_SCORE_TYPE.format(type=CentralityType.CLOS.display_name)]],
            on=ATOM_NUMBER).merge(
            deg_df[[ATOM_NUMBER, _CENTRALITY_SCORE_TYPE.format(type=CentralityType.DEG.display_name)]], on=ATOM_NUMBER)

        df = df[[ATOM_NUMBER, _CENTRALITY_SCORE_TYPE.format(type=CentralityType.BET.display_name),
                 _CENTRALITY_SCORE_TYPE.format(type=CentralityType.CLOS.display_name),
                 _CENTRALITY_SCORE_TYPE.format(type=CentralityType.DEG.display_name),
                 RESIDUE_NAME, CHAIN_ID, RESIDUE_NUMBER, INSERTION, SEGMENT_ID]]

        return df

    @staticmethod
    def _find_graphml(pdb_name: str, path: Path) -> Optional[Path]:
        return next(path.glob(f"{pdb_name}_*_network.graphml"), None)

    @staticmethod
    def format_residue_label(key: _NODE_KEY_TYPE) -> str:
        """Format residue key as human-readable label."""
        residue_name, chain_id, residue_number, insertion, segment_id = key
        res_num_insertion = f"{residue_number}{insertion}" if insertion not in _VALIDATION_VALUES else residue_number
        parts = [residue_name, chain_id, str(res_num_insertion)]
        if segment_id and segment_id not in _VALIDATION_VALUES:
            parts.append(segment_id)
        return ", ".join(parts)

    @staticmethod
    def _build_node_id_to_key(network: nx.Graph, include_res_name: bool = True) -> dict:
        """Node ID -> unique residue key tuple mapping."""
        id_to_key = {}
        for node_id, data in network.nodes(data=True):
            residue_name = str(data.get(RESIDUE_NAME))
            chain_id = str(data.get(CHAIN_ID))
            residue_number = int(data.get(RESIDUE_NUMBER))
            insertion = data.get(INSERTION, '')
            if str(insertion) in _VALIDATION_VALUES:
                insertion = ''
            segment_id = data.get(SEGMENT_ID, '')
            if str(segment_id) in _VALIDATION_VALUES:
                segment_id = ''

            if include_res_name:
                id_to_key[node_id] = (residue_name, chain_id, residue_number, str(insertion), str(segment_id))
            else:
                id_to_key[node_id] = (chain_id, residue_number, str(insertion), str(segment_id))

        return id_to_key

    @staticmethod
    def _sort_edge_keys(key_u, key_v):
        """sorts keys based on chain_id, residue_number, insertion, segment_id in order."""

        def sort_key(k):
            residue_name, chain_id, residue_number, insertion, segment_id = k
            return chain_id, residue_number, insertion, segment_id

        return tuple(sorted([key_u, key_v], key=sort_key))

    def _edges_to_residue_keys(self, network: nx.Graph, id_to_key: dict[int, _NODE_KEY_TYPE]) -> set:
        """ it converts sorted residue key tuple sets from network edges. """
        residue_edges = set()
        for u, v in set(network.edges()):
            key_u = id_to_key.get(u)
            key_v = id_to_key.get(v)
            if key_u is None or key_v is None:
                logging.info(f"Warning: node key not found for edge ({u}, {v}), skipping...")
                continue
            residue_edges.add(self._sort_edge_keys(key_u, key_v))
        return residue_edges

    @staticmethod
    def _merge_names(names: set[str]) -> str:
        # deterministic "CYS/GLY"
        return "/".join(sorted(names))

    def _build_merge_map(self, source_edges, target_edges) -> dict[_RESIDUE_KEY_TYPE, str]:
        key_to_names: dict[_RESIDUE_KEY_TYPE, set[str]] = {}

        def _collect(edges) -> None:
            for u, v in edges:
                for node_key in (u, v):
                    key: _RESIDUE_KEY_TYPE = (node_key[1], node_key[2], node_key[3], node_key[4])
                    key_to_names.setdefault(key, set()).add(node_key[0])  # residue_name

        _collect(source_edges)
        _collect(target_edges)

        return {k: self._merge_names(v) for k, v in key_to_names.items()}

    @staticmethod
    def _rewrite_edges(edges, merge_map: dict[_RESIDUE_KEY_TYPE, str]) -> set[_EDGE_KEY_TYPE]:

        out: set[_EDGE_KEY_TYPE] = set()

        for u, v in edges:
            def rewrite(n: _NODE_KEY_TYPE) -> _NODE_KEY_TYPE:
                residue_name, chain_id, residue_number, insertion, segment_id = n
                merged_residue_name = merge_map.get((chain_id, residue_number, insertion, segment_id), residue_name)
                return merged_residue_name, chain_id, residue_number, insertion, segment_id

            uu = rewrite(u)
            vv = rewrite(v)
            out.add((uu, vv))

        return out

    @staticmethod
    def _format_common_edge_residue(node_key: _NODE_KEY_TYPE) -> str:
        residue_name, chain_id, residue_number, insertion, segment_id = node_key
        res_part = f"{residue_number}{insertion}" if insertion else f"{residue_number}"
        base = f"{residue_name}, {chain_id}, {res_part}"
        if segment_id:
            return f"{base},{segment_id}"
        return base

    @log_time("Compare Edges")
    def _compare_edges(self) -> pd.DataFrame:
        """ compares the edges between source and target networks.
            It plots the distribution in 2D and 3D format.
        """
        source_id_to_key_with_res_name = self._build_node_id_to_key(self.source_network)
        target_id_to_key_with_res_name = self._build_node_id_to_key(self.target_network)

        source_id_to_key = self._build_node_id_to_key(self.source_network, False)
        target_id_to_key = self._build_node_id_to_key(self.target_network, False)

        # Reverse map: key -> node_id
        source_key_to_id = {v: k for k, v in source_id_to_key.items()}
        target_key_to_id = {v: k for k, v in target_id_to_key.items()}

        source_edges_keys = self._edges_to_residue_keys(self.source_network, source_id_to_key_with_res_name)
        target_edges_keys = self._edges_to_residue_keys(self.target_network, target_id_to_key_with_res_name)

        merge_map = self._build_merge_map(source_edges_keys, target_edges_keys)

        source_edge_pairs = self._rewrite_edges(source_edges_keys, merge_map)
        target_edge_pairs = self._rewrite_edges(target_edges_keys, merge_map)

        common_edges = source_edge_pairs & target_edge_pairs
        added_edges = target_edge_pairs - source_edge_pairs
        removed_edges = source_edge_pairs - target_edge_pairs

        rows = []

        for key_u, key_v in common_edges:
            u_key = key_u[1:]
            v_key = key_v[1:]

            src_u_id = source_key_to_id[u_key]
            src_v_id = source_key_to_id[v_key]

            tgt_u_id = target_key_to_id[u_key]
            tgt_v_id = target_key_to_id[v_key]

            source_weight = self.source_network[src_u_id][src_v_id].get(_WEIGHT, 1.0)
            target_weight = self.target_network[tgt_u_id][tgt_v_id].get(_WEIGHT, 1.0)

            weight_change = target_weight - source_weight

            weight_change_percentage = (
                    (target_weight - source_weight) / source_weight * 100) if source_weight != 0 else 0

            if weight_change > 0:
                change_type = ChangeType.INCREASE
            elif weight_change < 0:
                change_type = ChangeType.DECREASE
            else:
                change_type = ChangeType.UNCHANGED

            rows.append([
                self._format_common_edge_residue(key_u),
                self._format_common_edge_residue(key_v),
                EdgeStatusType.COMMON,
                source_weight,
                target_weight,
                weight_change,
                weight_change_percentage,
                change_type
            ])

        for key_u, key_v in added_edges:
            tgt_u_id = target_key_to_id[key_u[1:]]
            tgt_v_id = target_key_to_id[key_v[1:]]
            target_weight = self.target_network[tgt_u_id][tgt_v_id].get(_WEIGHT, 1.0)

            rows.append([
                self._format_common_edge_residue(key_u),
                self._format_common_edge_residue(key_v),
                EdgeStatusType.ADDED,
                0.0,
                target_weight,
                target_weight,
                float('inf'),
                ChangeType.ADDED
            ])

        for key_u, key_v in removed_edges:
            src_u_id = source_key_to_id[key_u[1:]]
            src_v_id = source_key_to_id[key_v[1:]]
            source_weight = self.source_network[src_u_id][src_v_id].get(_WEIGHT, 1.0)

            rows.append([
                self._format_common_edge_residue(key_u),
                self._format_common_edge_residue(key_v),
                EdgeStatusType.REMOVED,
                source_weight,
                0.0,
                -source_weight,
                float('-inf'),
                ChangeType.REMOVED
            ])

        _EDGE_WEIGHT_COLUMNS = [_SOURCE_RESIDUE, _TARGET_RESIDUE, _EDGE_STATUS, _SOURCE_WEIGHT, _TARGET_WEIGHT,
                                _WEIGHT_CHANGE, _WEIGHT_CHANGE_PERCENTAGE, _CHANGE_TYPE]

        df = pd.DataFrame(rows, columns=_EDGE_WEIGHT_COLUMNS)

        logging.info("Creating edge weight distribution 2D plot...")
        self._visualize_edge_weights_distribution_2d(df=df)

        logging.info("Creating edge weight distribution 3D plot...")
        self._visualize_edge_weights_distribution_3d(df=df)

        return df

    def _visualize_edge_weights_distribution_2d(self, df: pd.DataFrame = None) -> None:
        """ Visualize edge weight distributions and changes """

        if df is None:
            logging.info("Given dataframe is None.")
            return

        df_processed = df.copy()

        if df_processed.empty:
            logging.info("No edges found.")
            return

        plt.figure(figsize=(12, 8))

        for change_type, color in _EDGE_WEIGHT_COLOR_MAP.items():
            mask = df_processed[_CHANGE_TYPE] == change_type
            plt.scatter(df_processed.loc[mask, _SOURCE_WEIGHT],
                        df_processed.loc[mask, _TARGET_WEIGHT],
                        alpha=0.55, s=22, c=color, label=change_type.value.capitalize(),
                        edgecolors='none')

        max_val = max(df_processed[_SOURCE_WEIGHT].max(), df_processed[_TARGET_WEIGHT].max())
        plt.plot([0, max_val], [0, max_val], '--', color='#4b5563', linewidth=1.5, label='No change')

        padding = max_val * 0.02
        plt.xlim(0 - padding, max_val + padding)
        plt.ylim(0 - padding, max_val + padding)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel(f'Edge weight ({self.source_pdb_name})', fontsize=FONT_STYLES['xlabel']['fontsize'])
        plt.ylabel(f'Edge weight ({self.target_pdb_name})', fontsize=FONT_STYLES['ylabel']['fontsize'])
        plt.title(f'{self.source_pdb_name} vs {self.target_pdb_name} Edge Weights',
                  fontsize=FONT_STYLES['title']['fontsize'])
        plt.legend(fontsize=FONT_STYLES['legend']['fontsize'])
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_file = self.output_path / 'edge_weight_distributions_2d.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {output_file}")

    def _visualize_edge_weights_distribution_3d(self, df=None) -> None:
        """ Interactive Plotly visualization of edge weight changes in a 3D format. """
        if df is None:
            logging.info("Given dataframe is None.")
            return
        df_processed = df.copy()

        if df_processed.empty:
            logging.info("No common edges found.")
            return

        fig = go.Figure()

        for change_type, color in _EDGE_WEIGHT_COLOR_MAP.items():
            subset = df_processed[df_processed[_CHANGE_TYPE] == change_type]
            if subset.empty:
                continue

            fig.add_trace(go.Scatter(
                x=subset[_SOURCE_WEIGHT],
                y=subset[_TARGET_WEIGHT],
                mode='markers',
                marker=dict(size=7, color=color, opacity=0.7),
                name=change_type.value.capitalize(),
                text=[
                    f"{self._format_centrality_node_label(row[_SOURCE_RESIDUE])} ↔ "
                    f"{self._format_centrality_node_label(row[_TARGET_RESIDUE])}<br>"
                    f"{self.source_pdb_name}: {row[_SOURCE_WEIGHT]:.3f}<br>"
                    f"{self.target_pdb_name}: {row[_TARGET_WEIGHT]:.3f}<br>"
                    f"Δ: {row[_WEIGHT_CHANGE]:.3f}"
                    for _, row in subset.iterrows()],
                hovertemplate='%{text}<extra></extra>'
            ))

        # Diagonal line
        max_val = max(df_processed[_SOURCE_WEIGHT].max(), df_processed[_TARGET_WEIGHT].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(color='#4b5563', dash='dash', width=1.5),
            name='No change',
            showlegend=False
        ))

        fig.update_layout(
            title=dict(text=f'{self.source_pdb_name} vs {self.target_pdb_name} Edge Weights', x=0.5, xanchor='center'),
            xaxis_title=f'Edge weight ({self.source_pdb_name})',
            yaxis_title=f'Edge weight ({self.target_pdb_name})',
            height=900,
            width=900,
            showlegend=True,
            legend=dict(
                x=1.12,
                y=0.90,
                xanchor="right",
                yanchor="top"
            ),
            xaxis=dict(
                domain=[0.08, 0.88]
            ),
            yaxis=dict(
                domain=[0.08, 0.92]
            ),
            template="simple_white",
            font=dict(size=18, family=FONT_FAMILY, color="black"),
            margin=dict(l=120, r=120, t=100, b=120),
            hovermode="closest"
        )

        output_file = self.output_path / 'edge_weight_distributions_3d.html'
        fig.write_html(output_file, config=_ZOOM_CONFIG)
        logging.info(f"Saved: {output_file}")

    @staticmethod
    def _format_centrality_node_label(node: str, splitter: str = ",") -> str:
        parts = [p.strip() for p in str(node).split(splitter)]
        if len(parts) < 3:
            return str(node).strip()
        residue_name = "/".join(part.capitalize() for part in parts[0].split("/"))
        chain_id = parts[1]
        residue_number = parts[2]
        segment_id = parts[3] if len(parts) >= 4 and parts[3] else None
        label = f"{chain_id}: {residue_name}{residue_number}"
        if segment_id:
            label += f" ({segment_id})"
        return label

    @staticmethod
    def _get_contrast_color(rgba):
        r, g, b = rgba[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.6 else "white"

    def _plot_centrality_score_comparison_3d(self, centrality_type=CentralityType.BET, output_name=None,
                                             df=None) -> None:

        if output_name is None:
            output_name = f"centrality_{centrality_type.display_name}_comparison_3d.html"

        source_column = f"source_{centrality_type.display_name}"
        target_column = f"target_{centrality_type.display_name}"
        delta_column = f"delta_{centrality_type.display_name}"

        gx, gy = source_column, target_column

        df_processed = df.copy()

        grouped_df = (
            df_processed.groupby([gx, gy], dropna=False)
            .agg(
                nodes=("node", lambda s: list(s)),
                count=("node", "size"),
                delta_mean=(delta_column, "mean")
            )
            .reset_index()
        )

        grouped_df["nodes_html"] = grouped_df["nodes"].apply(lambda xs: "<br>".join(x for x in xs))

        fig = go.Figure()

        cmap = plt.cm.RdBu_r
        norm = mcolors.Normalize(vmin=grouped_df[_DELTA_MEAN].min(),
                                 vmax=grouped_df[_DELTA_MEAN].max())

        rgba_colors = cmap(norm(grouped_df[_DELTA_MEAN].values))

        text_colors = [self._get_contrast_color(rgba) for rgba in rgba_colors]

        max_val = max(df_processed[source_column].max(), df_processed[target_column].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray", width=1.5),
            name="No Change"
        ))

        fig.add_trace(go.Scatter(
            x=grouped_df[gx],
            y=grouped_df[gy],
            mode="markers+text",
            text=np.where(grouped_df[_COUNT] > 1, grouped_df[_COUNT].astype(str), ""),
            textposition="middle center",
            textfont=dict(size=12, color=text_colors),
            marker=dict(
                size=np.clip(10 + 3 * grouped_df[_COUNT], 10, 34),
                color=grouped_df[_DELTA_MEAN],
                colorscale="RdBu_r",
                line=dict(width=0.7, color="black"),
                colorbar=dict(
                    title="",
                    thickness=20,
                    len=0.75,
                    x=1.02,
                    ticks="outside",
                    tickwidth=2,
                    ticklen=6,
                    tickcolor="gray",
                    tickfont=dict(size=16, color="black", family=FONT_FAMILY)
                ),
                opacity=0.8
            ),
            hovertemplate="<b>Residues</b><br>%{customdata[0]}<br><br>" +
                          f"{self.source_pdb_name}: %{{x:.6f}}<br>" +
                          f"{self.target_pdb_name}: %{{y:.6f}}<br>" +
                          "count: %{customdata[1]}<br>" +
                          f"Δ: %{{marker.color:.4f}}" +
                          "<extra></extra>",
            customdata=np.stack([
                grouped_df["nodes_html"],
                grouped_df[_COUNT],
                grouped_df[_DELTA_MEAN]
            ], axis=1),
            name="Residues"
        ))

        fig.add_annotation(
            x=1.16,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Δ {centrality_type.display_name.capitalize()}",
            textangle=90,
            showarrow=False,
            font=dict(size=20, family=FONT_FAMILY, color="black"),
            align="center"
        )

        fig.update_layout(
            width=1000,
            height=900,
            title=dict(
                text=f"{centrality_type.display_name.capitalize()} Centrality Comparison",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title=f"{centrality_type.display_name.capitalize()} ({self.source_pdb_name})",
                domain=[0.08, 0.88]
            ),
            yaxis=dict(
                title=f"{centrality_type.display_name.capitalize()} ({self.target_pdb_name})",
                domain=[0.08, 0.92]
            ),
            template="simple_white",
            font=dict(size=17, family=FONT_FAMILY),
            margin=dict(l=120, r=120, t=100, b=120),
            hovermode="closest"
        )

        fig.write_html(os.path.join(self.output_path, output_name), config=_ZOOM_CONFIG)

    def _plot_centrality_score_comparison_2d(self, centrality_type=CentralityType.BET, output_name=None,
                                             df=None) -> None:

        if output_name is None:
            output_name = f"centrality_{centrality_type.display_name}_comparison_2d.png"

        source_column = f"source_{centrality_type.display_name}"
        target_column = f"target_{centrality_type.display_name}"
        delta_column = f"delta_{centrality_type.display_name}"

        df_processed = df.copy()
        s_g = f"_source_{centrality_type.display_name}"
        t_g = f"_target_{centrality_type.display_name}"
        df_processed[s_g] = df_processed[source_column]
        df_processed[t_g] = df_processed[target_column]

        grouped_df = (
            df_processed.groupby([s_g, t_g], dropna=False)
            .agg(
                delta_mean=(delta_column, "mean"),
                count=("node", "size"),
                nodes=("node", lambda s: list(s))
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        v_min, v_max = grouped_df[_DELTA_MEAN].min(), grouped_df[_DELTA_MEAN].max()
        norm = plt.Normalize(vmin=v_min, vmax=v_max)
        cmap = plt.cm.RdBu_r

        rgba_colors = cmap(norm(grouped_df[_DELTA_MEAN].values))
        text_colors = [self._get_contrast_color(rgba) for rgba in rgba_colors]
        sizes = np.clip(30 + 15 * grouped_df[_COUNT], 40, 260)
        ax.scatter(
            grouped_df[s_g], grouped_df[t_g],
            c=grouped_df[_DELTA_MEAN],
            s=sizes,
            cmap=cmap,
            norm=norm,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.6,
            zorder=2
        )

        for x, y, cnt, text_color in zip(grouped_df[s_g], grouped_df[t_g], grouped_df[_COUNT], text_colors):
            if cnt > 1:
                ax.text(x, y, str(cnt), ha="center", va="center", fontsize=8, color=text_color, zorder=10)

        max_val = max(df_processed[source_column].max(), df_processed[target_column].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label="No Change")

        ax.set_xlabel(f'{centrality_type.display_name.capitalize()} Centrality ({self.source_pdb_name})',
                      fontdict=FONT_STYLES["xlabel"])
        ax.set_ylabel(f'{centrality_type.display_name.capitalize()} Centrality ({self.target_pdb_name})',
                      fontdict=FONT_STYLES["ylabel"])
        ax.tick_params(axis='both', labelsize=FONT_STYLES['xtick_medium']['labelsize'])
        ax.set_title(f'{centrality_type.display_name.capitalize()} Centrality Comparison',
                     fontdict=FONT_STYLES["title"])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'Δ {centrality_type.display_name.capitalize()} (mean)', rotation=270, labelpad=20,
                       fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, output_name), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_delta_histogram(self, data: np.ndarray, x_label: str, filename: str):
        """Plot histogram distribution."""
        plt.figure()
        plt.hist(data, bins=40, edgecolor='black', alpha=0.7)
        plt.xlabel(x_label)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()

    @log_time("Compare Centrality Scores")
    def _compare_centrality(self):
        merge_keys = [CHAIN_ID, RESIDUE_NUMBER, INSERTION, SEGMENT_ID]
        score_columns = [_CENTRALITY_SCORE_TYPE.format(type=CentralityType.BET.display_name),
                         _CENTRALITY_SCORE_TYPE.format(type=CentralityType.CLOS.display_name),
                         _CENTRALITY_SCORE_TYPE.format(type=CentralityType.DEG.display_name)]

        df = self.source_centrality_df.merge(self.target_centrality_df, on=merge_keys,
                                             suffixes=(_SOURCE_SUFFIX, _TARGET_SUFFIX))

        for column in score_columns:
            df[column.replace(_SCORE, _DELTA)] = df[f'{column}{_TARGET_SUFFIX}'] - df[f'{column}{_SOURCE_SUFFIX}']

        header_columns = ["node", "source_betweenness", "target_betweenness", "delta_betweenness", "source_closeness",
                          "target_closeness", "delta_closeness", "source_degree", "target_degree", "delta_degree"]

        rows = []

        for _, row in df.iterrows():
            res_name = (
                f"{row[_RESIDUE_NAME_SOURCE]}/{row[_RESIDUE_NAME_TARGET]}"
                if row[_RESIDUE_NAME_SOURCE] != row[_RESIDUE_NAME_TARGET]
                else row[_RESIDUE_NAME_SOURCE]
            )
            node_label = self._format_centrality_node_label(self.format_residue_label(
                (
                    res_name,
                    row[CHAIN_ID],
                    row[RESIDUE_NUMBER],
                    "" if pd.isna(row[INSERTION]) or row[INSERTION] in _VALIDATION_VALUES else str(row[INSERTION]),
                    "" if pd.isna(row[SEGMENT_ID]) or row[SEGMENT_ID] in _VALIDATION_VALUES else str(row[SEGMENT_ID]),
                )
            ))

            rows.append([
                node_label,
                row["betweenness_centrality_score_source"],
                row["betweenness_centrality_score_target"],
                row["betweenness_centrality_delta"],
                row["closeness_centrality_score_source"],
                row["closeness_centrality_score_target"],
                row["closeness_centrality_delta"],
                row["degree_centrality_score_source"],
                row["degree_centrality_score_target"],
                row["degree_centrality_delta"]
            ])

        delta_columns = {
            _CENTRALITY_DELTA.format(type=CentralityType.BET.display_name): 'Δ Betweenness Centrality',
            _CENTRALITY_DELTA.format(type=CentralityType.CLOS.display_name): 'Δ Closeness Centrality',
            _CENTRALITY_DELTA.format(type=CentralityType.DEG.display_name): 'Δ Degree Centrality'
        }

        for col, x_label in delta_columns.items():
            self._plot_delta_histogram(df[col].values, x_label, f"{col}_distribution.png")

        saved_df = pd.DataFrame(rows, columns=header_columns)
        saved_df.to_csv(self.output_path / CENTRALITY_CHANGES_CSV, index=False, header=True)
        for centrality_type in [CentralityType.BET, CentralityType.CLOS, CentralityType.DEG]:
            self._plot_centrality_score_comparison_2d(centrality_type=centrality_type, df=saved_df)
            self._plot_centrality_score_comparison_3d(centrality_type=centrality_type, df=saved_df)

    def _get_node_ids(self, network: nx.Graph) -> list[int]:
        node_ids: list[int] = []
        for node_key in self.common_node_keys:
            chain_id, residue_number, insertion, segment_id = node_key
            for node_id, attrs in network.nodes(data=True):
                if (attrs.get(CHAIN_ID) == chain_id and int(attrs.get(RESIDUE_NUMBER)) == residue_number
                        and (attrs.get(INSERTION) or "") == insertion and (attrs.get(SEGMENT_ID) or "") == segment_id):
                    node_ids.append(node_id)
                    break
        return sorted(node_ids)

    def _compute_modes(self, network: nx.Graph, num_modes: int) -> tuple[np.ndarray, np.ndarray, csr_array] | None:

        nodes = self._get_node_ids(network)

        laplacian = nx.laplacian_matrix(network, nodelist=nodes).astype(float)

        if not nx.is_connected(network):
            logging.error("Network must be connected for spectral partitioning.")
            return

        k = num_modes + 1
        eigen_values, eigen_vectors = eigsh(laplacian, k=k, which="SM")

        idx = np.argsort(eigen_values)
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        spectral_modes = eigen_vectors[:, 1:k]  # Fiedler +  {num_modes-1} more num_modes included

        return spectral_modes, eigen_values[1:k], laplacian

    def _save_eigen_vectors_and_values(self, spectral_modes: np.ndarray, eigen_values: np.ndarray = None,
                                       prefix: str = "source"):
        num_modes = spectral_modes.shape[1]
        mode_columns = [f"Mode_{i + 1}" for i in range(num_modes)]
        df_modes = pd.DataFrame(spectral_modes, columns=mode_columns)
        df_modes.to_csv(self.output_path / f"{prefix}_eigenvector.csv", index=False)
        if eigen_values is not None:
            df_eigenvalues = pd.DataFrame({
                "Mode": [f"Mode_{i + 1}" for i in range(len(eigen_values))],
                "Eigenvalue": eigen_values
            })

            df_eigenvalues.to_csv(self.output_path / f"{prefix}_eigenvalues.csv", index=False)

    @log_time("Community Structure Analysis")
    def _community_structure_analysis(self, num_modes=10, save_plot=True):
        source_modes, source_eigen_values, _ = self._compute_modes(self.source_network, num_modes=num_modes)
        target_modes, target_eigen_values, _ = self._compute_modes(self.target_network, num_modes=num_modes)

        self._save_eigen_vectors_and_values(source_modes, source_eigen_values, prefix=self.source_pdb_name)
        self._save_eigen_vectors_and_values(target_modes, target_eigen_values, prefix=self.target_pdb_name)

        target_modes_norm = np.where(target_modes < 0, -1, 1)
        source_modes_norm = np.where(source_modes < 0, -1, 1)
        self._save_eigen_vectors_and_values(target_modes_norm, None, prefix=f"{self.source_pdb_name}_norm")
        self._save_eigen_vectors_and_values(source_modes_norm, None, prefix=f"{self.target_pdb_name}_norm")

        num_nodes = target_modes_norm.shape[0]
        overlap = np.abs(target_modes_norm.T @ source_modes_norm) / num_nodes

        if save_plot:
            self._save_community_structure_analysis(overlap)

    def _save_community_structure_analysis(self, overlap: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(overlap, vmin=0, vmax=1, cmap='jet', origin='lower')
        ticks = np.arange(overlap.shape[0])
        labels = np.arange(1, overlap.shape[0] + 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontdict=FONT_STYLES["xlabel"])
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontdict=FONT_STYLES["xlabel"])
        ax.set_xlabel(f'Modes ({self.source_pdb_name})', fontsize=18)
        ax.set_ylabel(f'Modes ({self.target_pdb_name})', fontsize=18)
        ax.set_title('Community Structure', fontsize=22, pad=12)
        cbar = fig.colorbar(im, ax=ax, shrink=0.96, aspect=25, pad=0.03)
        cbar.ax.tick_params(**FONT_STYLES["colorbar"]["tick_params_label"])
        plt.tight_layout()
        plt.savefig(self.output_path / 'community_structure.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _parse_label(label: str) -> tuple[str, str, str, Optional[str]]:
        label = label.strip().replace("'", "").replace('"', "")
        parts = label.split(":")
        residue_name = parts[0]
        chain_id = parts[1]
        residue_number = parts[2]
        segment_id = parts[3] if len(parts) == 4 else None
        return residue_name, chain_id, residue_number, segment_id

    @log_time("Computing Graph Signal Processing")
    def _compute_graph_signal_processing(self, num_modes=10, save_plot: bool = True, has_title: bool = False):
        source_modes, source_eigen_values, _ = self._compute_modes(network=self.source_network, num_modes=num_modes)
        target_modes, target_eigen_values, _ = self._compute_modes(network=self.target_network, num_modes=num_modes)

        shift_values = ((target_eigen_values - source_eigen_values) / source_eigen_values) * 100

        if save_plot:
            self._save_shift_signals(has_title, shift_values)

    def _save_shift_signals(self, has_title: bool, shift_values: list[float], has_marker_title: bool = False) -> None:
        modes = np.arange(1, len(shift_values) + 1)
        shift_values = np.array(shift_values, dtype=float)

        plt.figure()
        plt.plot(modes, shift_values, marker='o')
        plt.axhline(0, color='black', linewidth=0.8, alpha=0.3, linestyle='--')

        plt.xlabel("Modes", fontsize=18)
        plt.ylabel("Eigenvalue shift (%)", fontsize=18)

        if has_title:
            plt.title("Percent Eigenvalue Shift per Mode", fontsize=22)

        plt.xticks(modes)

        if np.any(shift_values > 0) and np.any(shift_values < 0):
            y_min = min(shift_values)
            y_max = max(shift_values)
            margin = (y_max - y_min) * 0.1
            plt.ylim(y_min - margin, y_max + margin)

        if has_marker_title:
            for x, y in zip(modes, shift_values):
                offset = 1.2 if y < 0 else 0.8
                plt.text(x, y + offset, f"{y:.2f}", ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)

        plt.tight_layout()
        plt.grid(alpha=0.4)
        plt.savefig(self.output_path / 'shift_modes.png', dpi=300, bbox_inches='tight')
        plt.close()

    @log_time("Plotting Hub Residues")
    def _plot_hub_residues(self):
        source_high_bet_df = self._get_high_percentage_score(pdb_name=self.source_pdb_name,
                                                             path=self.source_input_path,
                                                             centrality_type=CentralityType.BET)

        target_high_bet_df = self._get_high_percentage_score(pdb_name=self.target_pdb_name,
                                                             path=self.target_input_path,
                                                             centrality_type=CentralityType.BET)

        self._visualize_high_betweenness_2d(pdb_name=self.source_pdb_name,
                                            pdb_save_name=self.source_pdb_name_save,
                                            network=self.source_network,
                                            high_betweenness_df=source_high_bet_df,
                                            save=True,
                                            node_keys_to_id_map=self.source_node_keys_to_id_map)
        self._visualize_high_betweenness_2d(pdb_name=self.target_pdb_name,
                                            pdb_save_name=self.target_pdb_name_save,
                                            network=self.target_network,
                                            high_betweenness_df=target_high_bet_df,
                                            save=True,
                                            node_keys_to_id_map=self.target_node_keys_to_id_map)

        self._visualize_high_betweenness_3d(pdb_name=self.source_pdb_name,
                                            pdb_save_name=self.source_pdb_name_save,
                                            network=self.source_network,
                                            high_betweenness_df=source_high_bet_df,
                                            save=True,
                                            node_keys_to_id_map=self.source_node_keys_to_id_map)
        self._visualize_high_betweenness_3d(pdb_name=self.target_pdb_name,
                                            pdb_save_name=self.target_pdb_name_save,
                                            network=self.target_network,
                                            high_betweenness_df=target_high_bet_df,
                                            save=True,
                                            node_keys_to_id_map=self.target_node_keys_to_id_map)

    @staticmethod
    def _format_node_label_by_node_id(network: nx.Graph, node: int) -> str:
        """ Formats node label like residue_name:chain_id:residue_number+insertion:segment_id """
        data = network.nodes[node]
        residue_name = data.get(RESIDUE_NAME, '')
        chain_id = data.get(CHAIN_ID, '')
        residue_number = str(data.get(RESIDUE_NUMBER, ''))

        insertion = data.get(INSERTION, '')
        if insertion and insertion not in _VALIDATION_VALUES:
            residue_number += str(insertion)

        segment_id = data.get(SEGMENT_ID, '')
        if segment_id and segment_id not in _VALIDATION_VALUES:
            label = f"{residue_name}:{chain_id}:{residue_number}:{segment_id}"
        else:
            label = f"{residue_name}:{chain_id}:{residue_number}"

        return label

    @staticmethod
    def _generate_node_key(row: pd.Series) -> _RESIDUE_KEY_TYPE:
        """ generate node key from attributes of a DataFrame row."""

        def _norm(v) -> str:
            v = '' if str(v) in [str(x) for x in _VALIDATION_VALUES] else v
            return '' if v is None else str(v)

        return row[CHAIN_ID], row[RESIDUE_NUMBER], _norm(row.get(INSERTION, '')), _norm(row.get(SEGMENT_ID, ''))

    def _find_high_betweenness_residues(self, high_betweenness_df: pd.DataFrame,
                                        node_keys_to_id_map: dict[_RESIDUE_KEY_TYPE, int]):
        highlight_nodes = set()
        scores = {}
        for _, row in high_betweenness_df.iterrows():
            node_id = node_keys_to_id_map.get(self._generate_node_key(row), None)
            if node_id is not None:
                highlight_nodes.add(node_id)
                scores[node_id] = row[CENTRALITY_SCORE]
        return highlight_nodes, scores

    def _visualize_high_betweenness_2d(self, pdb_name: str, pdb_save_name: str, network: nx.Graph,
                                       high_betweenness_df: pd.DataFrame, save: bool = True,
                                       node_keys_to_id_map: dict[_RESIDUE_KEY_TYPE, int] = None) -> None:
        """
        2D network visualization — high betweenness residues larger and colored while other residue nodes are smaller
        and colored with grey.

        Parameters
        ----------
        pdb_name : str
        pdb filename : str
        network : networkx.Graph
        high_betweenness_df : pd.DataFrame
            columns: atom_number, centrality_score, residue_name, chain_id, residue_number, insertion, segment_id
        save : bool
            save the figure if it is True.
        """

        highlight_residues, scores = self._find_high_betweenness_residues(high_betweenness_df, node_keys_to_id_map)

        pos = {}
        for node, data in network.nodes(data=True):
            pos[node] = (data.get(X, 0), data.get(Y, 0))

        all_residue_nodes = list(network.nodes())
        node_colors = []
        node_sizes = []
        node_labels = {}

        for node in all_residue_nodes:
            if node in highlight_residues:
                node_colors.append('#0761E8')
                node_sizes.append(120)
                node_labels[node] = self._format_centrality_node_label(
                    self._format_node_label_by_node_id(network, node),
                    splitter=":")
            else:
                node_colors.append('#d1d5db')
                node_sizes.append(15)

        edge_colors = []
        edge_widths = []
        for u, v in network.edges():
            if u in highlight_residues and v in highlight_residues:
                edge_colors.append('black')
                edge_widths.append(2.0)
            else:
                edge_colors.append('#9ca3af')
                edge_widths.append(0.5)

        plt.figure(figsize=(14, 10))

        nx.draw_networkx_edges(network, pos,
                               edgelist=network.edges(),
                               edge_color=edge_colors,
                               width=edge_widths,
                               alpha=0.4)

        normal_residue_nodes = [n for n in all_residue_nodes if n not in highlight_residues]
        nx.draw_networkx_nodes(network, pos,
                               nodelist=normal_residue_nodes,
                               node_color='#4b5563',
                               node_size=30,
                               alpha=0.7,
                               edgecolors='#374151',
                               linewidths=0.5)

        highlight_list = list(highlight_residues)
        h_scores = [scores.get(n, 0) for n in highlight_list]

        min_s, max_s = min(h_scores), max(h_scores)

        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(CentralityType.BET.display_name,
                                                                       ['#0761E8', '#b91c1c'])
        norm = plt.Normalize(vmin=min_s, vmax=max_s)
        h_colors = [cmap(norm(s)) for s in h_scores]

        nx.draw_networkx_nodes(network, pos,
                               nodelist=highlight_list,
                               node_color=h_colors,
                               node_size=90,
                               edgecolors='#374151',
                               linewidths=1,
                               alpha=0.8)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6, pad=0.02)
        cbar.set_label('Betweenness Centrality', fontsize=12, labelpad=20, rotation=270)

        label_pos = {node: (x, y + (max(pos.values(), key=lambda p: p[1])[1] -
                                    min(pos.values(), key=lambda p: p[1])[1]) * 0.02)
                     for node, (x, y) in pos.items() if node in node_labels}

        nx.draw_networkx_labels(network, label_pos,
                                labels=node_labels,
                                font_size=9,
                                font_color='#111827',
                                font_family=FONT_FAMILY
                                )

        plt.title(f'{pdb_name} — High Betweenness Centrality Residues', fontsize=18)
        plt.axis('off')
        plt.margins(0.02)

        if save:
            output_file = self.output_path / f'{pdb_save_name}_high_betweenness_network_2d.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
            logging.info(f"{output_file} has been saved!")

        plt.close()

    def _visualize_high_betweenness_3d(self, pdb_name: str, pdb_save_name: str, network: nx.Graph,
                                       high_betweenness_df: pd.DataFrame, save: bool = True,
                                       node_keys_to_id_map: dict[_RESIDUE_KEY_TYPE, int] = None) -> None:
        """
        3D interactive Plotly visualization — high betweenness residues colored and large others are gray and small

        Parameters
        ----------
        pdb_name: str
        pdb_save_name: str
        network: networkx.Graph
        high_betweenness_df: pd.DataFrame
        save: bool
        """
        highlight_residue_nodes, scores = self._find_high_betweenness_residues(high_betweenness_df, node_keys_to_id_map)

        fig = go.Figure()

        xe_n, ye_n, ze_n = [], [], []
        xe_s, ye_s, ze_s = [], [], []

        for u, v in network.edges():
            x0, y0, z0 = network.nodes[u].get(X, 0), network.nodes[u].get(Y, 0), network.nodes[u].get(Z, 0)
            x1, y1, z1 = network.nodes[v].get(X, 0), network.nodes[v].get(Y, 0), network.nodes[v].get(Z, 0)

            if u in highlight_residue_nodes and v in highlight_residue_nodes:
                xe_s += [x0, x1, None]
                ye_s += [y0, y1, None]
                ze_s += [z0, z1, None]
            else:
                xe_n += [x0, x1, None]
                ye_n += [y0, y1, None]
                ze_n += [z0, z1, None]

        fig.add_trace(go.Scatter3d(
            x=xe_n, y=ye_n, z=ze_n,
            mode='lines',
            line=dict(color='#9ca3af', width=1),
            opacity=0.4, name='Edges',
            showlegend=False,
            hoverinfo='none'
        ))

        fig.add_trace(go.Scatter3d(
            x=xe_s, y=ye_s, z=ze_s,
            mode='lines',
            line=dict(color='#374151', width=4),
            opacity=0.7,
            name='Hub edges',
            showlegend=False,
            hoverinfo='none'
        ))

        normal_nodes = [n for n in network.nodes() if n not in highlight_residue_nodes]
        if normal_nodes:
            fig.add_trace(go.Scatter3d(
                x=[network.nodes[n].get(X, 0) for n in normal_nodes],
                y=[network.nodes[n].get(Y, 0) for n in normal_nodes],
                z=[network.nodes[n].get(Z, 0) for n in normal_nodes],
                mode='markers',
                marker=dict(size=4, color='#4b5563', opacity=0.6, line=dict(width=0.5, color='#374151')),
                name='Other Residues',
                hoverinfo='text',
                text=[self._format_centrality_node_label(self._format_node_label_by_node_id(network, n), splitter=":")
                      for n in
                      normal_nodes]
            ))

        highlight_list = list(highlight_residue_nodes)
        if highlight_list:
            h_scores = [scores.get(n, 0) for n in highlight_list]
            fig.add_trace(go.Scatter3d(
                x=[network.nodes[n].get(X, 0) for n in highlight_list],
                y=[network.nodes[n].get(Y, 0) for n in highlight_list],
                z=[network.nodes[n].get(Z, 0) for n in highlight_list],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=h_scores,
                    colorscale=[[0, '#0761e8'], [1, '#b91c1c']],
                    colorbar=dict(
                        title="",
                        thickness=20,
                        len=0.75,
                        x=1.02,
                        ticks="outside",
                        tickwidth=2,
                        ticklen=6,
                        tickcolor="gray",
                        tickfont=dict(size=16, color="black", family=FONT_FAMILY)
                    ),
                    line=dict(width=1, color='#374151'),
                    opacity=0.9,
                ),
                text=[self._format_centrality_node_label(self._format_node_label_by_node_id(network, n), splitter=":")
                      for n in
                      highlight_list],
                textposition='top center',
                textfont=dict(size=14, color='black', family=FONT_FAMILY),
                name='High Betweenness Residues',
                hoverinfo='text',
                hovertext=[
                    f"{self._format_centrality_node_label(self._format_node_label_by_node_id(network, n), splitter=':')}"
                    f"<br>Betweenness: {scores.get(n, 0):.6f}"
                    for n in highlight_list
                ],
            ))

        fig.add_annotation(
            x=1.15,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Betweenness Centrality",
            textangle=90,
            showarrow=False,
            font=dict(size=20, family=FONT_FAMILY, color="black"),
            align="center"
        )

        fig.update_layout(
            autosize=True,
            title=dict(text=f'{pdb_name} — High Betweenness Centrality Residues',
                       x=0.5, xanchor='center', font=dict(size=_HTML_3D_SIZE)),
            scene=dict(
                domain=dict(x=[0, 1], y=[0, 1]),
                aspectmode="data",
                xaxis=dict(
                    title=dict(text=X.capitalize(), font=dict(size=14)),
                    tickfont=dict(size=14),
                    gridcolor='#e5e7eb'
                ),
                yaxis=dict(
                    title=dict(text=Y.capitalize(), font=dict(size=14)),
                    tickfont=dict(size=14),
                    gridcolor='#e5e7eb'
                ),
                zaxis=dict(
                    title=dict(text=Z.capitalize(), font=dict(size=14)),
                    tickfont=dict(size=14),
                    gridcolor='#e5e7eb'
                ),
                bgcolor='white',
            ),
            height=900,
            width=1200,
            showlegend=True,
            template='plotly_white',
            legend=dict(font=dict(size=_HTML_3D_SIZE)),
            font=dict(family=FONT_FAMILY)
        )

        if save:
            output_file = self.output_path / f'{pdb_save_name}_high_betweenness_network_3d.html'
            fig.write_html(output_file, config=_ZOOM_CONFIG)
            logging.info(f"{output_file} has been saved!")


def parse_residue_pairs(pairs_list):
    if not pairs_list:
        return None
    result = []
    for pair_str in pairs_list:
        left, right = pair_str.split(";")

        def parse_one(r):
            parts = r.split(":")
            chain_id = parts[0]
            residue_number = int(parts[1])
            insertion = parts[2] if len(parts) > 2 else ""
            segment_id = parts[3] if len(parts) > 3 else ""
            return chain_id, residue_number, insertion, segment_id

        result.append((parse_one(left), parse_one(right)))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rinpy compare",
        description=(
            "RinPy NetworkComparator performs comparative analysis "
            "between two protein residue interaction networks (RINs)."
        ),
        epilog=(
            "Example:\n"
            "rinpy compare --source_input_path ./4obe "
            "--target_input_path ./5v9u "
            "--output_path ./analysis_output "
            "--num_modes 20\n\n"
            "# num_modes specifies the number of non-zero Laplacian modes used."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--source_input_path",
        required=True,
        help="Directory containing the source state RIN results (e.g., apo structure)."
    )

    parser.add_argument(
        "--target_input_path",
        required=True,
        help="Directory containing the target state RIN results (e.g., ligand-bound structure)."
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory where comparison results will be saved."
    )

    parser.add_argument(
        "--residue_pairs",
        nargs="+",
        default=None,
        help=(
            "Optional residue pairs for Communication Path Efficiency analysis, "
            "including sequential path efficiency, end-to-end efficiency, "
            "path-restricted internal efficiency, and allosteric coupling.\n\n"
            "Format for each pair:\n"
            "chain:residue_number[:insertion[:segment_id]];"
            "chain:residue_number[:insertion[:segment_id]]\n\n"
            "If insertion or segment_id is not provided, it is treated as an empty string.\n\n"
            "Examples:\n"
            "--residue_pairs \"A:10;A:144\"\n"
            "--residue_pairs \"A:10:A;A:144\"\n"
            "--residue_pairs \"A:10;A:144\" \"A:15;A:30\""
        )
    )

    parser.add_argument(
        "--num_modes",
        type=int,
        default=20,
        help=(
            "Number of non-zero Laplacian eigenvalues used in community structure analysis. "
            "The trivial zero eigenvalue is excluded."
        )
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    residue_pairs = parse_residue_pairs(args.residue_pairs)
    logging.info(f"residue_pairs: {residue_pairs}")
    comparator = NetworkComparator(
        source_input_path=args.source_input_path,
        target_input_path=args.target_input_path,
        output_path=args.output_path,
        residue_pairs=residue_pairs
    )
    comparator.run(num_modes=args.num_modes)


if __name__ == '__main__':
    main()
