# -*- coding: utf-8 -*-
"""
    __author__ = 'Zehra Sarica'
    __email__ = ['sarica16@itu.edu.tr','zehraacar559@gmail.com']
"""

import csv
import logging
import os.path
from pathlib import Path
from typing import Optional

import networkx as nx

from rinpy.constants import CHAIN_ID, SEGMENT_ID, INSERTION, RESIDUE_NUMBER, RESIDUE_NAME

_WEIGHT = 'weight'
_ALLOSTERIC_COUPLING_CSV = "allosteric_coupling.csv"
_SEQUENTIAL_EFFICIENCY_CSV = "communication_path_efficiency.csv"
_EMPTY_SPACE = ''
_VALIDATION_VALUES = [None, '', "''", '""']

_RESIDUE_KEY_TYPE = tuple[str, int, str, str]
_RESIDUE_PAIR_TYPE = tuple[_RESIDUE_KEY_TYPE, _RESIDUE_KEY_TYPE]


class CommunicationPathEfficiency:
    """ computes the communication path efficiency for the given source and target networks.
    it calculates the sequential efficiency and allosteric coupling between the source and target residues.

    Parameters
    -----------
     source_network : nx.Graph
        Residue interaction network graph of the source protein - apo or wild type.
    target_network : nx.Graph
        Residue interaction network graph of the target protein - holo or mutant type
    output_path : Path
        Directory path where the results will be saved.
    residue_pairs : list of tuple, optional
        List of residue pairs for shortest path analysis.

        Each element represents a (source_residue, target_residue) pair.

        Each residue is uniquely identified by:
            (chain_id, residue_number, insertion_code, segment_id)

        Format:
            [
                ((chain_id, residue_number, insertion_code, segment_id),
                 (chain_id, residue_number, insertion_code, segment_id)),
                ...
            ]

        Example:
            [
                (('A', 12, '', ''), ('A', 144, '', '')),
                (('A', 13, '', ''), ('A', 145, '', ''))
            ]
    source_pdb_name : str
        Name of the source PDB file.
    target_pdb_name : str
        Name of the target PDB file.
    """

    def __init__(self, source_network: nx.Graph, target_network: nx.Graph, output_path: Path,
                 residue_pairs: list[_RESIDUE_PAIR_TYPE], source_pdb_name: str,
                 target_pdb_name: str) -> None:
        self.source_network = source_network
        self.target_network = target_network
        self.output_path = output_path

        self.residue_pairs = residue_pairs

        self.source_pdb_name = source_pdb_name
        self.target_pdb_name = target_pdb_name

        self.source_seq_effi_score = None
        self.source_end_to_end_effi_score = None
        self.source_path_rest_int_effi_score = None
        self.target_seq_effi_score = None
        self.target_end_to_end_effi_score = None
        self.target_path_rest_int_effi_score = None

    @staticmethod
    def _format_residue_pair_item(key: _RESIDUE_KEY_TYPE) -> str:
        chain_id, residue_number, insertion, segment_id = key
        insertion = (insertion or "").strip()
        segment_id = (segment_id or "").strip()
        name = f"{chain_id}{int(residue_number)}"
        if insertion:
            name += insertion
        if segment_id:
            name += f"({segment_id})"
        return name

    def _residue_pair_name(self, residue_pair: _RESIDUE_PAIR_TYPE) -> str:
        source, target = residue_pair
        return f"{self._format_residue_pair_item(source)}_{self._format_residue_pair_item(target)}"

    def compute(self) -> None:
        """ computes the sequential efficiency, end-to-end efficiency, path restricted internal efficiency scores
            of source and target networks. Also, computes the allosteric coupling scores of source and target networks
            for the given residue pairs.
        """
        for residue_pair in self.residue_pairs:
            source_results = self._compute_communication_path_efficiency(network=self.source_network,
                                                                         residue_pair=residue_pair)

            target_results = self._compute_communication_path_efficiency(network=self.target_network,
                                                                         residue_pair=residue_pair)
            if source_results is None or target_results is None:
                continue

            save_name = self._residue_pair_name(residue_pair=residue_pair)

            self.source_seq_effi_score, self.source_end_to_end_effi_score, self.source_path_rest_int_effi_score = source_results
            self.target_seq_effi_score, self.target_end_to_end_effi_score, self.target_path_rest_int_effi_score = target_results

            sequential_efficiency_score_map = {
                f'{self.source_pdb_name}_sequential_efficiency_score': self.source_seq_effi_score,
                f'{self.source_pdb_name}_end_to_end_efficiency_score': self.source_end_to_end_effi_score,
                f'{self.source_pdb_name}_path_restricted_efficiency_score': self.source_path_rest_int_effi_score,
                f'{self.target_pdb_name}_sequential_efficiency_score': self.target_seq_effi_score,
                f'{self.target_pdb_name}_end_to_end_efficiency_score': self.target_end_to_end_effi_score,
                f'{self.target_pdb_name}_path_restricted_efficiency_score': self.target_path_rest_int_effi_score
            }
            self._write_sequential_efficiency_scores_to_csv(sequential_efficiency_score_map, save_name)

            results = self._compute_allosteric_coupling(self.source_network, self.target_network, residue_pair)
            self._write_allosteric_coupling_scores_to_csv(results, save_name)

    def _write_sequential_efficiency_scores_to_csv(self, sequential_efficiency_score_map: dict[str, float],
                                                   save_name: str) -> None:
        """ save the sequential efficiency scores to a CSV file for the given results."""
        save_path = os.path.join(self.output_path, f"{save_name}_{_SEQUENTIAL_EFFICIENCY_CSV}")
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for metric, value in sequential_efficiency_score_map.items():
                writer.writerow([metric, f"{value:.6f}"])
        logging.info(f"sequential efficiency scores has been saved to {save_path}.")

    def _write_allosteric_coupling_scores_to_csv(self, results: dict[str, dict[str, float]], save_name: str) -> None:
        """ save the allosteric coupling scores to a CSV file for the given results."""
        save_path = os.path.join(self.output_path, f"{save_name}_{_ALLOSTERIC_COUPLING_CSV}")
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "metric", "value"])
            for category, metrics in results.items():
                for metric, value in metrics.items():
                    writer.writerow([category, metric, f"{value:.6f}"])

        logging.info(f"coupling scores has been saved to {save_path}.")

    @staticmethod
    def build_node_map(network: nx.Graph) -> dict[tuple[str, int, str, str], int]:
        """build node map for the given network."""
        node_map = {}
        for node, data in network.nodes(data=True):
            chain_id = str(data.get(CHAIN_ID))
            residue_number = int(data.get(RESIDUE_NUMBER))
            insertion = data.get(INSERTION, _EMPTY_SPACE)
            if str(insertion) in _VALIDATION_VALUES:
                insertion = _EMPTY_SPACE
            segment_id = data.get(SEGMENT_ID, _EMPTY_SPACE)
            if str(segment_id) in _VALIDATION_VALUES:
                segment_id = _EMPTY_SPACE
            key = (chain_id, residue_number, str(insertion), str(segment_id))
            node_map[key] = node
        return node_map

    @staticmethod
    def find_node(node_map: dict, residue_tuple: _RESIDUE_KEY_TYPE) -> dict:
        """ find the node with the given residue key tuple."""
        chain_id, residue_number, insertion, segment_id = residue_tuple
        if str(insertion) in _VALIDATION_VALUES:
            insertion = _EMPTY_SPACE
        if str(segment_id) in _VALIDATION_VALUES:
            segment_id = _EMPTY_SPACE

        key = (str(chain_id), int(residue_number), str(insertion), str(segment_id))
        return node_map.get(key)

    def _save_shortest_path_residues(self, network: nx.Graph, shortest_path_nodes: list[int],
                                     residue_pair: _RESIDUE_PAIR_TYPE):
        """ save the shortest path residues into a txt file"""
        if not shortest_path_nodes:
            return
        save_name = self._residue_pair_name(residue_pair=residue_pair)
        output_file = os.path.join(self.output_path, f"{save_name}_shortest_path_residues.txt")
        with open(output_file, "w") as f:
            for node_id in shortest_path_nodes:
                residue_name = network.nodes[node_id].get(RESIDUE_NAME, "")
                chain_id = network.nodes[node_id].get(CHAIN_ID, "")
                residue_number = network.nodes[node_id].get(RESIDUE_NUMBER, "")
                insertion = network.nodes[node_id].get(INSERTION, "")
                segment_id = network.nodes[node_id].get(SEGMENT_ID, "")
                insertion = insertion if insertion else "''"
                segment_id = segment_id if segment_id else "''"
                f.write(f"{residue_name}, {chain_id}, {residue_number}, {insertion}, {segment_id}\n")

    def _compute_communication_path_efficiency(self, network: nx.Graph, residue_pair: _RESIDUE_PAIR_TYPE) -> Optional[
        tuple[float, float, float]]:
        """ compute sequential efficiency of the network for the given residue pairs. """
        node_map = self.build_node_map(network)

        sequential_efficiency_values = []
        end_to_end_efficiency_values = []
        path_restricted_efficiency_values = []

        source_residue, target_residue = residue_pair

        source_node = self.find_node(node_map=node_map, residue_tuple=source_residue)
        target_node = self.find_node(node_map=node_map, residue_tuple=target_residue)
        if source_node is None or target_node is None:
            logging.warning(f"Nodes not found: {source_residue}-{target_residue}")
            return None
        try:
            path = nx.shortest_path(network, source_node, target_node, weight=_WEIGHT)
            self._save_shortest_path_residues(network, path, residue_pair)
            k = len(path)
            edge_count = k - 1

            if edge_count <= 0:
                return None

            weight_sum = 0
            for i in range(edge_count):
                u = path[i]
                v = path[i + 1]
                weight_sum += network[u][v].get(_WEIGHT, 1.0)

            if weight_sum == 0:
                return None

            # sequential efficiency
            efficiency_seq = edge_count / weight_sum
            sequential_efficiency_values.append(efficiency_seq)

            # End-to-end efficiency
            efficiency_end = 1 / weight_sum
            end_to_end_efficiency_values.append(efficiency_end)

            #  path restricted internal efficiency
            inverse_dist_sum = 0
            pair_count = 0

            for i in range(k):
                for j in range(i + 1, k):
                    try:
                        d = nx.shortest_path_length(network, path[i], path[j], weight=_WEIGHT)
                        if d and d > 0:
                            inverse_dist_sum += 1 / d
                            pair_count += 1
                    except nx.NetworkXNoPath:
                        continue
            if k > 1:
                path_efficiency = (2 / (k * (k - 1))) * inverse_dist_sum
                path_restricted_efficiency_values.append(path_efficiency)

        except nx.NetworkXNoPath:
            logging.error(f"No path found: {source_residue}-{target_residue}")
            return None

        seq_avg = sum(sequential_efficiency_values) / len(
            sequential_efficiency_values) if sequential_efficiency_values else 0
        end_avg = sum(end_to_end_efficiency_values) / len(
            end_to_end_efficiency_values) if end_to_end_efficiency_values else 0
        path_avg = sum(path_restricted_efficiency_values) / len(
            path_restricted_efficiency_values) if path_restricted_efficiency_values else 0

        return seq_avg, end_avg, path_avg

    def _compute_allosteric_coupling(self, source_network: nx.Graph, target_network: nx.Graph,
                                     residue_pair: _RESIDUE_PAIR_TYPE) -> dict[str, dict[str, float]]:
        """computes allosteric coupling metrics of the source and target network for the given pairs."""
        source_seq, source_end, source_path = self._compute_communication_path_efficiency(source_network, residue_pair)
        target_seq, target_end, target_path = self._compute_communication_path_efficiency(target_network, residue_pair)

        coupling_end_to_end = target_end - source_end
        coupling_path_restricted = target_path - source_path

        results = {
            f"{self.source_pdb_name}": {
                "end-to-end_efficiency": source_end,
                "path_restricted_internal_efficiency": source_path
            },
            f"{self.target_pdb_name}": {
                "end-to-end_efficiency": target_end,
                "path_restricted_internal_efficiency": target_path
            },
            "allosteric_coupling": {
                "end_to_end_efficiency": coupling_end_to_end,
                "path_restricted_internal_efficiency": coupling_path_restricted
            }
        }
        return results


def main():
    pass


if __name__ == '__main__':
    main()
