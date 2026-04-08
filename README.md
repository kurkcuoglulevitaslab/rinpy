<div align="center">
  <table border="0" cellpadding="0" cellspacing="0" style="border: none;">
    <tr>
      <td style="border: none; vertical-align: middle;">
        <img src="assets/RinPy_128x128.png" alt="RinPy logo">
      </td>
      <td style="border: none; vertical-align: middle; padding-left: 25px;">
        <h2 style="margin-top: -12px; margin-left: -40px">
          RinPy – Residue Interaction Network for Protein Structures
        </h2>
      </td>
    </tr>
  </table>
</div>

📖 Description
------------
**RinPy**, a pip-installable Python package, is designed for constructing, visualizing, and analyzing Residue Interaction Networks (RINs). RIN
describes a protein as a network of nodes interconnected by weighted edges. In this network, each node represents a
residue, nucleotide or a ligand at the average coordinate of its atomic coordinates. The edge weight between two nodes
is defined by the local interaction strength between the two residues. The average coordinates of the residues and/or
nucleotides are placed at the Cα atom or P atom, respectively, for protein-RNA/DNA complexes, such as the ribosome. Each node is annotated
with attributes such as Chain ID, Residue Number, Insertion Code, Segment ID, and its Cartesian coordinates.

**RinPy** integrates ensemble-scale statistical evaluation of multiple input structures and 
perturbation-aware comparative analysis within a unified graph-theoretical framework. 
It enables systematic comparison between two states of a protein complex, 
focusing on perturbation propagation, community structure composition, allosteric coupling, 
and communication efficiency. **RinPy** provides two integrated core modules: **RIN Process** and **Network Comparator**.

✨ Features
-------------

### RIN Process 
- Conversion of protein complexes and protein-RNA/DNA complexes, such as ribosome, into residue interaction networks (RINs).
- Support for weighted edges based on local interaction strength or affinity.
- Node annotation with Chain ID, Residue Number, Insertion Code, Segment ID, and Cartesian coordinates.
- Scalable pipeline for single PDB structures, large PDB ensembles, and PDBs from an MD trajectory.
- Cross-platform compatibility (Windows, macOS, Linux).
- Export of analysis results in PyMOL-compatible formats (e.g., .pdb and .pml files) to enable direct visualization.
- Centrality analysis (betweenness, closeness, degree).
- Identification of high-centrality residues using user-defined quantile thresholds.
- Frequency analysis of common hub residues across PDB structures within the given dataset.
- Graph spectral analysis for detecting potential hinge residues and dynamically connected domains.
- Interactive 2D heatmap visualizations for centrality metrics, residue frequency distributions, and graph spectral analysis results.
- Interactive 3D visualizations of graph spectral analysis results mapped onto protein structures.
### Network Comparator
Given a source and a target network (e.g., apo vs. holo), **Network Comparator** supports:
- Centrality comparison to quantify and visualize changes in degree, closeness, and betweenness centralities across the states.
- Edge weight comparison to detect increased, decreased, unchanged, added, and removed residue interactions between the two states.
- Community structure analysis to assess global domain organization and reordering of internal community under perturbation.
- Graph signal processing to evaluate perturbation-driven shifts in eigenvalue.
- Hub residue visualization to compare two states.
- Communication path efficiency analysis, such as sequential path efficiency, efficiency of the path, and allosteric coupling between user-defined residue pairs on the shortest path to quantify changes.
- Generation of comparative visual and tabular outputs, including 2D/3D network visualizations and structured CSV reports.

🖥️ RinPy GUI
--------------
To facilitate ease of use, the RinPy Graphical User Interface (GUI) was developed as a companion to this package for the scientific community. The standalone RinPy GUI can be downloaded from [**here**](https://drive.google.com/drive/folders/1GlLva31y7Ebpmpd8Dk6uQmGHCem2vWfO?usp=drive_link), and the **User Manual** is available [**User Manual**](./assets/RinPy_GUI_User_Manual.pdf). Additionally, the User Manual can be accessed directly within the RinPy GUI application via **Help** > **User Manual**.

⚙️ Installation
-----------------

### 📌 Prerequisites (Important)

- **Python ≥ 3.10 and <3.14** are required.
- RinPy depends on **NetworkX ≥ 3.4**, which requires Python 3.10 or newer.

We strongly recommend using a Conda-based environment manager (**Miniconda** or **Anaconda**) to ensure stable dependency handling.

- **Miniconda** (lightweight, recommended):  
  https://docs.conda.io/en/latest/miniconda.html

- **Anaconda** (full distribution):  
  https://www.anaconda.com/products/distribution

---

### 🚀 Installation via PyPI (Recommended)

RinPy is available on **PyPI** ([rinpy](https://pypi.org/project/rinpy/)) and can be installed directly using **pip**.

The following steps demonstrate how to create and activate a conda virtual environment, install **RinPy**, verify the
installation, and run the program from the command line:

```bash
# Create a conda virtual environment
conda create -n rinpy_env python=3.10 -y

# Activate the environment
conda activate rinpy_env or source activate rinpy_env

# Install RinPy
pip install rinpy

# (Optional) To upgrade to the latest version if RinPy is already installed
pip install --upgrade rinpy

# Check installation
rinpy --help
```
#### To run from Command Line for the RIN process:
```bash
python -m rinpy \
      --input_path INPUT_PATH \
      --output_path OUTPUT_PATH \
      --calculation_option_file path/to/calculation_options.json

or 

rinpy --input_path INPUT_PATH \
      --output_path OUTPUT_PATH \
      --calculation_option_file path/to/calculation_options.json
```
#### To run from Command Line for the Network Comparator:
```bash
python -m rinpy compare --source_input_path SOURCE_INPUT_PATH \
      --target_input_path TARGET_INPUT_PATH \
      --output_path OUTPUT_PATH \
      --residue_pairs "A:10;A:144" \
      --num_modes 20

or

rinpy compare --source_input_path SOURCE_INPUT_PATH \
      --target_input_path TARGET_INPUT_PATH \
      --output_path OUTPUT_PATH \
      --residue_pairs "A:10;A:144" \
      --num_modes 20
```

#### Parameter details for the terminal run for the RIN process:

- `--input_path`: Directory containing the input PDB files.
- `--output_path`:  Directory where results and output files are saved.
- `--calculation_option_file`: JSON file containing configuration parameters. To download the example JSON file,
  click [calculation_options.json](./src/rinpy/calculation_options.json).

#### Parameter details for the terminal run for the Network Comparator:

- `--source_input_path`: Directory containing the output of the RIN process, such as apo or wild states.
- `--target_input_path`: Directory containing the output of the RIN process, such as holo or mutant states.
- `--output_path`: Directory where results and output files are saved.
- `--residue_pairs`: Accepts a semicolon-separated list of residues in the
  format `chain_id:residue_number[:insertion][:segment_id]`, where `insertion` and `segment_id` should be
  provided only if available to ensure unique identification of the residue; otherwise, they may be omitted (e.g.,
  `"A:10;A:144"` `"A:12;A:56"` or `"A:15:D:ABC;A:120:C:AB"`).
- `--num_modes`: Number of non-zero Laplacian eigenvalues used in community structure analysis. The trivial zero
  eigenvalue is excluded.

### 🔧 Installation from Source (Alternative)

If you prefer to install RinPy from source instead of using pip, follow the steps below:

1. **Clone the repository:**

Replace `<username>` with GitHub username:

**Using HTTPS:**

```bash
git clone https://github.com/<username>/rinpy.git
```
**Using SSH (requires GitHub SSH keys):**
```bash
git clone git@github.com:<username>/rinpy.git
```
2. **If required, make scripts executable (macOS/Linux only):**

```bash
chmod -R +x rinpy
```

3. **Navigate to the project folder:**

```bash
cd rinpy
```

4. **Create a Python virtual environment (strongly recommended):**

**Using conda (strongly recommended):**

```bash
conda create -n rinpy_env python=3.10
conda activate rinpy_env or source activate rinpy_env
```

**Or using `venv` with a supported Python version (`3.10` ≤ Python < `3.14`):**

```bash
python3.10 -m venv rinpy_env

# Activate the virtual environment
# On Windows:
rinpy_env\Scripts\activate
# On macOS/Linux:
source rinpy_env/bin/activate
```
5. **Install the package:**

- **Editable (recommended):**

```bash
pip install -e .
```
- **Non-editable:**

```bash
pip install .
```

🚀 Usage
---------

**RinPy** can also be used programmatically through the **RINProcess** and **NetworkComparator** APIs within your Python
scripts. To perform comparative analysis, **RINProcess** must be executed first for all centrality types (betweenness,
closeness, degree), as it generates the required input files for **NetworkComparator**, including centrality measures (
betweenness, closeness, and degree) and the residue interaction network in **GraphML** format. Create a Python file
named **`main.py`**, insert the content given below, and execute the script via the terminal (python **`main.py`**) or
an equivalent environment.

### RIN Process Example

RinPy uses Python's **`multiprocessing`** module.

When running RinPy from a script on macOS or Windows, make sure your entry point is protected:

```python
from multiprocessing import freeze_support
from rinpy import RINProcess


def main():
    # Define calculation options as a JSON-like dictionary
    calculation_options = {
        'remove_hydrogen': {
            'is_checked': True,
            'value': 0
        },
        'betweenness': {
            'is_checked': True,
            'value': 5
        },
        'closeness': {
            'is_checked': True,
            'value': 100
        },
        'degree': {
            'is_checked': True,
            'value': 100
        },
        'cluster_number': {
            'is_checked': True,
            'value': 3
        },
        'cutoff': {
            'is_checked': True,
            'value': 4.5
        }
    }

    # Initialize RINProcess, 
    #
    # output_path is mandatory.
    # For input, provide ONLY ONE of the following:
    #   - input_path
    #   - pdb_ids
    #   - trajectory_file
    # The system checks inputs in the following order:
    # input_path → pdb_ids → trajectory_file
    #
    # If using pdb_ids, set input_path and trajectory_file to None.

    rin = RINProcess(
        input_path="path/to/input/files",
        output_path="path/to/output",
        pdb_ids=None,  # list of PDB IDs to process if download from protein data bank such ["4OBE", "4DSN"].
        ligand_dict=None,  # optional ligand information
        calculation_options=calculation_options,
        trajectory_file=None,  # "path/to/input/files"
        stride=1  # The default is 1. This parameter is used in conjunction with trajectory_file.
    )

    # Start the process
    rin.start_process()


if __name__ == "__main__":
    freeze_support()
    main()
```

### Network Comparator Example

**Note:** To perform comparative analysis, **RINProcess** must be executed first for all centrality types (betweenness,
closeness, degree).

```python
from rinpy import NetworkComparator

# Each residue is defined as a tuple in the format (chain_id, residue_number, insertion, segment_id).
# If insertion or segment ID is not available, use an empty string ''.
# The residue pairs are a list of user-defined pairs. Each residue must be specified as 
# (chain_id, residue_number, insertion, segment_id).

comparator = NetworkComparator(
    source_input_path=r"path to apo",
    target_input_path=r"path to holo",
    output_path=r"path to result folder",
    residue_pairs=[(('A', 10, '', ''), ('A', 144, '', ''))],
)

# num_modes specifies the number of non-zero Laplacian eigenvalues used in community structure 
# analysis.
comparator.run(num_modes=20)
```

---------------------------------------------------------

📝 Notes
----------
**RinPy** requires `--output_path`. In addition, exactly one of the following input options must be provided to **RINProcess**: 
`--input_path`, `--pdb_ids`, or `--trajectory_file`. These arguments are mutually exclusive and cannot be
used together. Argument details as follows:

- `--output_path`: Folder where processed RINs and results will be saved.
- `--input_path`: Folder containing your input PDB files.
- `--pdb_ids`: List of specific PDB IDs to process.
- `--ligand_dict`: Optional dictionary with ligand information.
- `--calculation_options`: JSON-like dictionary specifying which calculations to run and their parameters.
- `--trajectory_file`: The MD trajectory file (pdb format) which contains multiple snaphots from MD.
- `--stride`: Default is **1**. This parameter is used in conjunction with `--trajectory_file` parameter. It is
  applicable only when a trajectory file is provided, and determines the interval at which frames are extracted, meaning
  that PDB structures are generated every **stride** frames from the trajectory.

For **NetworkComparator**, the following arguments are required:

- `--source_input_path`: Directory containing the source-state RIN results (e.g., apo structure).
- `--target_input_path`: Directory containing the target-state RIN results (e.g., ligand-bound structure).
- `--output_path`: Directory where comparison results will be saved.
- `--residue_pairs`: Optional residue pairs used for communication path efficiency analysis, including sequential path
  efficiency, end-to-end efficiency, and allosteric coupling.

## 📊 Case Study of RinPy

After setting up a Conda virtual environment and installing **RinPy**, navigate to the **`tests`** folder. It contains
two
Python scripts and their corresponding `.sh` files for running the analyses.
The **`kras_sos1_input`** folder contains KRAS–SOS1 PDB files, whereas **`kras_input`** includes KRAS PDB structures
used in the
case study.

To analyze the MD trajectory PDB files, download the dataset
from [here](https://drive.google.com/file/d/1COAZqgiCGVhkYvCI6DQLv2YzJ5emcvRN/view?usp=drive_link).
You may update the existing a Python script and its corresponding `.sh` file under `tests` directory.

📄 License
------------
MIT License. See [LICENSE](LICENSE) file for details.

📘How to Cite
---------------

If you use this repository, please cite this study as follows:

```bibtex
@article{rinpy,
  author = {Sarica, Zehra and Sungur, Fethiye Aylin and Kurkcuoglu, Ozge},
  title = {RinPy, a Python Package for Residue Interaction Network Model to Analyze Protein Structures and Predict Ligand Binding Sites},
  journal = {Journal of Chemical Information and Modeling},
  volume={},
  year={2026},
  publisher={},
  doi = {10.1021/acs.jcim.6c00004}
}
```

📬 Contact
------------
For questions, contact: zehraacar559@gmail.com, sarica16@itu.edu.tr
