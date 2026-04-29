# intDesc-MD

intDesc-MD is a software tool for identifying and quantifying molecular interactions from molecular dynamics (MD) simulation trajectories.

**intDesc-MD** is software for automatic, comprehensive, and precise identification and visualization of various molecular interactions based on 3D structures. This repository presents **intDesc-MD**, a version of the **intDesc** program series that extends this capability to multiple structural frames obtained from MD simulation trajectories.

---

## Scope of this release

This public release focuses on the **`medium`** mode, which is the workflow used in the study associated with this repository. The source code and command-line help also contain **`ligand`** and **`mutant`** modes, but they are not documented in detail in this README.

---

## Features

- **intDesc-MD** allows detailed identification of numerous interactions, such as CH-O, CH-π, NH-π, S-π, S-O, and dipole interactions.
- **intDesc-MD** analyzes interactions of proteins or peptides mediated by a molecule defined as solvent.
- **intDesc-MD** enumerates the number of each interaction and outputs them as interaction descriptors.
- Each interaction criterion contains user-tunable parameters that can be changed in a configuration file.
- Regions for interaction calculations are defined using a YAML file.
- Trajectory coordinates are processed using **MDAnalysis**, which supports various trajectory formats.
- **intDesc-MD** requires a structure file in MOL2 format in which hydrogens and Gasteiger charges are assigned.
- **intDesc-MD** can generate per-frame output files (raw interaction list, interaction count list, one-hot list, and **PyMOL** script), as well as an aggregated interaction count file for the full trajectory.
- **intDesc-MD** supports parallel processing.

It provides two implementations:

- **CPU multithreaded version** (`src/trajectory_descriptor.py`)
- **MPI-based version** (`src/trajectory_descriptor_mpi.py`)

---

## Requirements and tested environment

A conda environment definition file is provided as `intDesc-MD/environment.yaml`. Please use this file as the primary reference for the exact package versions, channels, and transitive dependencies used for the current public release.

The current public release has been validated for the **serial CPU version** using the conda environment defined in `environment.yaml`. The major dependencies are:

- Python 3.11.14
- NetworkX 3.6.1
- NumPy 2.4.2
- pandas 3.0.0
- BioPandas 0.5.1
- PyYAML 6.0.3
- MDAnalysis 2.10.0

To create the conda environment, run the following commands after cloning the repository and switching to the `intDesc-MD` branch:

```bash
cd intDesc-MD
conda env create -f environment.yaml
conda activate intDescMD_env
```

The complete dependency list, including build strings and additional packages installed in the tested environment, is provided in `environment.yaml`.

The **MPI version** was **not** separately verified in the current installation test. The system manual lists **mpi4py** as the required Python package for the MPI implementation, but `mpi4py` is not included in the provided `environment.yaml`. For MPI execution, please install an `mpi4py` version compatible with your Python version and local MPI environment.

---

## Setup

Clone the repository and switch to the `intDesc-MD` branch:

```bash
git clone https://github.com/riken-yokohama-AI-drug/intDesc.git
cd intDesc
git checkout intDesc-MD
```

The public directory for this release is `intDesc-MD/`. Move into this directory before creating the environment or running the program:

```bash
cd intDesc-MD
```

---

## Repository structure

The current public directory is organized as follows:

```text
intDesc-MD/
├── environment.yaml
├── install_test
│   ├── data.zip
│   └── run_test.sh
├── sample
│   └── medium/
│       ├── cyclic_gly6.dcd
│       ├── cyclic_gly6.mol2
│       ├── mol_select.yaml
│       ├── vdw_radius.yaml
│       ├── param.yaml
│       └── priority.yaml
└── src
    ├── edit_parametar.py
    ├── group.yaml
    ├── interaction.py
    ├── mol2.py
    ├── my_math.py
    ├── trajectory.py
    ├── trajectory_descriptor.py
    ├── trajectory_descriptor_mpi.py
    └── water_definition.txt

```

Main components:

- `environment.yaml`  
  Conda environment definition file for reproducing the tested Python environment.
- `src/trajectory_descriptor.py`  
  CPU multithreaded version of intDesc-MD.
- `src/trajectory_descriptor_mpi.py`  
  MPI-based version intended for HPC environments.
- `src/trajectory.py`  
  Trajectory loading utilities based on MDAnalysis.
- `src/mol2.py`  
  MOL2 parsing and molecule-type assignment.
- `src/interaction.py`  
  Core interaction detection logic.
- `src/my_math.py`  
  Geometry helper functions used in interaction calculations.
- `src/edit_parametar.py`  
  Utility script for editing interaction parameter files.
- `src/group.yaml`  
  Interaction group definition file.
- `src/water_definition.txt`  
  Water molecule definition file used for output generation.
- `sample/`  
  Example input files.
- `install_test/`  
  Installation test data and scripts.
  The `install_test/` directory contains a compressed test dataset (`data.zip`) and a test script (`run_test.sh`). 
  When `run_test.sh` is executed, `data.zip` is extracted and a temporary `data/` directory is generated for the installation test.
  The test also generates a `result/` directory containing output files used for MD5 checksum validation.

Note that Python cache files such as `__pycache__/` are not part of the intended repository contents and do not need to be included in the public release.

---

## Installation test

An automated installation test is currently provided for the **serial CPU version only**.

The MPI version is intended for HPC environments and may require site-specific MPI settings, so an automated installation test is **not** included in this release.

The install test provided in this repository is a **minimal validated example for the `medium` mode**, using a **DCD trajectory** together with a **MOL2 structure file**. Other trajectory formats supported by **MDAnalysis** may also be used in normal analyses, but the install test is based on DCD for simplicity and reproducibility.

From the `intDesc-MD/` directory, move to the test directory and run:

```bash
cd install_test/
bash run_test.sh
```

The test script validates the generated output files using `md5sum`.  
During the test, MDAnalysis may emit a warning when reading the DCD trajectory. This is a known warning from MDAnalysis and does not affect the validated output files; see the Notes section below.

---

## How to run intDesc-MD

In this README, only the **`medium`** mode is documented in detail. The following commands assume that they are executed from the `intDesc-MD/` directory.

### CPU multithreaded version

```text
python src/trajectory_descriptor.py medium \
    [trajectory file] \
    [mol2 file] \
    [interaction target molecule specification file] \
    [van der Waals radius definition file] \
    [interaction criteria file] \
    [interaction priority file] \
    [prefix of output file name] \
    [options]
```

### MPI-based version

```text
mpiexec -n [number of processes] python src/trajectory_descriptor_mpi.py medium \
    [trajectory file] \
    [mol2 file] \
    [interaction target molecule specification file] \
    [van der Waals radius definition file] \
    [interaction criteria file] \
    [interaction priority file] \
    [prefix of output file name] \
    [options]
```

### Frequently used options for `medium`

- `--allow_mediate_pos [number]`  
  Position between solvent atoms that allows detection of solvent-mediated interactions.
- `--on_14`  
  Detect 1-3 / 1-4 interactions.
- `--dup`  
  Keep duplicate interactions.
- `--no_mediate`  
  Disable solvent-mediated interaction detection.
- `--switch_ch_pi`  
  Use the old definition for CH-π / NH-π / OH-π detection.  
  *(CPU multithreaded version only)*
- `--out_raw`  
  Output the raw interaction list.
- `--out_count`  
  Output the per-frame interaction count list.
- `--out_count_traj`  
  Output the aggregated interaction count list for the whole trajectory.
- `--out_one_hot`  
  Output the one-hot list.
- `--out_pml`  
  Output a PyMOL script.
- `--start [frame]`  
  First frame to read.
- `--stop [frame]`  
  Last frame to read.
- `--interval [frame step]`  
  Frame loading interval.
- `--process [number]`  
  Number of worker processes.  
  *(CPU multithreaded version only)*

> **Important:** At least one of `--out_count` or `--out_count_traj` must be specified.

### Minimal example (`medium` mode)

```bash
python src/trajectory_descriptor.py medium \
    sample.dcd \
    sample.mol2 \
    sample/mol_select.yaml \
    data/vdw_radius.yaml \
    data/param.yaml \
    data/priority.yaml \
    result/sample1 \
    --out_count
```

---

## Interaction target molecule specification file (`mol_select.yaml`)

This YAML file defines which molecular components in the MOL2 structure are treated as peptide, ligand, protein, membrane, or solvent during interaction analysis.

### General syntax

```yaml
[molecule_type_name]:
  [field]: [value]
```

### Molecule types relevant to `medium`

The `medium` mode can use the following entry types:

- `peptide` / `peptide_[N]`
- `ligand` / `ligand_[N]`
- `protein` / `protein_[N]`
- `membrane` / `membrane_[N]`
- `solvent` / `solvent_[N]`

Multiple numbered entries are allowed, for example `solvent_1`, `solvent_2`, `peptide_1`, and so on.

### Supported fields

- `chain`  
  Chain ID. A single chain such as `A` or a list such as `[A, B]`.
- `name`  
  Residue name. A single name such as `GLY` or a list such as `[GLY, ARG]`.
- `num`  
  Residue number. A single integer, a list, or a range string such as `"100:110"`.
- `type`  
  `main` or `side`. In this program, `main` corresponds to atoms `C`, `N`, `CA`, `O`, `H`, and `HA`; all others are treated as `side`.
- `num_identity`  
  Medium-mode-only option for enabling intra-residue interaction detection for selected residue numbers. This must be used together with another selector such as `chain`, `name`, or `num`.

### Selection rules

- `name` and `num` cannot be used at the same time in the same entry.
- If multiple chains are specified in `chain: [A, B]`, `name` and `num` should not be combined with that multi-chain specification.
- `num_identity` is available only in `medium` mode.

### Example used for the installation test

The installation test uses the following `mol_select.yaml`:

```yaml
peptide:
  chain: A
solvent_1:
  name: TIP
solvent_2:
  name: POP
```

In this example:

- chain `A` is treated as the peptide,
- residue name `TIP` is treated as solvent,
- residue name `POP` is also treated as solvent.

Here, `TIP` and `POP` are abbreviated residue names used in the structure file, corresponding to **TIP3 water** and **POPC lipid**, respectively.

In this repository, POP is intentionally defined as `solvent_2` to match the workflow used in the associated study. In this setting, direct peptide-POP interactions are still detected, and POP can also participate in solvent-mediated interaction categories. This configuration was used in the study even though lipid-mediated peptide interactions were not included in the final analysis reported in the paper. If POP is instead defined under `membrane:`, direct peptide-membrane interactions are detected, but solvent-mediated categories involving that component are not generated.

### Alternative membrane definition

If you want to treat POP strictly as a membrane component rather than as a solvent-like mediator, use a membrane entry such as:

```yaml
peptide:
  chain: A
membrane:
  name: POP
solvent:
  name: TIP
```

---

## Output files

The following sections summarize the main output files relevant to the `medium` workflow.

### Raw interaction list

`prefix_frame[N]_raw_list.txt`

This is a structured text file describing each detected interaction in detail. In `medium` mode, the file is organized as repeated blocks.

#### Block structure

- `K ...`  
  Pair-type label for the interaction block.
- `I1-2 ...`  
  Interaction label and geometric values (distance, angle, etc.) for the primary atom pair.
- `I3-4 ...`  
  Interaction label and geometric values for the secondary atom pair in solvent-mediated interactions. This line appears only when applicable.
- Atom records such as `PC*`, `SC*`, `MemC*`, `LC*`, `ProC*`  
  Atom identity lines for the interacting atoms.
- Neighbor records such as `PN*`, `SN*`, `MemN*`, `LN*`, `ProN*`  
  Bonded-neighbor lines for those atoms.

#### Pair labels used in `medium` raw output

- `P-P` : peptide-peptide
- `P-S[N]` : peptide-solvent
- `P-Mem` : peptide-membrane
- `L-Pro` : ligand-protein
- `L-S[N]` : ligand-solvent
- `Pro-S[N]` : protein-solvent
- `P-S[N]-P` : peptide-solvent-peptide
- `L-S[N]-Pro` : ligand-solvent-protein
- `Pro-S[N]-Pro` : protein-solvent-protein

#### Atom-line prefixes in `medium` raw output

- `PC*` / `PN*` : interacting peptide atom / its bonded neighbor
- `SC*` / `SN*` : interacting solvent atom / its bonded neighbor
- `MemC*` / `MemN*` : interacting membrane atom / its bonded neighbor
- `LC*` / `LN*` : interacting ligand atom / its bonded neighbor
- `ProC*` / `ProN*` : interacting protein atom / its bonded neighbor

Each atom line contains residue name, residue number, atom name, atom number, and atom type. Neighbor lines additionally include the bond type to the corresponding central atom.

#### Simplified example

```text
K P-S1-P
I1-2 HB_NH_O 2.9450 145.2000 112.3000
I3-4 HB_OH_O 2.8010 133.7000 109.5000 118.1000
PC1 GLY 3 N 41 N.am
PN1 GLY 3 H 42 H 1
SC2 TIP 501 O 12001 O.3
SN2 TIP 501 H1 12002 H 1
SC3 TIP 501 H2 12003 H
PC4 GLY 4 O 56 O.2
PN4 GLY 4 C 55 C.2 2
```

The exact number of geometric values after `I1-2` and `I3-4` depends on the interaction type.

### Interaction count list

`prefix_frame[N]_interaction_count_list.csv`

This is a per-frame CSV-like text file containing interaction counts. It does **not** include a header row. Each line is:

```text
[label],[count]
```

#### Label format

The label uses `#` as a separator and encodes both the pair type and the interaction label.

Typical label formats relevant to `medium` are:

- `P#[interaction_label]#P` : peptide-peptide
- `P#[interaction_label]#S[N]` : peptide-solvent
- `P#[interaction_label]#Mem` : peptide-membrane
- `Pro#[interaction_label]#S[N]` : protein-solvent
- `P#[interaction_label]#S[N]#[interaction_label]#P` : peptide-solvent-peptide
- `Pro#[interaction_label]#S[N]#[interaction_label]#Pro` : protein-solvent-protein

Depending on the system definition, medium-mode analyses may also contain ligand-related labels such as ligand-protein or ligand-solvent counts.

#### Example

```text
P#HB_NH_O#S1,12
P#CH_O#P,5
P#HB_NH_O#S1#HB_OH_O#P,3
```

### Trajectory-level count list

`prefix_trajectory.csv`

This file contains aggregated interaction counts across all processed frames.

In the current implementation, the first line is:

```text
flames,<number_of_processed_frames>
```

and the remaining lines follow the same `[label],[count]` format as the per-frame interaction count list.

If `--out_count_traj` is used **without** `--out_count`, the per-frame count files are removed after aggregation.

### One-hot list

`prefix_frame[N]_one_hot_list.csv`

This file stores detected interactions in a one-hot encoded table. In `medium` mode, the file consists of:

1. one-hot columns describing interaction classes,
2. metadata columns describing the interacting atom pair.

#### One-hot label patterns used in `medium`

- `P[N]P[N]_[interaction_label]` : peptide-peptide
- `P[N]S[N]_[interaction_label]` : peptide-solvent
- `P[N]Mem_[interaction_label]` : peptide-membrane
- `L[N]Pro[N]_[interaction_label]` : ligand-protein
- `L[N]S[N]_[interaction_label]` : ligand-solvent
- `Pro[N]S[N]_[interaction_label]` : protein-solvent

#### Metadata columns

- `dist`
- `interaction_label`
- `molcular_type`
- `chain`
- `residue`
- `residue_number`
- `atom_name`
- `atom_number`
- `atom_type`
- `partner_molcular_type`
- `partner_chain`
- `partner_residue`
- `partner_residue_number`
- `partner_atom_name`
- `partner_atom_number`
- `partner_atom_type`

In other words, each row records one detected interaction together with both the one-hot label and the atom-level metadata for the focal atom and its partner atom.

### PyMOL visualization script

`prefix_frame[N].pml`

This file is used for visualization in PyMOL.

### Optional ligand-only output not covered here

The codebase also contains an `Interaction Sum list` output for `ligand` mode, but it is not part of the `medium`-focused workflow documented in this README.

---

## Notes

### Runtime stability

The current public release includes minor runtime-stability improvements identified during pre-release testing. In older internal versions, some systems could terminate unexpectedly depending on the input system and execution conditions.

### MDAnalysis warning when reading DCD files

When reading DCD trajectories, **MDAnalysis** may emit a `DeprecationWarning` related to an internal future behavior change in the DCD reader. This warning does **not** affect the current install test or the current tested environment described above.

### Trajectory format support

The install test uses **DCD** as the minimal validated example. In normal usage, other trajectory formats supported by **MDAnalysis** may also be used.

---

## Notes on MOL2 file format

For large molecular systems (for example, membrane-solvent systems with a large number of atoms), the standard MOL2 format may become misaligned because of column-width limitations.

In such cases, please ensure that:

- atom indices and coordinate columns are properly aligned
- the MOL2 file format is adjusted to avoid column overflow

We internally use a preprocessing script to handle this issue, but it is not currently included in this repository.

---

## Citation

If you use this software, please cite:

*(TBD — will be updated upon publication)*

---

## Author

Tsutomu Yamane  
RIKEN Yokohama
