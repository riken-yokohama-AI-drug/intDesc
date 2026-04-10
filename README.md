# intDesc-MD

intDesc-MD is a software tool for identifying and quantifying molecular interactions from molecular dynamics (MD) simulation trajectories.

**intDesc-MD** is software for automatic, comprehensive, and precise identification and visualization of various molecular interactions based on 3D structures. This repository presents **intDesc-MD**, a version of the **intDesc** program series that extends this capability to multiple structural frames obtained from molecular dynamics (MD) simulation trajectories.

---

## Features
- **intDesc-MD** allows the detailed identification of numerous interactions, such as CH-O, CH-π, NH-π, S-π, S-O, and dipole interactions.
- **intDesc-MD** analyzes interactions of proteins or peptides mediated by a molecule defined as solvent.
- **intDesc-MD** enumerates the number of each interaction and outputs them as interaction descriptors.
- Each interaction criterion contains user-tunable parameters, which can be changed in the configuration file.
- In **intDesc-MD**, the regions for interaction calculations are defined using a YAML file.
- In **intDesc-MD**, trajectory coordinates are processed using the **MDAnalysis** module, which supports various trajectory formats. For details, see:
  https://www.mdanalysis.org/docs/documentation_pages/coordinates/init.html
- **intDesc-MD** requires a structure file in MOL2 format, in which hydrogens and Gasteiger charges are assigned.
- **intDesc-MD** generates four files for each frame in the trajectory (raw list, interaction count list, one-hot list, and **PyMOL** script for visualization), as well as one aggregate interaction count list for all frames.
- **intDesc-MD** supports parallel processing to handle a large number of structures.

It provides two implementations:
- **CPU multithreaded version** (trajectory_descriptor.py)
- **MPI-based version** (trajectory_descriptor_mpi.py)

---

## Requirements
The program does not require installation. Place this repository in an environment where the following Python packages are available:

- python 3.9.23
- networkx 3.2.1
- numpy 1.23.5
- pandas 1.5.3
- biopandas 0.5.1
- pyyaml 6.0.2
- mdanalysis 2.7.0

---

## Setup
Clone the repository:

git clone https://github.com/your-username/your-repository.git

Then move to the directory and run the test:

cd install_test/
bash run_test.sh

---

## How to run intDesc

### CPU multithreaded version ###
```text
$ python trajectory_descriptor.py medium \
        [trajectory file] \
        [mol2 file] \
        [Interaction target molecule specification file] \
        [van der waals radius definition file] \
        [interaction criteria file] \
        [interaction priority file] \
        [prefix of output file name] \
        (--allow_mediate_position [number]) \
        (--on_14) \
        (--dup) \
        (--no_mediate) \
        (--out_raw) \
        (--out_count) \
        (--out_count_traj) \
        (--out_one_hot) \
        (--out_pml) \
        (--start [the starting frame number for loading]) \
        (--stop [the ending frame number for loading]) \
        (--interval [Loading interval of trajectory frames]) \
        (--process [number of parallel processes])
```

### MPI-based version ###
```text
$ mpiexec -n [number of processes] python trajectory_descriptor_mpi.py medium \
        [trajectory file] \
        [mol2 file] \
        [Interaction target molecule specification file] \
        [van der waals radius definition file] \
        [interaction criteria file] \
        [interaction priority file] \
        [prefix of output file name] \
        (--allow_mediate_position [number]) \
        (--on_14) \
        (--dup) \
        (--no_mediate) \
        (--out_raw) \
        (--out_count) \
        (--out_count_traj) \
        (--out_one_hot) \
        (--out_pml) \
        (--start [the starting frame number for loading]) \
        (--stop [the ending frame number for loading]) \
        (--interval [Loading interval of trajectory frames])
```

---

## Interaction target molecule specification file

This file in YAML format defines the molecular structure to be calculated.

---

## Output files

Typical output files are as follows:

### Raw interaction list
*prefix_frame[N]_raw_list.txt*

Example:
```
Atom1  Atom2  InteractionType  Distance
C12    O45    CH-O             3.2
```

### Interaction count list
*prefix_frame[N]_interaction_count_list.csv*

Example:
```
Interaction,Count
CH-O,15
CH-pi,8
NH-pi,3
```

### One-hot encoded list
*prefix_frame[N]_one_hot_list.csv*

Example:
```
CH-O,CH-pi,NH-pi
1,0,1
```

### PyMOL visualization script
*prefix_frame[N].pml*

- Used for visualization in PyMOL

### Trajectory-level summary
*prefix_trajectory_interaction_count_list.csv*

- Aggregated interaction counts across frames

---

## Notes on MOL2 file format

For large molecular systems (e.g., membrane–solvent systems with a large number of atoms),  
the standard MOL2 format may become misaligned due to column width limitations.

In such cases, please ensure that:
- Atom indices and coordinate columns are properly aligned
- The MOL2 file format is adjusted to avoid column overflow

We internally use a preprocessing script to handle this issue, but it is not currently included in this repository.

---

## Citation

If you use this software, please cite:

(TBD — will be updated upon publication)

---

## Author
Tsutomu Yamane  
RIKEN Yokohama
