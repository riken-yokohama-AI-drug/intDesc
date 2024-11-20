# intDesc
intDesc is software for automatic, comprehensive, and precise identification and visualization of various molecular interactions based on the 3D structure. This repository provides the ability to detect interactions for residues of interest in protein-protein interactions.

## Features
- intDesc allows the detailed identification of numerous interactions, such as CH-O, CH-π, NH-π, S-π, S-O, and dipole interactions. 
- intDesc analyzes interactions mediated by a water molecule
- intDesc enumerates the number of each interaction and output them as interaction descriptors.
- Each interaction criterion contains user-tunable parameters, which can be changed in the configuration file. 
- intDesc requires a structure file in MOL2 format, in which hydrogens and Gasteiger charges are assigned.
- intDesc generates four files containing information on interactions (raw list, interaction count list, one-hot list, and interaction sum list) and one PyMOL script file for visualization.

## Requirements
- python 3.9
- networkx 3.2.1
- numpy 1.23.5
- pandas 1.5.3
- biopandas 0.2.7
- pyyaml

## Install
This program can be executed by git-cloning this repository.
The validation for the installation of this program is as follows:

 cd install_test/
 bash run_test.sh

## How to run intDesc
```text
$ python interaction_descriptor.py mutant ¥  
        [mol2 file] ¥  
        [Interaction target molecule specification file] ¥  
        [van der waals radius definition file] ¥  
        [interaction criteria file] ¥  
        [interaction priority file] ¥  
        [prefix of output file name] ¥  
        (--on_14) ¥  
        (--dup) ¥  
        (--no_mediate) ¥  
        (--no_out_total) ¥  
        (--no_out_pml)  
```

  - [mol2 file] Specify the input mol2 file.
  - [Interaction target molecule specification file] See the following section.
  - [van der waals radius definition file] 
        Specify the vdW radius file (See input/vdw_radius.yaml)
  - [interaction threshold setting file]
        Specify interaction criteria file (See input/param.yaml)
  - [interaction priority file] ¥
        Specify the priority file (See input/priority.yaml)
  - --on_14, Set this option if you wish to detect 1-3, 1-4 interactions,
        among the interactions detected between atoms connected by covalent bonds.
  - --dup, Set this option if you want to detect overlapping interactions between the same heavy atoms.
  - --no_mediate, Set this option if you do not want to detect solvent-mediated interactions.
  - --no_out_total, Set this option if you do not want to output a "total results file”.
  - --no_out_pml, Specify this option if you do not want to output a "visualization file”.

### [Interaction target molecule specification file]
This file in yaml format defines the molecular structure to be calculated.

Format:
```text
[Molecular type name]:
[item]: [value]
```

Example:
```text
mutant_1:
  num: 53
  name: THR
  type: side
  chain: A
mutant_2:
  num: 52
  name: THR
  type: side
  chain: A
solvent:
  name: HOH
antibody:
  chain: [B,A]
antigen:
  chain: [C]
```

In the example, interactions will be detected between the region specified by mutant_[N] item(s) and the region specified by the antibody and antigen items. In the mutant_[N] item, the residue ID, the residue name, and chain ID are given by num, name, and chain items, respectively. If the type item is "side", the target region will be restricted to the side chain. The the type item is "main", the target region will include main chain atoms, names of which are C, N, CA, O, H, HA. See the install_test directory for more examples.

## Citation

```text
@article{Chiba2024,
  doi = {TBD},
  url = {TBD},
  year = {2024},
  month = TBD,
  publisher = {TBD},
  author = {Shuntaro Chiba and Tsutomu Yamane and Yasushi Okuno and Mitsunori Ikeguchi and Masateru Ohta},
  title = {TBD},
  journal = {TBD}
}
```

