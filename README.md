# intDesc-AbMut
intDesc is software for automatic, comprehensive, and precise identification and visualization of various molecular interactions based on the 3D structure. This repository explains intDesc's ability of detecting interactions for residues of interest in protein-protein interactions.

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

## Install and Test
This program can be executed by git-cloning this repository.
To validate the installation, run the provided test:

```text
 cd install_test
 unzip data.zip
 bash run_test.sh
```

The script calculates the interaction descriptors for the provided structures and input parameters, and compares them with precomputed reference results.
If the results match, OK will be printed, indicating that the installation was successful.


## Quick Example (A_IT003)

The installation test case (A_IT003) also serves as a minimal example of intDesc.

In this case, interactions between the antibody (chains L and H) and the antigen (chain O) are analyzed, focusing on residues L-Leu33 and L-Cys23 and their surrounding residues. The computed interaction descriptors are written to the results directory.

For visualization, open data/A_IT003/edited_1_repHOH_addH.mol2 in PyMOL and load results/A_IT003_dup.pml. The .pml file highlights the detected interactions for inspection.

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
  - [interaction priority file] 
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

### Output files
Typical output files are as follows.
  - prefix_raw_list.txt: A list of atom information that involved in the detected interactions.
  - prefix.pml: A pymol script to visualize the detected interactions. This file will be loaded into the pymol session that opens the mol2 file used in the interaction_descriptor.py program.
  - prefix_interaction_count_list.csv:  A list of the number of interactions detected for each individual type of interaction.

## Using Docker 

### Build Docker Image
``` text
$ docker build -t intdesc:mutant -f docker/Dockerfile .
```

### Run Using Docker
``` text
$ docker run --rm \
           -u $(id -u):$(id -g) \
           -v "$(pwd):$(pwd)" \
           -w "$(pwd)" \
           intdesc:mutant \
                 [mol2 file]  \
                 [Interaction target molecule specification file]  \
                 [van der waals radius definition file]  \
                 [interaction criteria file]  \
                 [interaction priority file]  \
                 [prefix of output file name] \
                 (--on_14)  \
                 (--dup)  \
                 (--no_mediate)  \
                 (--no_out_total)  \
                 (--no_out_pml)
```

### Important Notes on File Paths (Docker)

When running Docker with:

``` text
-v "$(pwd):$(pwd)" -w "$(pwd)"
```
only files located in the current working directory and its subdirectories are visible inside the container.

Therefore:

- All input files (e.g., .mol2, .yaml) must be placed in the current directory or in directories beneath it.

- Files located in parent directories (e.g., ../priority.yaml) will not be accessible unless the parent directory is explicitly mounted.


## Citation

```text
@article{Chiba2025,
  doi = {https://doi.org/10.1101/2025.11.18.688974},
  url = {},
  year = {2025},
  month = {November},
  publisher = {},
  author = {Shuntaro Chiba and Masateru Ohta and Tsutomu Yamane and Yasushi Okuno and Mitsunori Ikeguchi},
  title = {intDesc-AbMut: Describing and understanding how antibody mutations impact their environmental interactions},
  journal = {bioRxiv}
}
```





