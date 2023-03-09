# intDesc
 intDesc is software for automatic, comprehensive, and precise identification and visualization of various protein-ligand interactions between ligand, water, and protein based on the 3D structure of the ligand/protein complex.

## Features
- intDesc allows the detailed identification of 65 interactions, such as CH-O, CH-π, NH-π, S-π, CH-F, NH-F, S-O, orthogonal multipolar interactions, and halogen bond. Furthermore, intDesc analyzes ligand-protein interactions via water and can enumerate the number of each interaction and output them as interaction descriptors.
- intDesc is written in python3 and runs on the command line. Each interaction criterion contains user-tunable parameters, which can be changed in the configuration file. 
- intDesc requires ligand/protein complex structure in MOL2 format and ligand ID information as inputs. Moreover, the ligand/protein complex structure requires hydrogens and Gasteiger charges. The ligand ID information is the three-letter code of the ligand written in the MOL2 file.
- intDesc generates four files containing information on interactions (raw list, interaction count list, one-hot list, and interaction sum list) and one PyMOL script file for visualization.
- Please refer to Ohta, M. et al., “Comprehensive and precise identification, visualization, and enumeration of ligand-protein interactions” (Submitted) for the details and the applications. The data discussed in this paper are contained in the directory "Results_of_Paper" in the intDesc repository.


## Environment
intDesc is available in the python 3.9 environment. It also requires the python packages numpy, pyyaml, networkx, and biopandas.

## Usage
Download intDesc.zip, extract the file, and refer to the manual (intDesc/manuals).

## Remarks
- See the manual for the types of interactions analyzed by intDesc.
- A document on the definition of interactions is currently being prepared.
