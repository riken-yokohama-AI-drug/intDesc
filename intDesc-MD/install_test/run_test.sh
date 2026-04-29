#!/bin/bash

set -eu

CMD="../src/trajectory_descriptor.py"
TRJ="data/sample1/cyclic_gly6.dcd"
MOL2="data/sample1/cyclic_gly6.mol2"
MOL="data/sample1/mol_select.yaml"
VDW="data/sample1/vdw_radius.yaml"
PARAM="data/sample1/param.yaml"
PRIORITY="data/sample1/priority.yaml"

# Extract input data
unzip data.zip

# Create output directory
dir_name="result"
if [ ! -e ${dir_name} ]; then
    mkdir ${dir_name}
fi

# Execute intDesc-MD
python ${CMD} medium ${TRJ} ${MOL2} ${MOL} ${VDW} ${PARAM} ${PRIORITY}  ${dir_name}/sample1_test1 --out_count --start 1 --stop 10 --interval 1 --process 2
python ${CMD} medium ${TRJ} ${MOL2} ${MOL} ${VDW} ${PARAM} ${PRIORITY}  ${dir_name}/sample1_test2 --out_raw --out_count --start 1 --stop 10 --interval 1 --process 2
python ${CMD} medium ${TRJ} ${MOL2} ${MOL} ${VDW} ${PARAM} ${PRIORITY}  ${dir_name}/sample1_test3 --out_raw --out_count --out_pml  --start 1 --stop 10 --interval 1 --process 2

# Check output files
cd ${dir_name}
md5sum -c ../data/check.md5
