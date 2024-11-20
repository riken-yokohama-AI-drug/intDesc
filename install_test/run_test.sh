#!/bin/bash

set -eu

CMD="../src/interaction_descriptor.py"
VDW="data/vdw_radius.yaml"
PARAM="data/param.yaml"
PRIORITY="data/priority.yaml"

if [ ! -e data ];then
    unzip data.zip
fi

dir_name="result"
if [ ! -e ${dir_name} ]; then
    mkdir ${dir_name}
fi

# Mutant
mut_list=("A_IT001" "A_IT001_2" "A_IT002" "A_IT003")
for id in ${mut_list[@]}
do
    mol2="data/${id}/*.mol2"
    python ${CMD} mutant ${mol2} data/${id}/mol_select.yaml ${VDW} ${PARAM} ${PRIORITY} ${dir_name}/${id} &
    PID1=$!
    python ${CMD} mutant ${mol2} data/${id}/mol_select.yaml ${VDW} ${PARAM} ${PRIORITY} ${dir_name}/${id}_dup --dup &
    PID2=$!
    python ${CMD} mutant ${mol2} data/${id}/mol_select.yaml ${VDW} ${PARAM} ${PRIORITY} ${dir_name}/${id}_dup_on14 --dup --on_14 &
    PID3=$!
    python ${CMD} mutant ${mol2} data/${id}/mol_select.yaml ${VDW} ${PARAM} ${PRIORITY} ${dir_name}/${id}_on14 --on_14 &
    PID4=$!
    wait $PID1 $PID2 $PID3 $PID4
done

# Check if the results are identical to pre-calculated results.
cd ${dir_name}
md5sum -c ../data/check.md5
