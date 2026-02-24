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

docker_cmd="docker run --rm -v $PWD:$PWD -w $PWD intdesc:mutant"

# Mutant
mut_list=("A_IT001" "A_IT001_2" "A_IT002" "A_IT003")
for id in ${mut_list[@]}
do
    mol2="data/${id}/*.mol2"
    ${docker_cmd} ${mol2} data/${id}/mol_select.yaml ${VDW} ${PARAM} ${PRIORITY} ${dir_name}/${id} 
    ${docker_cmd} ${mol2} data/${id}/mol_select.yaml ${VDW} ${PARAM} ${PRIORITY} ${dir_name}/${id}_dup --dup 
    ${docker_cmd} ${mol2} data/${id}/mol_select.yaml ${VDW} ${PARAM} ${PRIORITY} ${dir_name}/${id}_dup_on14 --dup --on_14 
    ${docker_cmd} ${mol2} data/${id}/mol_select.yaml ${VDW} ${PARAM} ${PRIORITY} ${dir_name}/${id}_on14 --on_14 
done

# Check if the results are identical to pre-calculated results.
cd ${dir_name}
md5sum -c ../data/check.md5
