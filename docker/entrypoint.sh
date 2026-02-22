#!/bin/bash

function usage() {
#    echo 'usage: docker run --rm -u [UID]:[GID] -v $(pwd):$(pwd) -w $(pwd) intdesc:mutant [-h] [-v VDW_FILE] [-t PARAMETER_FILE] [-p PRIORITY_FILE]'
#    echo '                                                                                [--on_14] [--dup] [--allow_mediate_pos ALLOW_MEDIATE_POS]'
#    echo '                                                                                [--no_mediate] [--no_out_total] [--no_out_pml]'
#    echo "                                                                                mol2_file molcular_select_file output"
    echo 'usage: docker run --rm -u [UID]:[GID] -v $(pwd):$(pwd) -w $(pwd) intdesc:mutant [-h]'
    echo '                                                                      [VDW_FILE]'
    echo '                                                                      [PARAMETER_FILE]'
    echo '                                                                      [PRIORITY_FILE]'
    echo '                                                                      [--on_14]'
    echo '                                                                      [--dup]'
    echo '                                                                      [--allow_mediate_pos ALLOW_MEDIATE_POS]'
    echo '                                                                      [--no_mediate]'
    echo '                                                                      [--no_out_total]'
    echo '                                                                      [--no_out_pml]'
    echo "                                                                      mol2_file"
    echo "                                                                      molcular_select_file"
    echo "                                                                      output"
    echo ""
    echo "positional arguments:"
    echo "  mol2_file             Tripos Mol2 file (.mol2)"
    echo "  molcular_select_file  Molecular structure specification file (.yaml)"
    echo "  output                Output file prefix"
    echo ""
    echo "optional arguments:"
    echo "  -h, --help            show this help message and exit"
#    echo "  -v VDW_FILE           Van Der Waals Radius difinication file (.yaml)"
#    echo "  -t PARAMETER_FILE     parameter setting file (.yaml)"
#    echo "  -p PRIORITY_FILE      priority difinication file (.yaml)"
    echo "  VDW_FILE              Van Der Waals Radius difinication file (.yaml)"
    echo "  PARAMETER_FILE        parameter setting file (.yaml)"
    echo "  PRIORITY_FILE         priority difinication file (.yaml)"
    echo "  --on_14               detect 1-3, 1-4 interaction"
    echo "  --dup                 detect duplicate interactions"
    echo "  --allow_mediate_pos   ALLOW_MEDIATE_POS"
    echo "                        Position between solvent atoms that allow detection of solvent-mediated interactions (≧ 1)"
    echo "  --no_mediate          Not detect solvent-mediated interactions."
    echo "  --no_out_total        .csv will not be output"
    echo "  --no_out_pml          .pml will not be output"
    exit 1
}

SRC_DIR="/usr/local/src/interaction_descriptor"
#VDW_FILE="${SRC_DIR}/vdw_radius.yaml"
#PARAM_FILE="${SRC_DIR}/param.yaml"
#PRIORITY_FILE="${SRC_DIR}/priority.yaml"
OPTIONS=()

# 引数解析
#args=$(getopt -o v:t:p:h -l on_14,dup,allow_mediate_pos:,no_mediate,no_out_total,no_out_pml,help -- "$@") || exit 1
#args=$(getopt -o v:t:p:h -l on_14,dup,allow_mediate_pos:,no_mediate,no_out_total,no_out_pml,help -- "$@") || exit 1
args=$(getopt -o h -l on_14,dup,allow_mediate_pos:,no_mediate,no_out_total,no_out_pml,help -- "$@") || exit 1
eval "set -- $args"

while [ $# -gt 0 ]; do
    case $1 in
#        -v) VDW_FILE=$2; shift 2 ;;
#        -t) PARAM_FILE=$2; shift 2 ;;
#        -p) PRIORITY_FILE=$2; shift 2 ;;
        --on_14) OPTIONS+=("--on_14"); shift ;;
        --dup) OPTIONS+=("--dup"); shift ;;
        --allow_mediate_pos) OPTIONS+=("--allow_mediate_pos" $2); shift 2 ;;
        --no_mediate) OPTIONS+=("--no_mediate"); shift ;;
        --no_out_total) OPTIONS+=("--no_out_total"); shift ;;
        --no_out_pml) OPTIONS+=("--no_out_pml"); shift ;;
        -h|--help) usage ;;
        --) shift; args=($@); break ;;
        -\?) usage ;;
    esac
done

if [ ${#args[@]} -ne 6 ]
then
    usage
fi

MOL2_FILE="${args[0]}"
SELECT_FILE="${args[1]}"
VDW_FILE="${args[2]}"
PARAM_FILE="${args[3]}"
PRIORITY_FILE="${args[4]}"
OUTPUT="${args[5]}"

python ${SRC_DIR}/interaction_descriptor.py mutant \
                                            ${MOL2_FILE} \
                                            ${SELECT_FILE} \
                                            ${VDW_FILE} \
                                            ${PARAM_FILE} \
                                            ${PRIORITY_FILE} \
                                            ${OUTPUT} \
                                            ${OPTIONS[@]}
