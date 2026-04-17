#!/bin/bash

set -eu

CMD="../trajectory_descriptor.py"
TRJ="data/sample1/cyclic_gly6.dcd"
MOL2="data/sample1/cyclic_gly6.mol2"
MOL="data/sample1/mol_select.yaml"
VDW="data/sample1/vdw_radius.yaml"
PARAM="data/sample1/param.yaml"
PRIORITY="data/sample1/priority.yaml"

# 入力データの解凍
unzip data.zip

# 出力ディレクトリ作成
dir_name="result"
if [ ! -e ${dir_name} ]; then
    mkdir ${dir_name}
fi

# トラジェクトリ
#python ${CMD} medium ${TRJ} ${MOL2} ${MOL} ${VDW} ${PARAM} ${PRIORITY}  ${dir_name}/sample1 --out_raw --out_count --out_one_hot --out_pml  --start 1 --stop 10 --interval 1 --process 2
python ${CMD} medium ${TRJ} ${MOL2} ${MOL} ${VDW} ${PARAM} ${PRIORITY}  ${dir_name}/sample1_test1 --out_count --start 1 --stop 10 --interval 1 --process 2
python ${CMD} medium ${TRJ} ${MOL2} ${MOL} ${VDW} ${PARAM} ${PRIORITY}  ${dir_name}/sample1_test2 --out_raw --out_count --start 1 --stop 10 --interval 1 --process 2
python ${CMD} medium ${TRJ} ${MOL2} ${MOL} ${VDW} ${PARAM} ${PRIORITY}  ${dir_name}/sample1_test3 --out_raw --out_count --out_pml  --start 1 --stop 10 --interval 1 --process 2

# 出力ファイルのチェック
cd ${dir_name}
md5sum -c ../data/check.md5
