import argparse
import glob
import os

import warnings

warnings.simplefilter('ignore',FutureWarning)

from mol2 import Mol2
from interaction import Interaction


def calculate(
    exec_type,
    mol2,
    molcular_select_file,
    parametar_file,
    vdw_file,
    priority_file,
    water_definition_file,
    interaction_group_file,
    output,
    allow_mediate_position,
    on_14,
    dup,
    no_mediate,
    no_out_total,
    no_out_pml,
    switch_ch_pi,
):
    """main function

    Args:
        exec_type (str): Execution functions (Lig, Mut, Med)
        mol2 (str): Crystal structure information file (.mol)
        molcular_select_file (str): Molecular structure specification file
        parametar_file (str): interaction threshold setting file
        vdw_file (str): Van Del Waals radius setting file
        priority_file (str): interaction priority setting file
        water_definition_file (str): Water molecule name specification file
        output (str): Output file prefix
        allow_mediate_position (int): Numeric value representing the positional relationship between solvent atoms
        on_14 (bool): True: 1-3, 1-4 Detecting interactions
        dup (bool): True: Allows detection of overlapping interactions between the same heavy atoms
        no_mediate (bool): True: No solvent-mediated interactions detected
        no_out_total (bool): True: Do not output tally result files
        no_out_pml (bool): True: Do not output visualization files
        switch_ch_pi (bool): True: CH_PI, NH_PI, and OH_PI are determined by the old definitions.
    """
    mol2_files = []
    if os.path.isdir(mol2):
        mol2_files = glob.glob(os.path.join(mol2, "flame*.mol2"))

    else:
        mol2_files = [mol2]

    is_flame = True if len(mol2_files) != 1 else False
    trajectory_total = {}
    for file in mol2_files:
        output_prefix = output
        prefix = os.path.basename(file).split(".")[0]
        if is_flame:
            if "/" in output:
                split_path = os.path.split(output)
                output_prefix = f"{prefix}_{split_path[1]}"
                output_prefix = os.path.join(split_path[0], output_prefix)
            else:
                output_prefix = f"{prefix}_{output}"
        mol = Mol2(interaction_type=exec_type)
        mol.read_mol2(mol2_file=file)
        mol.add_molcular_type(molcular_select_file=molcular_select_file)

        init = Interaction(
            df_atom=mol.df_atom,
            df_bond=mol.df_bond,
            interaction_parameter_file=parametar_file,
            vdw_difine_file=vdw_file,
            priority_file=priority_file,
            exec_type=exec_type,
        )
        init.calculate(no_mediate, switch_ch_pi)

        if not on_14:
            init.drop_13_14()

        if not dup:
            # deletion of duplicates
            init.drop_duplicate()

        # Deletion of solvent-mediated interactions
        if allow_mediate_position is not None and no_mediate is False:
            init.drop_mediate_interaction(allow_mediate_position)

        # Interaction detection result output
        init.write_interaction(file, output_prefix)

        if not no_out_total:
            # Interaction aggregation result output
            init.write_total_interaction(output_prefix, interaction_group_file)
            with open(f"{output_prefix}_interaction_count_list.csv", "r", encoding="utf8") as file:
                for row in file.readlines():
                    label = row.split(",")[0]
                    num = int(row.split(",")[1])
                    if label in trajectory_total:
                        trajectory_total[label] += num
                    else:
                        trajectory_total[label] = num

        if not no_out_pml:
            # Interaction visualization file output
            init.write_pml(
                output=output_prefix,
                suffix=os.path.basename(output_prefix),
                model_prefix=prefix,
                water_def_file=water_definition_file,
            )

        if exec_type == "Lig":
            # One-hot list file output
            init.write_one_hot_list(output_prefix, interaction_group_file)

            # Output of Interaction Sum list file
            init.write_interaction_sum_list(output_prefix, interaction_group_file)

    if len(mol2_files) != 1:
        # Output of Interaction tabulation results (trajectory)
        with open(f"{output}_trajectory.csv", "w", encoding="utf8") as file:
            file.write(f"flames,{len(mol2_files)}\n")
            for key, val in trajectory_total.items():
                file.write(f"{key},{val}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="exec_type")
    subparsers.required = True

    # Ligand-Protein Interaction Descriptor Calculations
    ligand_parser = subparsers.add_parser("ligand")
    ligand_parser.add_argument(
        "mol2_file", help="Tripos Mol2 file (.mol2) or directory containing Mol2"
    )
    ligand_parser.add_argument("molcular_select_file", help="Molecule difinication file (.yaml)")
    ligand_parser.add_argument("vdw_file", help="Van Der Waals Radius difinication file (.yaml)")
    ligand_parser.add_argument("parameter_file", help="parameter setting file (.yaml)")
    ligand_parser.add_argument("priority_file", help="priority difinication file (.yaml)")
    ligand_parser.add_argument("output", help="Output file prefix")
    ligand_parser.add_argument("--on_14", help="detect 1-3, 1-4 interaction", action="store_true")
    ligand_parser.add_argument("--dup", help="detect duplicate interactions", action="store_true")
    ligand_parser.add_argument(
        "--allow_mediate_pos",
        default=None,
        type=int,
        help="Position between solvent atoms that "
        "allow detection of solvent-mediated interactions (≧ 1)",
    )
    ligand_parser.add_argument(
        "--no_mediate", help="Not detect solvent-mediated interactions.", action="store_true"
    )
    ligand_parser.add_argument(
        "--no_out_total", help=".csv will not be output", action="store_true"
    )
    ligand_parser.add_argument("--no_out_pml", help=".pml will not be output", action="store_true")
    ligand_parser.add_argument(
        "--switch_ch_pi",
        help="CH_PI, NH_PI, OH_PI Determined by the old definition.",
        action="store_true",
    )

    # Antibody-antigen interaction descriptor calculation
    mutant_parser = subparsers.add_parser("mutant")
    mutant_parser.add_argument(
        "mol2_file", help="Tripos Mol2 file (.mol2) or directory containing Mol2"
    )
    mutant_parser.add_argument("molcular_select_file", help="Molecule difinication file (.yaml)")
    mutant_parser.add_argument("vdw_file", help="Van Der Waals Radius difinication file (.yaml)")
    mutant_parser.add_argument("parameter_file", help="parameter setting file (.yaml)")
    mutant_parser.add_argument("priority_file", help="priority difinication file (.yaml)")
    mutant_parser.add_argument("output", help="Output file prefix")
    mutant_parser.add_argument("--on_14", help="detect 1-3, 1-4 interaction", action="store_true")
    mutant_parser.add_argument("--dup", help="detect duplicate interactions", action="store_true")
    mutant_parser.add_argument(
        "--allow_mediate_pos",
        default=None,
        type=int,
        help="Position between solvent atoms that "
        "allow detection of solvent-mediated interactions (≧ 1)",
    )
    mutant_parser.add_argument(
        "--no_mediate", help="Not detect solvent-mediated interactions.", action="store_true"
    )
    mutant_parser.add_argument(
        "--no_out_total", help=".csv will not be output", action="store_true"
    )
    mutant_parser.add_argument("--no_out_pml", help=".pml will not be output", action="store_true")
    mutant_parser.add_argument(
        "--switch_ch_pi",
        help="CH_PI, NH_PI, OH_PI Determined by the old definition.",
        action="store_true",
    )

    # Interaction Descriptors calculation of medium molecules
    medium_parser = subparsers.add_parser("medium")
    medium_parser.add_argument(
        "mol2_file", help="Tripos Mol2 file (.mol2) or directory containing Mol2"
    )
    medium_parser.add_argument("molcular_select_file", help="Molecule difinication file (.yaml)")
    medium_parser.add_argument("vdw_file", help="Van Der Waals Radius difinication file (.yaml)")
    medium_parser.add_argument("parameter_file", help="parameter setting file (.yaml)")
    medium_parser.add_argument("priority_file", help="priority difinication file (.yaml)")
    medium_parser.add_argument("output", help="Output file prefix")
    medium_parser.add_argument("--on_14", help="detect 1-3, 1-4 interaction", action="store_true")
    medium_parser.add_argument("--dup", help="detect duplicate interactions", action="store_true")
    medium_parser.add_argument(
        "--allow_mediate_pos",
        default=None,
        type=int,
        help="Position between solvent atoms that "
        "allow detection of solvent-mediated interactions (≧ 1)",
    )
    medium_parser.add_argument(
        "--no_mediate", help="Not detect solvent-mediated interactions.", action="store_true"
    )
    medium_parser.add_argument(
        "--no_out_total", help=".csv will not be output", action="store_true"
    )
    medium_parser.add_argument("--no_out_pml", help=".pml will not be output", action="store_true")
    medium_parser.add_argument(
        "--switch_ch_pi",
        help="CH_PI, NH_PI, OH_PI Determined by the old definition.",
        action="store_true",
    )

    args = parser.parse_args()

    if args.allow_mediate_pos is not None and args.allow_mediate_pos < 1:
        raise ValueError("'--allow_mediate_pos' is 1 or more")

    water_definition_file = os.path.join(os.path.dirname(__file__), "water_definition.txt")
    interaction_group_file = os.path.join(os.path.dirname(__file__), "group.yaml")

    calculate(
        exec_type=str(args.exec_type[0:3]).capitalize(),
        mol2=args.mol2_file,
        molcular_select_file=args.molcular_select_file,
        parametar_file=args.parameter_file,
        vdw_file=args.vdw_file,
        priority_file=args.priority_file,
        water_definition_file=water_definition_file,
        interaction_group_file=interaction_group_file,
        output=args.output,
        allow_mediate_position=args.allow_mediate_pos,
        on_14=args.on_14,
        dup=args.dup,
        no_mediate=args.no_mediate,
        no_out_total=args.no_out_total,
        no_out_pml=args.no_out_pml,
        switch_ch_pi=args.switch_ch_pi,
    )
