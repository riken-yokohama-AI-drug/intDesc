import argparse
import os
from concurrent import futures

from mol2 import Mol2
from interaction import Interaction
from trajectory import Trajectory


def main(
    interaction_type,
    trajectory,
    mol2,
    molcular_select_file,
    parametar_file,
    vdw_file,
    priority_file,
    water_def_file,
    group_file,
    output,
    **kwargs,
):
    """Frame-level worker function.

    Args:
        interaction_type (str): Execution mode (Lig, Mut, Med).
        trajectory (str): Trajectory file.
        mol2 (str): Structural information file (.mol2).
        molcular_select_file (str): Molecule selection file.
        parametar_file (str): Interaction threshold settings file.
        vdw_file (str): van der Waals radius settings file.
        priority_file (str): Interaction priority settings file.
        water_def_file (str): Water molecule definition file.
        output (str): Output file prefix.
    """

    mol = Mol2(interaction_type=interaction_type)
    mol.read_mol2(mol2_file=mol2)
    mol.add_molcular_type(molcular_select_file=molcular_select_file)

    start = kwargs.get("start", 1)
    stop = kwargs.get("stop", None)
    interval = kwargs.get("interval", 1)
    process = kwargs.get("process", 4)
    out_count = kwargs.get("out_count", False)
    out_count_traj = kwargs.get("out_count_traj", False)

    traj = Trajectory(
        trajectory=trajectory, topology=mol2, start=start, stop=stop, interval=interval
    )
    process_list = []
    csv_list = []
    with futures.ProcessPoolExecutor(max_workers=process) as executor:
        for frame, df in traj.load_frames():
            csv_list.append(f"{output}_frame{frame}_interaction_count_list.csv")
            process_list.append(
                executor.submit(
                    calc,
                    interaction_type=interaction_type,
                    df_frame=df.copy(),
                    df_atom=mol.df_atom.copy(),
                    df_bond=mol.df_bond.copy(),
                    mol_file=mol2,
                    parametar_file=parametar_file,
                    vdw_file=vdw_file,
                    priority_file=priority_file,
                    water_def_file=water_def_file,
                    group_file=group_file,
                    output_prefix=f"{output}_frame{frame}",
                    **kwargs,
                )
            )
        futures.wait(fs=process_list, return_when=futures.FIRST_EXCEPTION)

        traj_total = {}
        for process in process_list:
            process_execption = process.exception()
            if process_execption is not None:
                raise process_execption

            # Aggregate interaction counts over the entire trajectory
            if out_count_traj:
                frame_total = process.result()
                for row in frame_total:
                    traj_total[row[0]] = row[1] + traj_total.get(row[0], 0)

        if out_count_traj:
            with open(f"{output}_trajectory.csv", "w", encoding="utf8") as file:
                file.write(f"flames,{len(process_list)}\n")
                for key, val in traj_total.items():
                    file.write(f"{key},{val}\n")

        if out_count_traj and not out_count:
            for csv_file in csv_list:
                os.remove(csv_file)


def calc(
    interaction_type,
    df_frame,
    df_atom,
    df_bond,
    mol_file,
    parametar_file,
    vdw_file,
    priority_file,
    water_def_file,
    group_file,
    output_prefix,
    **kwargs,
):
    """Frame-level worker function.

    Args:
        interaction_type (str): Execution mode (Lig, Mut, Med).
        df_frame (DataFrame): Coordinate table for a trajectory frame.
        df_atom (DataFrame): Mol object atom table.
        df_bond (DataFrame): Mol object bond table.
        mol_file (str): Structural information file (.mol2).
        molcular_select_file (str): Molecule selection file.
        parametar_file (str): Interaction threshold settings file.
        vdw_file (str): van der Waals radius settings file.
        priority_file (str): Interaction priority settings file.
        water_def_file (str): Water molecule definition file.
        output_prefix (str): Output file prefix.
    """
    no_mediate = kwargs.get("no_mediate", False)
    switch_ch_pi = kwargs.get("switch_ch_pi", False)
    on_14 = kwargs.get("on_14", False)
    dup = kwargs.get("dup", False)
    allow_mediate_position = kwargs.get("allow_mediate_position", None)
    out_raw = kwargs.get("out_raw", False)
    out_one_hot = kwargs.get("out_one_hot", False)
    out_sum = kwargs.get("out_sum", False)
    out_pml = kwargs.get("out_pml", False)

    df_atom[["x", "y", "z"]] = df_frame.loc[:, ["x", "y", "z"]].to_numpy()
    init = Interaction(
        df_atom=df_atom.copy(),
        df_bond=df_bond.copy(),
        interaction_parameter_file=parametar_file,
        vdw_difine_file=vdw_file,
        priority_file=priority_file,
        exec_type=interaction_type,
    )
    del df_frame, df_atom, df_bond
    init.calculate(no_mediate, switch_ch_pi)

    if not on_14:
        init.drop_13_14()

    if not dup:
        # Remove duplicate interactions
        init.drop_duplicate()

    # Remove solvent-mediated interactions
    if allow_mediate_position is not None and no_mediate is False:
        init.drop_mediate_interaction(allow_mediate_position)

    # Write interaction detection results
    if out_raw:
        init.write_interaction(mol_file, output_prefix)

    if interaction_type == "Lig" and out_one_hot:
        init.write_one_hot_list(output_prefix, group_file)

    if interaction_type == "Med" and out_one_hot:
        init.write_one_hot_list_medium(output_prefix, group_file)

    if out_sum:
        init.write_interaction_sum_list(output_prefix, group_file)

    if out_pml:
        init.write_pml(
            output=output_prefix,
            suffix=os.path.basename(output_prefix),
            model_prefix=os.path.basename(mol_file).split(".")[0],
            water_def_file=water_def_file,
        )

    total = init.write_total_interaction(output_prefix, group_file, True)

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="exec_type")
    subparsers.required = True

    for exec_type in ["ligand", "mutant", "medium"]:
        subparser = subparsers.add_parser(exec_type)
        subparser.add_argument("trajectory", help="Trajectory file")
        subparser.add_argument(
            "mol2_file", help="Tripos Mol2 file (.mol2) or directory containing Mol2"
        )
        subparser.add_argument("molcular_select_file", help="Molecule difinication file (.yaml)")
        subparser.add_argument("vdw_file", help="Van Der Waals Radius difinication file (.yaml)")
        subparser.add_argument("parameter_file", help="parameter setting file (.yaml)")
        subparser.add_argument("priority_file", help="priority difinication file (.yaml)")
        subparser.add_argument("output", help="Output file prefix")
        subparser.add_argument("--on_14", help="detect 1-3, 1-4 interaction", action="store_true")
        subparser.add_argument("--dup", help="detect duplicate interactions", action="store_true")
        subparser.add_argument(
            "--allow_mediate_pos",
            default=None,
            type=int,
            help="Position between solvent atoms that "
            "allow detection of solvent-mediated interactions (≧ 1)",
        )
        subparser.add_argument(
            "--no_mediate", help="Not detect solvent-mediated interactions.", action="store_true"
        )
        subparser.add_argument(
            "--switch_ch_pi",
            help="CH_PI, NH_PI, OH_PI Determined by the old definition.",
            action="store_true",
        )
        subparser.add_argument("--out_raw", help="Output Raw list file", action="store_true")
        subparser.add_argument(
            "--out_count", help="Output Interaction count list file", action="store_true"
        )
        subparser.add_argument(
            "--out_count_traj",
            help="Output Interaction count list file (trajectory)",
            action="store_true",
        )
        if exec_type == "ligand":
            subparser.add_argument(
                "--out_one_hot", help="Output One-hot list file", action="store_true"
            )
            subparser.add_argument(
                "--out_sum", help="Output Interaction sum list file", action="store_true"
            )
        if exec_type == "medium":
            subparser.add_argument(
                "--out_one_hot", help="Output One-hot list file", action="store_true"
            )
        subparser.add_argument("--out_pml", help="Output pml file", action="store_true")
        subparser.add_argument("--start", type=int, default=1, help="Start frame (≧ 1)")
        subparser.add_argument("--stop", type=int, default=None, help="End frame (≧ 1)")
        subparser.add_argument(
            "--interval", type=int, default=1, help="loading frame interval (≧ 1)"
        )
        subparser.add_argument(
            "--process", type=int, default=4, help="Number of parallelization (≧ 1)"
        )
    args = parser.parse_args()

    if args.allow_mediate_pos is not None and args.allow_mediate_pos < 1:
        raise ValueError("'--allow_mediate_pos' is 1 or more")
    elif args.start < 1:
        raise ValueError("'--start' is 1 or more")
    elif args.stop is not None and args.stop < 1:
        raise ValueError("'--stop' is 1 or more")
    elif args.stop is not None and args.start > args.stop:
        raise ValueError("'--stop' is more than '--start'")
    elif args.interval < 1:
        raise ValueError("'--interval' is 1 or more")
    elif args.process < 1:
        raise ValueError("'--process' is 1 or more")
    elif not args.out_count and not args.out_count_traj:
        raise ValueError("'--out_count' or '--out_count_traj' or both is not specified")

    water_definition_file = os.path.join(os.path.dirname(__file__), "water_definition.txt")
    interaction_group_file = os.path.join(os.path.dirname(__file__), "group.yaml")

    main(
        interaction_type=str(args.exec_type[0:3]).capitalize(),
        trajectory=args.trajectory,
        mol2=args.mol2_file,
        molcular_select_file=args.molcular_select_file,
        parametar_file=args.parameter_file,
        vdw_file=args.vdw_file,
        priority_file=args.priority_file,
        water_def_file=water_definition_file,
        group_file=interaction_group_file,
        output=args.output,
        allow_mediate_position=args.allow_mediate_pos,
        on_14=args.on_14,
        dup=args.dup,
        no_mediate=args.no_mediate,
        out_raw=args.out_raw,
        out_count=args.out_count,
        out_count_traj=args.out_count_traj,
        out_one_hot=vars(args).get("out_one_hot", False),
        out_sum=vars(args).get("out_sum", False),
        out_pml=args.out_pml,
        switch_ch_pi=args.switch_ch_pi,
        start=args.start,
        stop=args.stop,
        interval=args.interval,
        process=args.process,
    )
