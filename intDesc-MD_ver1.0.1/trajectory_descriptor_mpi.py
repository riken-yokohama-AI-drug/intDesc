import argparse
import os
import sys
import warnings
import pandas as pd
import numpy as np
import MDAnalysis as mda
from mpi4py import MPI
from mol2 import Mol2
from interaction import Interaction


def master(
    communicator,
    trajectory,
    topology,
    process,
    start,
    stop,
    interval,
    output,
    out_count,
    out_count_traj,
):
    """Run the master process.
    - Distribute frame indices to worker processes.
    - Aggregate calculation results from all frames.

    Args:
        communicator (comm): Communicator
        interaction_type (str): Operation name (ligand, mutant, medium).
        trajectory (str): Trajectory file.
        topology (str): Topology file (.mol2).
        process (int): Number of processes including the master process.
        start (int): First frame number.
        stop (int): Last frame number.
        interval (int): Frame loading interval.
        output (str): Output file prefix.
    """
    warnings.simplefilter("ignore", UserWarning)
    uni = mda.Universe(topology, trajectory, in_memory_step=1)
    total_frame = len(uni.trajectory)
    del uni

    start_idx = start - 1
    stop_idx = total_frame
    if stop is not None and stop < total_frame:
        stop_idx = stop

    if start_idx >= stop_idx:
        raise Exception(f"frame total: {total_frame}")

    target_frames = np.array([idx for idx in range(start_idx, stop_idx, interval)])
    splited_frames = np.array_split(target_frames, process - 1)

    # Assign frame indices to worker processes
    for to_rank in range(1, process):
        data = splited_frames[to_rank - 1]
        req = communicator.isend(data, dest=to_rank, tag=to_rank)
        req.wait()

    # Merge results from worker processes
    if out_count_traj:
        traj_total = {}
        for from_rank in range(1, process):
            req = communicator.irecv(source=from_rank, tag=from_rank)
            frame_idx = req.wait()
            for idx in frame_idx:
                count_file = f"{output}_frame{idx + 1}_interaction_count_list.csv"
                df = pd.read_csv(count_file, header=None)
                frame_total = df.to_numpy().tolist()
                if not out_count:
                    os.remove(count_file)

                for key, val in frame_total:
                    traj_total[key] = val + traj_total.get(key, 0)

        with open(f"{output}_trajectory.csv", "w", encoding="utf8") as file:
            file.write(f"flames,{len(target_frames)}\n")
            for key, val in traj_total.items():
                file.write(f"{key},{val}\n")


def worker(
    communicator,
    worker_rank,
    interaction_type,
    trajectory,
    topology,
    molcular_select_file,
    param_file,
    vdw_file,
    priority_file,
    water_definition_file,
    group_file,
    output,
    **kwargs,
):
    """Run interaction descriptor calculation on a worker process.

    Args:
        communicator (comm): Communicator
        worker_rank (int): Process ID.
        interaction_type (str): Operation name (ligand, mutant, medium).
        trajectory (str): Trajectory file.
        topology (str): Topology file (.mol2).
        molcular_select_file (str): Molecule selection file.
        param_file (str): Parameter file.
        vdw_file (str): vdW definition file.
        priority_file (str): Interaction priority file.
        water_definition_file (str): Water definition file.
        group_file (str): Interaction-group definition file.
        output (str): Output file prefix.
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
    out_count = kwargs.get("out_count", False)
    out_count_traj = kwargs.get("out_count_traj", False)

    mol = Mol2(interaction_type=interaction_type)
    mol.read_mol2(mol2_file=topology)
    mol.add_molcular_type(molcular_select_file=molcular_select_file)

    warnings.simplefilter("ignore", UserWarning)
    uni = mda.Universe(topology, trajectory, in_memory_step=1)

    req = communicator.irecv(source=0, tag=worker_rank)
    frames_idx = req.wait()
    for idx in frames_idx:
        timestep = uni.trajectory[idx]
        frame_no = timestep.frame + 1
        df_frame = pd.DataFrame(timestep.positions, columns=["x", "y", "z"])
        df_atom = mol.df_atom.copy()
        df_atom[["x", "y", "z"]] = df_frame.loc[:, ["x", "y", "z"]].to_numpy()

        init = Interaction(
            df_atom=df_atom,
            df_bond=mol.df_bond.copy(),
            interaction_parameter_file=param_file,
            vdw_difine_file=vdw_file,
            priority_file=priority_file,
            exec_type=interaction_type,
        )
        init.calculate(no_mediate, switch_ch_pi)

        if not on_14:
            init.drop_13_14()

        if not dup:
            # Remove duplicate interactions
            init.drop_duplicate()

        # Remove solvent-mediated interactions
        if allow_mediate_position is not None and no_mediate is False:
            init.drop_mediate_interaction(allow_mediate_position)

        output_prefix = f"{output}_frame{frame_no}"
        if out_raw:
            init.write_interaction(topology, output_prefix)

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
                model_prefix=os.path.basename(topology).split(".")[0],
                water_def_file=water_definition_file,
            )

        init.write_total_interaction(output_prefix, group_file, True)

    # Send data back to the master process
    req = communicator.isend(frames_idx, dest=0, tag=worker_rank)
    req.wait()


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
            "--no_mediate",
            help="Not detect solvent-mediated interactions.",
            action="store_true",
        )
        subparser.add_argument("--out_raw", help="Output Raw list file", action="store_true")
        subparser.add_argument(
            "--out_count",
            help="Output Interaction count list file",
            action="store_true",
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
                "--out_sum",
                help="Output Interaction sum list file",
                action="store_true",
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
    elif not args.out_count and not args.out_count_traj:
        raise ValueError("'--out_count' or '--out_count_traj' or both is not specified")

    water_def_file = os.path.join(os.path.dirname(__file__), "water_definition.txt")
    interaction_group_file = os.path.join(os.path.dirname(__file__), "group.yaml")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if size == 1:
        print("MPI > 1", file=sys.stderr)
        sys.exit(1)

    if rank == 0:
        master(
            communicator=comm,
            trajectory=args.trajectory,
            topology=args.mol2_file,
            process=size,
            start=args.start,
            stop=args.stop,
            interval=args.interval,
            output=args.output,
            out_count=args.out_count,
            out_count_traj=args.out_count_traj,
        )
    else:
        worker(
            communicator=comm,
            worker_rank=rank,
            interaction_type=str(args.exec_type[0:3]).capitalize(),
            trajectory=args.trajectory,
            topology=args.mol2_file,
            molcular_select_file=args.molcular_select_file,
            param_file=args.parameter_file,
            vdw_file=args.vdw_file,
            priority_file=args.priority_file,
            water_definition_file=water_def_file,
            group_file=interaction_group_file,
            output=args.output,
            allow_mediate_position=args.allow_mediate_pos,
            on_14=args.on_14,
            dup=args.dup,
            no_mediate=args.no_mediate,
            out_raw=args.out_raw,
            out_count=args.out_count,
            out_one_hot=vars(args).get("out_one_hot", False),
            out_sum=vars(args).get("out_sum", False),
            out_pml=args.out_pml,
        )
