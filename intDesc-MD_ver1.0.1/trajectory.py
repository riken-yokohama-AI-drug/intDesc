# [MDAnalysis]
#
#  Copyright (C) 1989, 1991 Free Software Foundation, Inc.
#                        51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
# This software is released under the GNU GENERAL PUBLIC LICENSE Version 2
# see https://github.com/MDAnalysis/mdanalysis/blob/develop/LICENSE

import warnings
import pandas as pd
import MDAnalysis as mda


class Trajectory:
    """Trajectory class."""

    def __init__(self, trajectory, topology, start, stop, interval):
        """Trajectory class.

        Args:
            trajectory (str): Trajectory file.
            topology (str): Mol2 file.
            start (int): First frame to read (start=1,2,...).
            stop (int): Last frame to read (end=1,2,...).
            interval (int): Frame loading interval.
        """
        self.trajectory = trajectory
        self.topology = topology

        # Suppress warnings emitted during trajectory loading
        warnings.simplefilter("ignore", UserWarning)
        self.uni = mda.Universe(self.topology, self.trajectory, in_memory_step=1)

        # Convert start/stop frame numbers to zero-based indices
        if len(self.uni.trajectory) < start:
            raise ValueError(f"{trajectory} is {len(self.uni.trajectory)} frame")
        self._start = start - 1
        self._stop = len(self.uni.trajectory) if stop is None else stop
        self._interval = interval

    def load_frames(self):
        """Load trajectory frames.

        Yields:
            (int, DataFrame): Frame number and coordinate table.
        """
        for timestep in self.uni.trajectory[self._start : self._stop : self._interval]:
            df = pd.DataFrame(timestep.positions, columns=["x", "y", "z"])
            yield timestep.frame + 1, df
