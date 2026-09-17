# Your next step — capture the real setup, without moving it

The 15 motion CSVs are in `proposal/commands`. **Do not run them on the arm yet.**
They are not anchored to your real setup or screened in Newton.

1. Have the robot operator teach a clear free-space pose, peg removed and
   gripper held fixed. Do not run another robot-control client at the same time.
2. On the robot workstation, from this folder, using the reviewed RDK 1.9.3
   installation and Python 3.10+, run:

   ```sh
   python collect_rdk.py snapshot --robot-serial YOUR_ROBOT_SERIAL --output setup_01
   ```

3. Share `setup_01/snapshot.json` and `setup_01/profile.to_review.json`.

This command connects and reads only. It does not move, enable or home the arm.
If your installed RDK is a different version, stop and report the version; do
not upgrade the robot or bypass the version check for this script.

We can then finalize the motion anchor and controller settings, screen the
exact commands in Isaac Lab/Newton and obtain operator approval. No successful
screen or hardware approval is included in this package.
