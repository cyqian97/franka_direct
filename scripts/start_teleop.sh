source ~/miniconda3/etc/profile.d/conda.sh
conda activate robot
pip install grpcio grpcio-tools protobuf==3.20.1
bash franka_direct/python/generate_stubs.sh
python scripts/simple_teleop_direct_torque.py --cam0 35994006 --cam1 15468057
