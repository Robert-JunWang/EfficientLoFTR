#!/bin/bash -l
SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

main_cfg_path="configs/loftr/eloftr_full.py"

profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1  # per gpu

ckpt_path="weights/eloftr_outdoor.ckpt"

dump_dir="dump/eloftr_full_megadepth"
data_cfg_path="configs/data/megadepth_test_1500.py"
#size="1152"
size="512"
python ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark \
    --megasize $size \
    --npe \
    --thr 0.1 \
    --deter \
    --ransac_times 5
# Following the RoMa protocol, we repeat RANSAC 5 times to enhance robustness; however, this increases script runtime.
