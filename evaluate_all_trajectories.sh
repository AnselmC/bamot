while getopts ":f:t:s:e:" arg; do
    case $arg in
        f) feature=$OPTARG;;
        t) tags=$OPTARG;;
        s) start=$OPTARG;;
        e) end=$OPTARG;;
    esac
done

for scene in $(seq -f "%02g" $start $end)
do
    python evaluate_trajectories.py data/KITTI/tracking/training/trajectories/00$scene/$feature/$tags -s data/evaluation/$feature/
done
python full_evaluation.py data/evaluation/$feature/$tags -s
