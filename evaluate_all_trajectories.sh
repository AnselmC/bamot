while getopts ":f:t:s:e:g:" arg; do
    case $arg in
        f) feature=$OPTARG;;
        t) tags=$OPTARG;;
        s) start=$OPTARG;;
        e) end=$OPTARG;;
        g) use_gt=$OPTARG;;
    esac
done

for scene in $(seq -f "%02g" $start $end)
do
    if [ $use_gt = 1 ]; then
        python evaluate_trajectories.py data/KITTI/tracking/training/trajectories/00$scene/$feature/$tags -s data/evaluation/$feature/ --track-ids-match
    else
        python evaluate_trajectories.py data/KITTI/tracking/training/trajectories/00$scene/$feature/$tags -s data/evaluation/$feature/
    fi
done
python full_evaluation.py data/evaluation/$feature/$tags -s
