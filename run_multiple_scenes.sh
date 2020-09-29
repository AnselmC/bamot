while getopts ":f:t:s:e:" arg; do
    case $arg in
        t) tags=$OPTARG;;
        s) start=$OPTARG;;
        e) end=$OPTARG;;
    esac
done

for scene in $(seq $start $end)
do
    python run_kitti_gt_mot.py -s $scene -v INFO -c --no-viewer --tags $tags
done

