while getopts ":f:t:s:e:" arg; do
    case $arg in
        f) feature=$OPTARG;;
        t) tags=$OPTARG;;
	s) start=$OPTARG;;
	e) end=$OPTARG;;
    esac
done

for scene in $(seq $start $end)
do
    python run_kitti_gt_mot.py -s $scene -v INFO -f $feature -c --no-viewer --tags $tags
done

