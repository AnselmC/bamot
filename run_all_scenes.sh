git update-index --really-refresh
if ! git diff-index --quiet HEAD; then
    echo "Git tree is dirty"
    exit
fi
while getopts ":f:t:c:" arg; do
    case $arg in
        f) feature=$OPTARG;;
        t) tags=$OPTARG;;
        c) cluster_size=$OPTARG;;
    esac
done

for scene in {0..19}
do
    python run_kitti_gt_mot.py -s $scene -v INFO -f $feature -c --no-viewer --tags $tags --cluster-size $cluster_size
done

