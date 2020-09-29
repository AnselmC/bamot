git update-index --really-refresh
if ! git diff-index --quiet HEAD; then
    echo "Git tree is dirty"
    exit
fi
while getopts ":f:t:c:" arg; do
    case $arg in
        t) tags=$OPTARG;;
    esac
done

for scene in {0..19}
do
    python run_kitti_gt_mot.py -s $scene -v INFO -c --no-viewer --tags $tags
done

