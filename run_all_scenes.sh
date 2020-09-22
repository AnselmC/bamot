while getopts ":f:t:" arg; do
    case $arg in
        f) feature=$OPTARG;;
        t) tags=$OPTARG;;
    esac
done
for scene in {0..19}
do
    echo $scene
done


for scene in {0..19}
do
    python run_kitti_gt_mot.py -s $scene -v INFO -f $feature -c --no-viewer --tags $tags
done

