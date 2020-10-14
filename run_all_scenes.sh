git update-index --really-refresh
if ! git diff-index --quiet HEAD; then
	echo "Git tree is dirty"
	exit
fi
while getopts ":f:t:c:s:e:" arg; do
	case $arg in
	t) tags=$OPTARG ;;
	c) config=$OPTARG ;;
	s) start=$OPTARG ;;
	e) end=$OPTARG ;;
	esac
done

if [ -z $config ]; then
	config="config.yaml"
fi

if [ -z $start ]; then
	start=0
fi

if [ -z $end ]; then
	end=20
fi

for scene in $(seq $start $end); do
	CONFIG_FILE=$config \
	SCENE=$scene \
	python run_kitti_gt_mot.py -s $scene -v INFO -c --no-viewer --tags $tags
done
