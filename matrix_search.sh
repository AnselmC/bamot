git update-index --really-refresh
if ! git diff-index --quiet HEAD; then
	echo "Git tree is dirty"
	exit
fi
while getopts ":t:c:" arg; do
	case $arg in
	t) tags=$OPTARG ;;
	c) config=$OPTARG ;;
	esac
done

for const_motion_weight in {6..14..4}; do
	for cluster_size in {0..10..4}; do
		if [ $cluster_size = 0 ]; then
			for mad_scale_factor in {1..3}; do
				MAD_SCALE_FACTOR=$mad_scale_factor \
				CONST_MOTION_WEIGHT=$const_motion_weight \
				CLUSTER_SIZE=$cluster_size \
				bash \
				run_all_scenes.sh \
				-c $config \
				-t $tags
			done
		else
			CONST_MOTION_WEIGHT=$const_motion_weight \
			CLUSTER_SIZE=$cluster_size \
			bash \
			run_all_scenes.sh \
			-c $config \
			-t $tags
		fi
	done
done
