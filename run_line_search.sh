for cluster_size in {4..12}; do
    CLUSTER_SIZE=$cluster_size bash run_all_scenes.sh -t cluster_size_$cluster_size
done
