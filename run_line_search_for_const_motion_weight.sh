for const_motion_weight in {1..20..2}; do
    CONST_MOTION_WEIGHT=$const_motion_weight bash run_all_scenes.sh -t const_motion_weight$const_motion_weight -c motion_config.yaml
done
