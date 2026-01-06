#!/bin/bash

# $1 ego_num
# $2 carla_port
# $3 route_name
# $4 agent
# $5 config
# $6 save_path
# $7 scenario_type

export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=simulation/leaderboard
export SCENARIO_RUNNER_ROOT=simulation/scenario_runner
export DATA_ROOT=simulation/assets/v2xverse_debug

export YAML_ROOT=simulation/data_collection/yamls
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}/team_code
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}

export ROUTES_DIR=${LEADERBOARD_ROOT}/data/Interdrive/$3

export SAVE_DIR=results
export RESULT_ROOT=${SAVE_DIR}/results_driving_$6
export EVAL_SETTING=v2x_final/${7}_${3}
export CHECKPOINT_ENDPOINT=${RESULT_ROOT}/${EVAL_SETTING}/results.json
export SAVE_PATH=${RESULT_ROOT}/image/${EVAL_SETTING}
mkdir -p $SAVE_PATH
mkdir -p ${RESULT_ROOT}/${EVAL_SETTING}

export TRAFFIC_SEED=2000
export CARLA_SEED=2000
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/no_scenarios.json
export SCENARIOS_PARAMETER=${LEADERBOARD_ROOT}/leaderboard/scenarios/scenario_parameter_$7.yaml
export ROUTES=$ROUTES_DIR
export PORT=$2
export TM_PORT=`expr $PORT + 5`
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export TEAM_AGENT=simulation/leaderboard/team_code/$4.py
export TEAM_CONFIG=simulation/leaderboard/team_code/agent_config/$5.yaml
if [ "$5" == "uniad" ]; then
    echo uniad config
    export TEAM_CONFIG=simulation/assets/UniAD/base_e2e_b2d.py+ckpt/UniAD/uniad_base_b2d.pth
elif [[ "$5" == lmdriver_config_* ]]; then
    echo lmdrive config
    export TEAM_CONFIG=simulation/leaderboard/team_code/agent_config/$5.py
elif [ "$5" == "lmdrive" ]; then
    echo lmdrive config
    export TEAM_CONFIG=simulation/leaderboard/team_code/agent_config/$5.py
fi

if [ "$REALTIME_MODE" == "1" ]; then
    echo "Running in real-time mode"
else
    echo "Running in ideal mode"
fi

export RESUME=0
export EGO_NUM=$1

python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_parameter.py \
--scenarios=${SCENARIOS}  \
--scenario_parameter=${SCENARIOS_PARAMETER}  \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--carlaProviderSeed=${CARLA_SEED} \
--trafficManagerSeed=${TRAFFIC_SEED} \
--ego-num=${EGO_NUM} \
--timeout 600 \
--routes_dir=$ROUTES_DIR \
--skip_existed 1
