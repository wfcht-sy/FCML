#!/bin/bash
# Utility script to set Gazebo wind parameters (mean velocity and gust parameters)
# Usage: ./set_wind.sh <MeanX> <MeanY> <MeanZ> <GustX> <GustY> <GustZ>
TOPIC_MEAN="/gazebo/default/mean_wind_cmd"
TOPIC_GUST="/gazebo/default/gust_params_cmd"

MX=${1:-0.0}; MY=${2:-0.0}; MZ=${3:-0.0}
GX=${4:-0.0}; GY=${5:-0.0}; GZ=${6:-0.0}

echo ">>> [Gazebo] Mean=[$MX, $MY, $MZ], Gust=[$GX, $GY, $GZ]"
gz topic -p $TOPIC_MEAN -m "linear_velocity: {x: $MX, y: $MY, z: $MZ}"
gz topic -p $TOPIC_GUST -m "linear_velocity: {x: $GX, y: $GY, z: $GZ}"