#!/bin/bash

for (( COUNTER=0; COUNTER<=31; COUNTER+=$1 )); do
    CPUFROM=$COUNTER
    CPUTO=$(($COUNTER + $1 - 1))
    export MKL_DEBUG_CPU_TYPE=5
    export OMP_NUM_THREADS=$1
    numactl --physcpubind=$CPUFROM-$CPUTO "${@:2}"&
done
