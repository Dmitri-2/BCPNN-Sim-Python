#!/bin/bash

biasMultipliers=(0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5)

for i in ${!biasMultipliers[@]}; do
   echo Running ${biasMultipliers[$i]}
   python Simulator.py 5 5 10 250 80 1250 ${biasMultipliers[$i]} no > bias-tests/bias-${biasMultipliers[$i]}
done

