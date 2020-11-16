#!/bin/bash

epochValues=(1 3 5 8 10 12 15 20 25 30 50)

for i in ${!epochValues[@]}; do
   echo Running ${epochValues[$i]}
   python Simulator.py ${epochValues[$i]} 5 10 250 80 1250 1.25 no > epoch-tests/epoch-${epochValues[$i]}
done

