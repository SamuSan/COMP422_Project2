#!/bin/bash
echo "Triaining"
./nntrain XOR.net XOR.pat
echo "Finished training"
echo "Testing"
./nntest XOR.net XOR.pat weights.dat
echo "Finished Testing"