#!/bin/bash

a=0
while [ $a -lt 10 ]
do
	echo $a " Run"
	echo "Triaining"
	./nntrain XOR.net XOR.pat >> XOR_output.txt 
	echo "Finished training"
	echo "Testing"
	./nntest XOR.net XOR.pat weights.dat >> XOR_output.txt
	echo "Finished Testing"
	a=`expr $a + 1`
done
