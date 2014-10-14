#!/bin/bash

a=0
while [ $a -lt 10 ]
do
	echo $a " Run"
	echo "Triaining"
	./nntrain digits.net digits_00_bpnn.pat >> digits_00_output.txt 
	echo "Finished training"
	echo "Testing"
	./nntest digits.net digits_00_bpnn.pat weights.dat >> digits_00_output.txt
	echo "Finished Testing"
	a=`expr $a + 1`
done
a=0
while [ $a -lt 10 ]
do
	echo $a " Run"
	echo "Triaining"
	./nntrain digits.net digits_15_bpnn.pat >> digits_15_output.txt 
	echo "Finished training"
	echo "Testing"
	./nntest digits.net digits_15_bpnn.pat weights.dat >> digits_15_output.txt
	echo "Finished Testing"
	a=`expr $a + 1`
done
a=0
while [ $a -lt 10 ]
do
	echo $a " Run"
	echo "Triaining"
	./nntrain digits.net digits_30_bpnn.pat >> digits_30_output.txt 
	echo "Finished training"
	echo "Testing"
	./nntest digits.net digits_30_bpnn.pat weights.dat >> digits_30_output.txt
	echo "Finished Testing"
	a=`expr $a + 1`
done
a=0
while [ $a -lt 10 ]
do
	echo $a " Run"
	echo "Triaining"
	./nntrain digits.net digits_60_bpnn.pat >> digits_60_output.txt 
	echo "Finished training"
	echo "Testing"
	./nntest digits.net digits_60_bpnn.pat weights.dat >> digits_60_output.txt
	echo "Finished Testing"
	a=`expr $a + 1`
done
