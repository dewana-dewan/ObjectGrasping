#!/bin/bash

# change cp to mv when you are doing the real thing, or let eb the same if you want to test for different testing datasets
# it just sorts every fourth batch of four files in a new folder

count=0
siz=4

for i in $(ls *.png *.txt| sort -V);
do
	echo $count
	if [ $((((count / siz)) % siz)) = "0" ]
	then
		# echo $count;
		# echo;
		echo $i;
		cp $i ./teste/;
		echo;
	fi
	count=$((count + 1))
done;
