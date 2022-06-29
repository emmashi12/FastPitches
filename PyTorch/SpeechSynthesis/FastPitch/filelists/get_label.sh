#!/usr/bin/bash

for file in `ls my_corpus/`; do
	if [[ $file == *.prom ]]
	then
          python get_boundary_label.py my_corpus/${file}
	fi
done

