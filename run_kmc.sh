#!/bin/bash

#define input and output directories

input_dir="./Sequences"
output_dir="./kmc_outputs"
tmp="./kmc_tmp"
k=8

#make directories

mkdir -p "$output_dir"
mkdir -p "$tmp"

#process kmers

for fasta_file in "$input_dir"/*.fasta; do

    base_name=$(basename "$fasta_file")
    base_name_no_ext="${base_name%.*}"


    kmc_db="$tmp/${base_name_no_ext}_kmc_db"
    output_txt="$output_dir/${base_name_no_ext}_kmers.txt"


    kmc -k$k -fm -ci1 -cs1677215 "$fasta_file" "$kmc_db" "$tmp"


    kmc_dump "$kmc_db" "$output_txt"
done

echo "done."
