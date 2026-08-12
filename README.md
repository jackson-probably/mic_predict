# Using antimicrobial susceptibility test values to predict MIC or FIC based on genomic features
The purpose of this pipeline is to generate kmers from genomic datasets, generate large presence/absence data tables from those kmers, and then use them to predict antimicrobial sensitivity based on MIC or FIC.

## **Requirements**
This pipeline is built to be minimal, and can be modified however you want! The only things that are strictly required are
- A directory containing either FASTQ (sequencing reads) or FASTA (assembled sequences/contigs) files
- A corresponding metadata file which contains the testable antimicrobial susceptibility values for the XGBoost model (generally, MIC or FIC values)

This pipeline was built in WSL, and requires Python

## General Workflow
- Use run_kmc.sh to generate txt files containing kmers from each genome provided
- generate_dataframe.py can then be used to combine kmers with each genome as a row and each unique kmer as a column with its corresponding value for each genome
- model.py will run the model, adjust the model parameters as you see fit based on your experimental needs 
