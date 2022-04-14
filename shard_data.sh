#!/bin/bash

DATA_DIR="../data/MIMIC/mimic-iii-csv"  # Location of complete dataset
NEW_DATA_DIR="./data"                   # Where to copy the smaller files
PCT_COPY=1/10                           # Amount of data to copy

echo "Original size of data directory"
du -h $DATA_DIR

for file in $DATA_DIR/*.csv; do
    echo "Next file: ${file}";
    lines_in_file=`wc -l < ${file} | xargs`
    copy_lines=$((${lines_in_file} * ${PCT_COPY}));
    fname=`basename ${file}`;
    echo "Copying ${copy_lines} of ${lines_in_file} to ${NEW_DATA_DIR}/${fname}";
    head -${copy_lines} ${file} > ${NEW_DATA_DIR}/${fname};
done

echo "Size of new data directory"
du -h $NEW_DATA_DIR

