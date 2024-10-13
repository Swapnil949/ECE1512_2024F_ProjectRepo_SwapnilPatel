#!/usr/bin/bash

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

#if split directory exists, exit with message that it already exists
if [ -d "$CUR_DIR/images-split" ]; then
    echo "The dataset has already been split. Exiting...";
    exit 0;
fi

echo "Current directory: $CUR_DIR";

cd $CUR_DIR;
md5sum --check MD5SUMs.txt;

echo "Unzipping the dataset...";
unzip -q -n images.zip;

python split_train_test.py