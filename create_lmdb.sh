TOOL_PATH=~/Downloads/caffe/distribute/bin/

./make_lmdb_index.py \
    ./data/$1/$2/ \
    ./data/$1/$2_hr/ \
    ./data/$1/$2_lr/ \
    ./data/$1/$2.txt

if [[ $? -ne 0 ]]; then
    echo "Index creation has failed."
    exit -1
fi

rm -r ./data/$1/$2.lmdb

$TOOL_PATH/convert_imageset.bin \
    ./data/$1/$2_hr/ \
    ./data/$1/$2.txt \
    ./data/$1/$2_hr.lmdb

$TOOL_PATH/convert_imageset.bin \
    ./data/$1/$2_lr/ \
    ./data/$1/$2.txt \
    ./data/$1/$2_lr.lmdb