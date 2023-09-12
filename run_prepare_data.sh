#!/bin/bash

RAW_DATA_DIR=data/raw_data/
OUTPUT_DIR=data/DiariST-AliMeeting/

mkdir -p $RAW_DATA_DIR
mkdir -p $OUTPUT_DIR

#
# Download AliMeeting
#
if [ ! -f  $RAW_DATA_DIR/Eval_Ali.tar.gz ]; then
    cd $RAW_DATA_DIR
    wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
    tar -zxvf Eval_Ali.tar.gz
    cd -
fi
if [ ! -f $RAW_DATA_DIR/Test_Ali.tar.gz ]; then
    cd $RAW_DATA_DIR
    wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz
    tar -zxvf Test_Ali.tar.gz
    cd -
fi

#
# Download translation
#
# if [ ! -f $RAW_DATA_DIR/AliMeetingTranslation.tar.gz ]; then
if [ "do " ]; then
    cd $RAW_DATA_DIR
    # wget ali_translation 
    tar -zxvf AliMeetingTranslation.tar.gz
    cd -
fi

#
# Generate DiariST-AliMeeting
#
for data_type in dev test; do
    if [ ! -f $OUTPUT_DIR/$data_type.json ]; then
        python diarist/data/generate_data_json.py \
            $RAW_DATA_DIR/AliMeetingTranslation/$data_type.tsv \
            diarist/data/$data_type.json \
            > $OUTPUT_DIR/$data_type.json
    fi
done

if [ ! -f data/DiariST-AliMeeting/.done ]; then
    python diarist/data/generate_diarist_alimeeting.py $RAW_DATA_DIR || exit 1
    touch data/DiariST-AliMeeting/.done
fi