#!/bi/bash

outdir=$1

mkdir -p  $outdir/AliMeetingTranslation

curl https://www.microsoft.com/en-us/research/uploads/prod/2023/09/readme.txt -o $outdir/AliMeetingTranslation/readme.txt
curl https://www.microsoft.com/en-us/research/uploads/prod/2023/09/test.txt -o $outdir/AliMeetingTranslation/test.txt
curl https://www.microsoft.com/en-us/research/uploads/prod/2023/09/dev.txt -o $outdir/AliMeetingTranslation/dev.txt
curl https://www.microsoft.com/en-us/research/uploads/prod/2023/09/train.txt -o $outdir/AliMeetingTranslation/train.txt
