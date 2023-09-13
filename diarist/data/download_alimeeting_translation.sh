#!/bi/bash

outdir=$1

mkdir -p  $outdir/AliMeetingTranslation
cd $outdir/AliMeetingTranslation

curl https://www.microsoft.com/en-us/research/uploads/prod/2023/09/README.txt -o README.txt
curl https://www.microsoft.com/en-us/research/uploads/prod/2023/09/test.txt -o test.txt
curl https://www.microsoft.com/en-us/research/uploads/prod/2023/09/dev.txt -o dev.txt
curl https://www.microsoft.com/en-us/research/uploads/prod/2023/09/train.txt -o train.txt

cd -