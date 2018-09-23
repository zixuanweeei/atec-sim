#/bin/bash
kfold=5
filename=$1
filename=${filename##*/}
filename=${filename%.*}
cp $1 ./data/$filename
echo $filename
python src/preprocessing.py $filename --filename $filename --task predict
python src/convert.py --task test --filename $filename
for ((i = 0; i < $kfold; i++))
do
    echo ============ Predict Process $i ============
    python src/main.py --task predict_proba --test_data ./data/$filename.tfrecords --result_name $2_$i --model_dir model_ckp_$i
done
python src/ensembling.py --result_pattern $2 --kfold $kfold --result_name $2