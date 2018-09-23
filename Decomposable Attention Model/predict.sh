for ((i = 0; i < 10; i++))
do
    echo ============ Predict Process $i ============
    python src/main.py --task predict_proba --test_data ./data/eval_$i.tfrecords --result_name data/result_$i.csv --model_dir model_ckp_$i
done
python basic_analysis/result_analysis.py