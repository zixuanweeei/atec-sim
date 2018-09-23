kfold=10
python src/preprocessing.py  atec_nlp_sim_train.csv atec_nlp_sim_train_add.csv --task kfold --kfold $kfold
for ((i = 0; i < $kfold; i++))
do
    python src/convert.py --task train --filename train_$i &
    python src/convert.py --task train --filename eval_$i &
done
for ((i = 0; i < $kfold; i++))
do
    tensorboard --logdir model_ckp_$i &    
    python src/main.py --task train --training_data data/train_$i.tfrecords --eval_data data/eval_$i.tfrecords --steps 9000 --batch_size 256 --dropout 0.2 --pos_weight 1.0 --learning_rate 0.0001 --model_dir model_ckp_$i
    # python src/main.py --task train --training_data data/train_$i.tfrecords --eval_data data/eval_$i.tfrecords --steps 9000 --batch_size 64 --dropout 0.2 --pos_weight 1.0 --learning_rate 0.0001 --model_dir model_ckp_$i
    # python src/main.py --task train --training_data data/train_$i.tfrecords --eval_data data/eval_$i.tfrecords --steps 10000 --batch_size 64 --dropout 0.0 --pos_weight 1.0 --learning_rate 0.00001 --model_dir model_ckp_$i
    kill -9 $!
done