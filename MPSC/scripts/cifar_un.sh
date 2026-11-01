export GPU=2
export alpha=0.5
export omega=0.25
export lr=0.001


python cifar_ONLRC.py --config config/cifar100.cfg --synthetic-data cifar80no --noise-type symmetric --closeset-noise-ratio 0.2 \
                --gpu ${GPU} --use-fp16 \
                --net cnn --batch-size 128 --lr ${lr} --opt adam --warmup-lr-scale 1 --warmup-epochs 10 --epochs 200 --epsilon 0.6 \
                --lr-decay linear --loss-func-aux mae --activation relu --eta 0.5 \
                --alpha ${alpha} --beta 1 --omega ${omega} --decay_epoch 80 --seed 0 \
                --log ONLRC_a${alpha}bo${omega}lr${lr} \
                --save-model
