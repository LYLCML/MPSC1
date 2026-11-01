export GPU=2
export alpha=0.5
export omega=0.25
export lr=0.01

python main_ONLRC.py --config config/food101n.cfg --gpu ${GPU} --use-fp16 \
               --net resnet50 --batch-size 96 --lr ${lr} --opt sgd --weight-decay 5e-4 --warmup-lr-scale 1 --warmup-epochs 6 --epochs 50 --epsilon 0.5 \
               --lr-decay step3 \
               --alpha ${alpha} --beta ${beta} --omega ${omega} \
               --loss-func-aux mae --activation relu --classifier mlp-1 \
               --log ONLRC_a${alpha}o${omega}lr${lr} --save-model
