export GPU=1
export alpha=0.5
export omega=0.25
export lr=0.005


python main_ONLRC_un.py --config config/aircraft_un.cfg --gpu ${GPU} --use-fp16 \
               --net resnet50 --batch-size 16 --lr ${lr} --opt sgd --warmup-lr-scale 1 --warmup-epochs 5 --epochs 25 --epsilon 0.5 \
               --lr-decay cosine \
               --alpha ${alpha} --beta 1 --omega ${omega} \
               --loss-func-aux mae --activation relu --classifier mlp-1 \
               --log ONLRC_un_bird_a${alpha}o${omega}lr${lr} --save-model
