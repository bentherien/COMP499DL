UNLABELED=10
SEED=0
SHOTS=10

echo "glico CIFAR100 samples per classt: $SHOTS"
# train

s="train_glico.py --data cifar-10 --rn  ben_test_cifar10_ --d conv --pixel  --z_init rndm --resume  --epoch 202 --noise_proj --dim 512 --seed ${SEED} --shot ${SHOTS} --unlabeled_shot ${UNLABELED}" 
echo $s
python3 $s

sleep 15

# eval

s="evaluation.py -d wideresnet --keyword cifar-100_my_test_10unsuprvised_pixel_classifier_conv_tr_fs_${SHOTS}_ce_noise_proj --is_inter --augment --epoch 200 --data cifar --pretrained --fewshot --shot $SHOTS --unlabeled_shot ${UNLABELED} --loss_method ce --seed ${SEED} --dim 512"
echo $s
python3 $s
