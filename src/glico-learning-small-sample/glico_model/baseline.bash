SHOTS=50
UNLABEL=1
SEED=0
echo " Baseline CIFAR random_erase shot: $SHOTS"
s=" baseline_classification.py --epoch 200 -d wideresnet --augment --data cifar  --fewshot --shot  $SHOTS --unlabeled_shot 10 --seed ${SEED}"
echo $s
python3 $s
echo " Baseline CIFAR random_erase shot: $SHOTS"