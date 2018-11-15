set -x

expname=$1
mkdir "/proj/BigLearning/ahjiang/output/cifar10/$expname/"

NUM_TRIALS=5
for i in `seq 1 $NUM_TRIALS`
do
  outfile="/proj/BigLearning/ahjiang/output/cifar10/$expname/kuangliu_cifar10_resnet_1_128_0.0_0.0005_trial$i_v2"
  python main.py -e=50 --augment --checkpoint=$expname > $outfile
  python main.py --resume -e=50 --augment --checkpoint=$expname >> $outfile
  python main.py --resume -e=50 --augment --checkpoint=$expname >> $outfile
  python main.py --resume --lr=0.01 -e=100 --augment --checkpoint=$expname >> $outfile
  python main.py --resume --lr=0.001 -e=100 --augment --checkpoint=$expname >> $outfile
done
