expname="181105_debug"
outfile="/proj/BigLearning/ahjiang/output/cifar10/$expname/kuangliu_cifar10_resnet_1_128_0.0_0.0005_trial1_v2"
python main.py -e=50 > $outfile
python main.py --resume -e=50 >> $outfile
python main.py --resume -e=50 >> $outfile
python main.py --resume --lr=0.01 -e=100 >> $outfile
python main.py --resume --lr=0.001 -e=100 >> $outfile
