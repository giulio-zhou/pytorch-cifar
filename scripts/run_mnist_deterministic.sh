set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$1
SAMPLING_STRATEGY=$2
BATCH_SIZE=$3
LR=$4
DECAY=$5
MAX_NUM_BACKPROPS=$6
SAMPLING_MIN=$7

NET="lecunn"

EXP_NAME=$EXP_PREFIX"_"$SAMPLING_STRATEGY

mkdir "/proj/BigLearning/ahjiang/output/mnist/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/mnist/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

NUM_TRIALS=1
for i in `seq 1 $NUM_TRIALS`
do
  OUTPUT_FILE="deterministic_mnist_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$LR"_"$DECAY"_trial"$i"_v2"
  PICKLE_PREFIX="deterministic_mnist_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$LR"_"$DECAY"_trial"$i

  echo $OUTPUT_DIR/$OUTPUT_FILE

  python main.py \
    --sb-strategy=deterministic \
    --dataset=mnist \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN \
    --sampling-strategy=$SAMPLING_STRATEGY \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
done
