#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"}
: ${BATCH_SIZE:=32}
: ${PHRASES:="phrases/devset10.tsv"}
: ${OUTPUT_DIR:="./output/audio_$(basename ${PHRASES} .tsv)"}
: ${DATASET_PATH:=LJSpeech-1.1}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${PITCH:=false}
: ${ENERGY:=true}
: ${CWT:=false}
: ${CWT_PROM:=false}
: ${CWT_CON:=false}
: ${CWT_3C:=false}
: ${GET_COUNT:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

: ${SPEAKER:=0}
: ${NUM_SPEAKERS:=1}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --waveglow $WAVEGLOW"
ARGS+=" --dataset-path $DATASET_PATH"
ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --n-speakers $NUM_SPEAKERS"
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$PITCH" = "true" ]       && ARGS+=" --pitch-conditioning"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$CWT" = "true" ]         && ARGS+=" --cwt-conditioning"
[ "$CWT_PROM" = "true" ]    && ARGS+=" --cwt-prominence"
[ "$CWT_CON" = "true" ]     && ARGS+=" --cwt-continuous"
[ "$CWT_3C" = "true" ]      && ARGS+=" --cwt-3C"
[ "$GET_COUNT" = "true" ]   && ARGS+=" --get-count"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"

mkdir -p "$OUTPUT_DIR"

python inference.py $ARGS "$@"
