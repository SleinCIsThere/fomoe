#!/bin/bash
source /opt/intel/oneapi/setvars.sh

export ONEAPI_DEVICE_SELECTOR=level_zero:*
QMOE_PINGPONG=1 QMOE_CAR_THRESHOLD=0.35 QMOE_CAR_WARMUP=1 QMOE_CAR_DAMPEN=1 \
 ./qwen-moe generate --max-tokens 512 --ram-cache 16000 --freq-profile 397b.freq \
 /run/media/niels/990_pro_1tb/Models/Qwen3.5-397B-A17B-UD-Q4_K_M-00001-of-00006.gguf \
 /run/media/niels/990_pro_1tb/qmoe-store/experts-qwen3.5-397b-a17b.qmoe \
 /run/media/niels/990_pro_1tb_2/qmoe-store/experts-qwen3.5-397b-a17b.qmoe \
 -- "What are the capitals of Germany and France? What about other european countries?"
