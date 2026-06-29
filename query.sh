#!/bin/bash
source /opt/intel/oneapi/setvars.sh

export ONEAPI_DEVICE_SELECTOR=level_zero:*
QMOE_PINGPONG=1 QMOE_CAR_THRESHOLD=0.35 QMOE_CAR_WARMUP=0 QMOE_CAR_DAMPEN=1 ./qwen-moe generate --max-tokens 512 --ram-cache 100000 --freq-profile 397b.freq    /run/media/n/64DEAA45DEAA0F7C/Models/Qwen3.5-397B-A17B-UD-Q4_K_M-00001-of-00006.gguf   /run/media/n/64DEAA45DEAA0F7C/qmoe-store/experts-qwen3.5-397b-a17b.qmoe   -- "What are the capitals of Germany and France? What about other european countries?"
