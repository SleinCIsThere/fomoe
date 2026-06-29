#!/bin/bash
source /opt/intel/oneapi/setvars.sh
make clean && make USE_SYCL=1
