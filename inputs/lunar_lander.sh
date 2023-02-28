#!/bin/sh

python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 11111111
python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 22222222
python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 33333333
python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 44444444
python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 55555555