#!/bin/sh

python train.py --flagfile inputs/Pendulum-v1 --seed 11111
python train.py --flagfile inputs/Pendulum-v1 --seed 22222
python train.py --flagfile inputs/Pendulum-v1 --seed 33333
python train.py --flagfile inputs/Pendulum-v1 --seed 44444
python train.py --flagfile inputs/Pendulum-v1 --seed 55555