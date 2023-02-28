#!/bin/sh

python train.py --flagfile inputs/MountainCar-v0 --seed 11111111
python train.py --flagfile inputs/MountainCar-v0 --seed 22222222
python train.py --flagfile inputs/MountainCar-v0 --seed 33333333
python train.py --flagfile inputs/MountainCar-v0 --seed 44444444
python train.py --flagfile inputs/MountainCar-v0 --seed 55555555