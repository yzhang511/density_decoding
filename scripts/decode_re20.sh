#!/bin/bash

pid="7d999a68-0215-4e45-8e6c-879c6ca2b771"
ephys_path="/media/peter/2TB/rs_decoding/7d999a68-0215-4e45-8e6c-879c6ca2b771/"
out_path="/home/tianxiao/saved_results"

# load h5 and save as numpy
python h5_to_numpy.py --root_path $ephys_path


# decode choice
echo "\nDecoding binary choices: "
for roi in ca1 dg lp po visa
do
python decode_cavi.py --pid $pid --ephys_path $ephys_path --out_path $out_path --brain_region $roi --max_iter 20 --featurize_behavior 
done
for roi in all
do
python decode_advi.py --pid $pid --ephys_path $ephys_path --out_path $out_path --brain_region $roi --batch_size 1 --max_iter 20 --learning_rate 1e-2 --featurize_behavior 
done


# decode wheel speed
echo "\nDecoding wheel speed: "
for roi in ca1 dg lp po visa all
do
python decode_advi.py --pid $pid --ephys_path $ephys_path --out_path $out_path --behavior wheel_speed --brain_region $roi --batch_size 1 --learning_rate 1e-2 --featurize_behavior 
done


# decode motion energy
echo "\nDecoding motion energy: "
for roi in ca1 dg lp po visa all
do
python decode_advi.py --pid $pid --ephys_path $ephys_path --out_path $out_path --behavior motion_energy --brain_region $roi --batch_size 1 --learning_rate 1e-2 --featurize_behavior 
done
