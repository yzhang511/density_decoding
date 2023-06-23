#!/bin/bash

for roi in po lp dg ca1 vis all
do
python decode_ibl.py --pid dab512bd-a02d-4c1f-8dbc-9155a163efc0 --ephys_path /mnt/3TB/yizi/danlab/Subjects/DY_016/subtract_res_wf_pcs/ --out_path /home/yizi/density_decoding/saved_results --brain_region $roi --behavior choice --max_iter 1000
done


for roi in po lp dg ca1 vis all
do
python decode_ibl.py --pid dab512bd-a02d-4c1f-8dbc-9155a163efc0 --ephys_path /mnt/3TB/yizi/danlab/Subjects/DY_016/subtract_res_wf_pcs/ --out_path /home/yizi/density_decoding/saved_results --brain_region $roi --behavior motion_energy --learning_rate 1e-3 
done



