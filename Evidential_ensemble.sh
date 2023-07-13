save_path=/Developer/VCU/McGillResearch/BetaTrial
data_path=/Developer/VCU/McGillResearch/qm9.csv
python train.py \
--data_path $data_path \
--dataset_type regression \
--save_dir $save_path \
--loss_function beta_nll \
--beta 0.5 \

