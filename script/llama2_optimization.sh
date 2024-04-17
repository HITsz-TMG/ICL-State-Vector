conda activate statev
cd XXXXX/StateVector
python state_vector_optimize.py \
--target_tasks "antonym,english-french,person-instrument" \
--data_root ./data \
--save_path ./result \
--model_path ./llama-2-7B \
--device "0" \
--max_files 1 \
--max_test_num 10
