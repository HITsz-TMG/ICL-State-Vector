conda activate statev
cd XXXXX/StateVector
python function_vector_implement.py \
--target_tasks "antonym,english-french,person-instrument" \
--data_root ./data \
--save_path ./result \
--model_path ./llama-2-7b \
--device "0" \
--max_files 1 \
--max_test_num 10
