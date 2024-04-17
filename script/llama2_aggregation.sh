conda activate statev
cd XXXXX/StateVector
python aggregation.py \
--target_tasks "antonym,english-french,person-instrument" \
--data_root ./agg_data \
--save_path ./agg_result \
--model_path ./llama-2-7B \
--device "0" \
--max_files 1 \
--max_test_num 10
