# In-Context Learning State Vector with Inner and Momentum Optimization

## Setup

```
conda create -n statev python=3.8
conda activate statev
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Model

- Download Llama-2-7B from [here](https://huggingface.co/meta-llama/Llama-2-7b) and put it under **/llama-2-7B** folder.
- Download GPT-J from [here](https://huggingface.co/EleutherAI/gpt-j-6b) and put it under **/gpt-j-6B** folder.

## Data

We present dataset examples for Antonym, English-French and Person-Instructment task.

- "data" folder contains datasets for inner and momentum optimization.

- "agg_data" folder contains datasets for aggregation.


## Running

Replacing the **cd XXXXX/StateVector** with correct file path in the scripts

### Optimization on llama-2-7B 
```
bash ./script/llama2_optimization.sh
```

### Aggregation on llama-2-7B
```
bash ./script/llama2_aggregation.sh
```

### Optimization on GPT-J
```
bash ./script/gptj_optimization.sh
```

### Aggregation on GPT-J
```
bash ./script/gptj_aggregation.sh
```
