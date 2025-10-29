# Face Search Inference Documentation

## Model serving
### Load image from tar
```bash
sudo docker load -i tritonserver-vnd.tar

```

### Run model serving
```bash
sudo docker run --shm-size=1g --gpus all --network host -d tritonserver-vnd:24.11-py3

```

## Infer 
### Setup env
```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt 

```

### Run infer
```bash
python infer.py \
    /path/to/images \
    --triton-url localhost:8001 \
    --detection-batch-size 8 \
    --extraction-batch-size 16


```


## Rerank
I prefer hard cases, with the input being a query embedding and its candidate embeddings returned after searching from the milvus vector database

Instructions to view test_rerank.py, replace query_embedding and candidate_embeddings when running in production