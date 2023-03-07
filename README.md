# Large-batch Training of Language Models

The original README of Megatron-LM is [README\_old.md](https://github.com/anxuthu/megatron/blob/main/README_old.md).

## Setup

### Dataset

#### EC2

Download from s3 (check [M\*EKS Tutorial](https://quip-amazon.com/mb4BAGjU3icv/M-EKS-Tutorial) for the setup).
```
# Wikipedia preprocessed for Megatron-LM. model: 4-layer BERT, T5
aws --profile gluonnlp s3 cp s3://mstar-eks-dev-us-east-2/annnxu/my-bert_text_sentence.bin ./
aws --profile gluonnlp s3 cp s3://mstar-eks-dev-us-east-2/annnxu/my-bert_text_sentence.idx ./
# Wikipedia + BookCorpus preprocessed for Megatron-LM. model: BERT large
aws --profile gluonnlp s3 cp s3://mstar-eks-dev-us-east-2/annnxu/bert_text_sentence.bin ./
aws --profile gluonnlp s3 cp s3://mstar-eks-dev-us-east-2/annnxu/bert_text_sentence.idx ./
# jsonl of BookCorpus before preprocess
aws --profile gluonnlp s3 cp s3://mstar-eks-dev-us-east-2/annnxu/bookcorpus.jsonl ./
# logs, read.py, plot.py to plot figures. ${id} in [2,3,4,5,6,78,9,10,11,12]. It is '78' because I went to ICML on week 7 and 8, so I put the logs of these two weeks together.
aws --profile gluonnlp s3 cp s3://mstar-eks-dev-us-east-2/annnxu/logs_week${id} ./logs_week${id}/ --recursive
```
The Wikipedia dataset is downloaded and preprocessed following Megatron-LM [README\_old.md](https://github.com/anxuthu/megatron/blob/main/README_old.md). The BookCorpus is downloaded from online, concatenated with Wikipedia, and then preprocessed with Megatron-LM in the same way.

Download vocabulary.
```
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt
```

Remember to move all downloaded above to directory ~/data for EC2 instances.

#### EKS

For EKS, specify the data path in the yaml file.

### Environment

#### EC2

I followed [MIST Intern Onboarding Guide](https://quip-amazon.com/0kvOAH2n0ni0/MIST-Intern-Onboarding-Guide) to create EC2 instances.

In each EC2 instance, create conda environment named "p37".
```
conda create -n p37 python=3.7 -y
conda activate p37
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install regex ninja nltk pybind11 -y
```

Install Apex.
```
cd ~
git clone https://github.com/anxuthu/apex.git
cd ~/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Download the codes from my [Weekly Progress](https://quip-amazon.com/2DmfAMlyvGRJ/An-Xu-Progress-Weekly-Presentation-Slides), unzip it, and move it to ~/.

#### EKS

Check [M\*EKS Tutorial](https://quip-amazon.com/mb4BAGjU3icv/M-EKS-Tutorial) for the setup; slack @zhenghuj for any questions regarding EKS.

I have already uploaded the docker image (for BERT large), so that it can be directly specified (747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-eks:annnxu) in the yaml file to be submitted to the EKS cluster.

To create a new docker image, I use an EC2 instance to run the following command (check [M\*EKS Tutorial](https://quip-amazon.com/mb4BAGjU3icv/M-EKS-Tutorial) "Build with your customized docker image" for prior procedures) after downloading the codes.
```
cd ~/megatron
sudo chmod 666 /var/run/docker.sock
DOCKER_BUILDKIT=1 docker build --no-cache -t mstar-eks -f Dockerfile .
docker tag mstar-eks:latest 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-eks:annnxu # replace "annnxu"
docker push 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-eks:annnxu # upload; replace "annnxu"
```

### Node Configuration

For 4-layer BERT and T5, I use EC2 g4dn.12xlarge inctances, each possessing 4 GPUs. The Amazon Machine Image (AMI) is "Deep Learning AMI (Ubuntu 18.04) Version 60.4".

For 4-layer BERT and T5 with tensor parallelism=8 (larger than 4), I use EC2 g4dn.metal instances, each possessing 8 GPUs.

Note: for distributed training with EC2 instances,
* locally run scripts with NNODES=1 and NODE\_RANK=0 first to create the index map, then set NNODES and NODE\_RANK for each instance following the distributed setting.
* make sure micro\_batch\_size x #GPUs <= global\_batch\_size.

For BERT large (24 layers) pre-training, I use the EKS cluster.

### 4-layer BERT

#### Shorter Training Steps

Check [./bert4\_scripts](https://github.com/anxuthu/megatron/tree/main/bert4_scripts), where "lamb" denotes FusedLAMB from Apex, "mylamb" denotes my PyTorch implementation of LAMB, "mylamb2" denotes our first proposed method layer-wise noise. I use one g4dn.12xlarge for each experiment, which should take 3-4 hours. Just run
```
./bert4_scripts/xxxx.sh
```

#### Longer Training Steps

Check [./bert4\_scripts2](https://github.com/anxuthu/megatron/tree/main/bert4_scripts2), where "mylamb3" denotes our method by increasing the learning rate for the embedding weight. I use 4 g4dn.12xlarge for each experiment, which should take 2 hours. Run
```
#lr=0.01 for B=512, 1k, 2k; lr=0.01 * (2 ** 0.5) for B=4k; lr=0.02 for B=8k, 16k.
#for mylamb3, set "--alpha 1.0"
./bert4_scripts2/xxxx.sh $MASTER_ADDR $NNODES $NODE_RANK $lr
```

For tensor parallel = 2, 4 experiments, the training time is in proportional to the data parallelism, so the training time is 4 and 8 hours respectively with 4 g4dn.12xlarge nodes. Run
```
./bert4_scripts2/xxxx_tp.sh $MASTER_ADDR $NNODES $NODE_RANK $TENSOR_PARALLELISM
```

For tensor parallel = 8 experiments, remember to set "GPUS\_PER\_NODE=8" instead. I use 8 g4dn.metal nodes and it takes about 3-4 hours. Run the same script above.
```
./bert4_scripts2/xxxx_tp.sh $MASTER_ADDR $NNODES $NODE_RANK $TENSOR_PARALLELISM
```

### BERT large (24 layers)

Check [./bert24\_yaml](https://github.com/anxuthu/megatron/tree/main/bert24_yaml). First setup the cluster
```
mstarx --profile gluonnlp config --cluster mstar-eks --region us-east-2 # cluster us-east-2
```
Cluster usage can be found in CloudWatch -> Dashboards -> mstar-eks. Job dag can be found in Airflow. The output is in /mnt\_out/annnxu/.

Submit the job to EKS via
```
mstarx --profile gluonnlp submit -f bert24_yaml/xxxx.yaml
```

Each experiment should take about 2 days with 8 p4 nodes. Remember to set node\_num in the yaml file.

For tensor parallelism experiment, add "--tensor-model-parallel-size" argument with 1, 2, 4, or 8 after "pretrain\_bert.py" in the yaml file. Tensor parallelism=4 should take about 8 days.

For 1-B BERT (85 layers), check [./bert85\_yaml](https://github.com/anxuthu/megatron/tree/main/bert85_yaml). Each experiment should take about 12 hours with 4 p4 nodes.

### T5 small (6 layers)
Check [./t5\_scripts](https://github.com/anxuthu/megatron/tree/main/t5_scripts). I ues 4 g4dn.12xlarge for each experiment, which should take 12 hours. Run
```
./t5_scripts/xxxx.sh $MASTER_ADDR $NNODES $NODE_RANK
```

For tensor parallel = 2, 4 experiments, the training time is 12, 24 hours with 8 g4dn.12xlarge nodes. Run
```
./t5_scripts/xxxx_tp.sh $MASTER_ADDR $NNODES $NODE_RANK $TENSOR_PARALLELISM
```

For tensor parallel = 8 experiments, remember to set "GPUS\_PER\_NODE=8" instead. I use 16 g4dn.metal nodes for T5 and it takes about 11 hours. Run the same script above.
```
./t5_scripts/xxxx_tp.sh $MASTER_ADDR $NNODES $NODE_RANK $TENSOR_PARALLELISM
```

### 1-B BERT (85 layers)
Check [./bert85\_yaml](https://github.com/anxuthu/megatron/tree/main/bert85_yaml).

Submit the job to EKS via
```
mstarx --profile gluonnlp submit -f bert24_yaml/xxxx.yaml
```

Each experiment should take about 12 hours with 4 p4 nodes. Remember to set node\_num in the yaml file.

