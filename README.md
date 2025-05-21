# LUNet

## 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```
## 2. Deployment BERT model
Either pip install, or go to hugging Face and download the pre-training weights!(https://huggingface.co/models)
pip install bert-serving-server  
pip install bert-serving-client 

## 3. Prepare data
Partial datasets used in this study are included in the file.The complete dataset is available from the corresponding author (shiiziyang1216@163.com) upon reasonable request. To ensure compliance with privacy regulations and licensing terms, requestors must:
### Provide institutional affiliation credentials
### Submit a signed data use agreement specifying:
Intended research purpose and Non-redistribution commitment

## 4. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 5. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. 

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [TransUNet](https://github.com/Beckschen/TransUNet)
* [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer)
