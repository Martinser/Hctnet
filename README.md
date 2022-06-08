# Hctnet: A new hybrid network based on CNN and ViT for theclassification of pneumonia

Official PyTorch implementation of Hctnet

Pneumonia has become the most frequent disease among children. CNN, Transformer, and hybrid networks have achieved outstanding performance to detect and classify pneumonia. However, there are still some problems. Most of the current hybrid networks simply splice CNN and ViT(Vision Transformer) back and forth. They don't integrate the two networks' most important modules(Conv and MSAs(Mutil Self-Attentions)) at a deeper level which may cause the network not to consider the feature extraction of global and local textures. And the network lacks adaptability in channel and space, which can’t meet the requirements for detecting pneumonia on X-ray images in different patients and environments. We propose a new hybrid network (Hctnet) based on CNN and ViT to classify pneumonia. Hctnet takes Convs and MSAs to cross-combine so that the network can extract features from multiple frequency bands and take the global features and local texture features into account. And it also adds attention mechanism to make the network adaptable in channel and space to solve the problems of imbalanced dataset categories and easy overfitting. Compared with baseline methods, Hctnet’s performance on the test is that Accuracy is 95.35%, Precision is 99.51%, Recall is 88.03, and F1 Score is 93.42%.

**Installation**

Please check INSTALL.md for installation instructions.


**Evaluation**

We give an example evaluation command for a ImageNet-1k pre-trained, then ImageNet-1K fine-tuned Hctnet:


python -m torch.distributed.launch --nproc_per_node=1 main.py \
--model hctnet --eval true \
--resume  \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k

The result should be
Accuracy(95.35%)	Precision(99.51%)	Recall(88.03%)	F1 Score(93.42%)

**Training**

python -m torch.distributed.launch --nproc_per_node=1 main.py \
--model hctnet --drop_path 0.1 \
--batch_size 16 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results

**Acknowledgement**

This repository is built using the timm library, ConvNeXt repositories.

