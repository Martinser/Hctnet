Hctnet: A new hybrid network based on CNN and ViT for theclassification of pneumonia

Official PyTorch implementation of Hctnet

Pneumonia has become the most frequent disease among children. CNN, Transformer, and hybrid networks have achieved outstanding performance to detect and classify pneumonia. However, there are still some problems. Most of the current hybrid networks simply splice CNN and ViT(Vision Transformer) back and forth. They don't integrate the two networks' most important modules(Conv and MSAs(Mutil Self-Attentions)) at a deeper level which may cause the network not to consider the feature extraction of global and local textures. And the network lacks adaptability in channel and space, which can’t meet the requirements for detecting pneumonia on X-ray images in different patients and environments. We propose a new hybrid network (Hctnet) based on CNN and ViT to classify pneumonia. Hctnet takes Convs and MSAs to cross-combine so that the network can extract features from multiple frequency bands and take the global features and local texture features into account. And it also adds attention mechanism to make the network adaptable in channel and space to solve the problems of imbalanced dataset categories and easy overfitting. Compared with baseline methods, Hctnet’s performance on the test is that Accuracy is 95.35%, Precision is 99.51%, Recall is 88.03, and F1 Score is 93.42%.








