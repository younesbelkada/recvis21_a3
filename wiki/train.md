# Training WIKI

### Model parameters :

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```name```  | Choice of the model to train. Chose between [```Resnet50```, ```ViT```, ```BirdNet``` and ```TransforBirds```] |
| ```path_out``` | Path to store the trained model |

### General parameters for training:

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```experiment```  | Name of the experiment. The script will create a subfolder inside ```path_out``` with this name |
| ```epochs``` | Number of epochs for training |
| ```batch_size``` | Batch size for training |
| ```learning_rate``` | Learning rate for training |
| ```momentum``` | Nesterov momentum to use |
| ```seed``` | Seed |
| ```log_intervals``` | Intervals for showing the training log |
| ```optimizer``` | Choice of optimizers between [```adam```, ```sgd```] |

### BirdNet parameters :

If you want to train our Aggregation model ```BirdNet```, you have first to pretrain a ResNet50 and a Vit, please specify their paths below:

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```path_resnet```  | Complete path to the trained ResNet50 |
| ```path_vit``` | Complete path to the trained ViT |

### Dataset parameters :

This section has to be modified only if you train your model on a single-dataset configuration. For a multi dataset configuration the parameters in the ```Multi_dataset``` parameters will be considered.

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```path_data``` | Path of the dataset folder |
| ```augment``` | whether to augment the training and validation set using the strategy proposed in the paper. Choice between [```yes```, ```no```] |

### Evaluation parameters

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```output_file``` | Name of the output file name for evaluation. |