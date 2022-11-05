# Content-based audio classification

# How to run experiment
## 1. Install libraries (python>=3.8 is required)
`pip install -r requirements.txt`

## 2. Place dataset
`datasets/audio/` for audio files and
`metadata/label.csv` for labels

![alt text](assets/dataset.png)

## 3. Train model
Select model in Config file mel_spectogram.ini: <br />
**[classifier] - > model_name** <br />
| Available Models  |
| ------------------| 
| alexnet           |
| piczakcnn         |


For training model on main classes  
`python -m src.experiment.frequency_domain_main_class`  

For training model on sub classes  
`python -m src.experiment.frequency_domain_sub_class`

# How to contribute
## Branch rule
Each pull request should follow the branch naming conventions as follows.

`/feature/{feature_name} (e.g. feature/add_lstm)`  
Branch contains new features, additions and optimizations.

`/hotfix/{fix_name} (e.g. hotfix/fix_data_loader)`  
Branch contains hotfix for existing features.


# Appendix
## Existing experiments
Here is the list of the experiments.

| Script                                     | Feature           | Model | Config               | Result        | Result   |Heat Matrix |Run Date |
| -------------------------------------------| ----------------  | ----- | -------------------- | ------------- | -------- |--------    |--------  |
| `frequency_domain_main_class.py`           | Mel-spectrogram   | AlexNet| frequency_domain.ini|![alt text](https://github.com/Filament-lab/local_audio_classification/blob/feature/alexnet_model/assets/model_results/alexnet/Training_Validation_Alexnet_2022-11-05.png)|![alt text](https://github.com/Filament-lab/local_audio_classification/blob/feature/alexnet_model/assets/model_results/alexnet/Training_Validation_Alexnet2_2022-11-05.png)|![alt text](https://github.com/Filament-lab/local_audio_classification/blob/feature/alexnet_model/assets/model_results/alexnet/Training_validation_Alexnet_2022-11-05_HeatMatrix.png)|2022-11-05|
| `frequency_domain_main_class.py`           | Mel-spectrogram   | Piczak CNN| frequency_domain.ini|![alt text](https://github.com/Filament-lab/local_audio_classification/blob/feature/alexnet_model/assets/model_results/piczakcnn/Training_Validation_PiczakCNN_2022-11-05.png)|![alt text](https://github.com/Filament-lab/local_audio_classification/blob/feature/alexnet_model/assets/model_results/piczakcnn/Training_Validation_PiczakCNN2_2022-11-05.png)|![alt text](https://github.com/Filament-lab/local_audio_classification/blob/feature/alexnet_model/assets/model_results/piczakcnn/Training_Validation_PiczakCNN_2022-11-05_HeatMatrix.png)|2022-11-05|







