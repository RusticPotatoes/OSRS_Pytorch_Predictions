# OSRS_Pytorch_Predictions

This is a work in progress, forecasted data and models have not been tested for accuracy. Intended for educational purposes. 

## Training Image
![alt text](https://github.com/RusticPotatoes/OSRS_Pytorch_Predictions/blob/main/resources/elder_maul_training.png)
## Forecast Image
![alt text](https://github.com/RusticPotatoes/OSRS_Pytorch_Predictions/blob/main/resources/elder_maul_forecast.png)
## Price History Image
![alt text](https://github.com/RusticPotatoes/OSRS_Pytorch_Predictions/blob/main/resources/elder_maul_history.png)

## Setup

1. Run the setup.py via the command: 

    ```bash
    pip /path/to/project/setup.py install
    ```
2. Check to see if you have gpu support enabled by running the check_gpu.py, will return true/false: 

    ```bash
    python /path/to/project/check_gpu.py
    ```
3. Add items to the items_to_predict.csv, if you are editing it via a text editor please be sure to add 4 commas: 

    ```
    item name,,,,
    ```
4. Run the pytorch_predictions.py file: 

    ```bash
    python /path/to/project/pytorch_predictions.py
    ```
5. Images will be generated and placed in an img folder located in the root project directory.  This directory includes sub folders for all prices, training, and forecast images. 

## Citations:
### Web
* https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
* https://www.kaggle.com/rodsaldanha/stock-prediction-pytorch
* https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
* https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
### Github
* https://github.com/chriskok/GEPrediction-OSRS
* https://github.com/JonasHogman/osrs-prices-api-wrapper
* https://gist.github.com/adoskk/c3d96e4c7ae15a48c2a9ea8bc835ca39
### Videos
* https://www.youtube.com/watch?v=nNkKTJTu-mU&t=2643s
* https://www.youtube.com/watch?v=D5TmBcpgm7k&list=PLX9loFun2zNmri7jHhLs7NV76wcGRzI45
### Discord
* https://discord.gg/ZummSXK
