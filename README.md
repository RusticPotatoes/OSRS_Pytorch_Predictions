# OSRS_Pytorch_Predictions

This is a work in progress, forecasted data and models have not been tested for accuracy. Intended for educational purposes. 

## Training Image
![alt text](https://github.com/RusticPotatoes/OSRS_Pytorch_Predictions/blob/main/resources/elder_maul_training.png)
## Forecast Image
![alt text](https://github.com/RusticPotatoes/OSRS_Pytorch_Predictions/blob/main/resources/elder_maul_forecast.png)
## Price History Image
![alt text](https://github.com/RusticPotatoes/OSRS_Pytorch_Predictions/blob/main/resources/elder_maul_history.png)

## Setting Up the Virtual Environment

1. Install `virtualenv` if you haven't already:

    ```bash
    pip3 install virtualenv
    ```

2. Navigate to your project directory and create a virtual environment:

    ```bash
    virtualenv venv
    ```

3. Activate the virtual environment:

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

4. Install the project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. To deactivate the virtual environment when you're done:

    ```bash
    deactivate
    ```

## Project Setup

1. Run the `setup.py` via the command:

    ```bash
    python3 setup.py install --user
    ```

2. Check to see if you have GPU support enabled by running `check_gpu.py`, will return true/false:

    ```bash
    python3 check_gpu.py
    ```

3. Add items to the `items_to_predict.csv`, if you are editing it via a text editor please be sure to add 4 commas:

    ```
    item name,,,,
    ```

4. Run the `pytorch_predictions.py` file:

    ```bash
    python3 pytorch_predictions.py
    ```

5. Images will be generated and placed in an `img` folder located in the root project directory. This directory includes subfolders for all prices, training, and forecast images.

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
