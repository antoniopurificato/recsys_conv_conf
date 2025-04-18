# Are Convolutional Sequential Recommender Systems Still Competitive? Introducing New Models and Insights

To run our code follow the next steps:

- Download the EasyLightning library:

```
pip3 install  --upgrade --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git
```

- Clone **this repo**:

```
git clone https://github.com/antoniopurificato/recsys_conv_conf.git
```

- Install the necessary requirements.

```
pip3 install -r requirements.txt
```

- Download the data;

```
cd ntb && bash download_data.bash
```

- Run a simple experiment using the following code.

```
python3 main.py
```

You can change the model you are running by selecting `SASRec`, `Caser2`, `CosRec2` in the `+rec_model` field of `cfg/model/model.yaml`


If you want to have a look to the Caser+ and CosRec+ models, look to the directory `new_models`.
