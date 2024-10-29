# Are Convolutional Sequential Recommender Systems Still Competitive? Introducing New Models and Insights

To run our code follow the next steps:

``mkdir easy_lightning``
- Download this [repo](https://anonymous.4open.science/r/easy_lightning-B93D) inside the `easy_lightning` folder. Since an anonymous repo can not be cloned, you have to download it on your machine manually. Insert the zip in the folder `easy_lightning` and unzip it.


``cd easy_lightning && unzip easy_lightning-B93D.zip``
- Install the repo just downloaded. Go in the parent directory of `easy_lightning`. If `easy_lightning` is in Desktop, you have to stay in Desktop;

``cd .. && pip3 install --upgrade --force-reinstall easy_lightning/ > /dev/null 2>&1``

- Create the folder `recsys_conv`.

``mkdir recsys_conv``
- Download **this repo** inside the `recsys_conv` folder. As before, it can't be cloned. Insert the zip in the folder `recsys_conv`.

- Select the right directory;

``cd recsys_conv && unzip recsys_conv_conf-5FE1``

- Install the necessary requirements.

``pip3 install -r requirements.txt``
- Download the data;

``cd ntb && bash download_data.bash``

- Run a simple experiment using the jupyter notebook.


``python3 main.py``


If you want to have a look to the Caser+ and CosRec+ models, look to the directory `new_models`.
