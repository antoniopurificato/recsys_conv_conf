# Are Convolutional Sequential Recommender System Models Still Competitive? A Detailed Analysis with New Insights

To run our code follow the next steps:

``mkdir easy_lightning``
- Download this [repo](https://anonymous.4open.science/r/easy_lightning-B93D). Since an anonymous repo can not be cloned, you have to download it on your machine manually. Insert the zip in the folder `easy_lightning` and unzip it.


``cd easy_lightning && unzip easy_lightning-B93D.zip``
- Install the repo just downloaded. Go in the parent directory of `easy_lightning`. If `easy_lightning` is in Desktop, you have to stay in Desktop;

``cd .. && pip3 install --upgrade --force-reinstall easy_lightning/ > /dev/null 2>&1``

- Create the folder `recsys_conv`.

``mkdir recsys_conv``
- Download **this repo**. As before, it can't be cloned. Insert the zip in the folder `recsys_conv`.

- Select the right directory;

``cd recsys_conv && unzip recsys_posneg_conf-CBC6``

- Install the necessary requirements.

``pip3 install -r requirements.txt``
- Download the data;

``cd ntb && bash download_data.bash``

- Run a simple experiment using the jupyter notebook.


``python3 main.py``