# Predicting poverty through time with publicly available data

TLDR: We are using data from Landsat and OpenStreetMap to predict consumption. Consumption is chosen, since we can compare it through time. We are analyzing four African countries, Nigeria, Tanzania, Ethiopia and Malawi. Our approach is capable of predicting consumption through time. The features explains up to 75% of the variation in local-level economic outcomes and for the temporal prediction up to 60%. 

## Running the code

TLDR: Go the [src](src/) dir and follow the steps. 

### Setup
To run the code on your local machine you have to install the requirements file by executing `conda env create -f environment.yml`.

#### Run local

Most of the code can be run on your local machine, you just have to clone the git repository. However, we strongly recommend to execute the [0.1_download_satellite_colab](src/1_feature_generation/0.1_download_satellite_colab.ipynb) and [1.1_cnn colab.ipynb](src/1_feature_generation/1.1_cnn colab.ipynb) on Google Colab. The benefit here is, you directly using tools like [Google Earth Engine](https://earthengine.google.com/) and [Pytorch](https://pytorch.org/), in addition you can use the GPU's to reduce significantly the computation time. Also note that running the CNN requires at least 28GB of RAM.
Librairiews depending on GPU libraries might require some additional steps to install. For example, if you are using a Mac, you might have to install [CUDA](https://developer.nvidia.com/cuda-toolkit-archive). Furthermore the xgboost library might install itself as CPU only, this would heavily slow down computations. To install the GPU version of xgboost, you have to install py-xgboost-gpu packeage. This package is not available on windows and the DLLs libraries should be used.  
#### Google Colab

You can directly run a Jupyter Notebook from a GitHub project. To do so, follow the steps:

1. Open Google Colab and click on "Open the notebook"
2. This opens a box with different tabs, go on GitHub and connect your GitHub account
3. Once connected, you have access to the drop-down menu of your Git repositories. Search the project "predicting-poverty-through-time" and the list of files in the repository will be displayed. 
4. Open the one you want to run 

### LSMS

We are using the surveys of the WorldBank as our true gold standart. You have to download the [LSMS surveys](https://microdata.worldbank.org/index.php/catalog/lsms) more or less manually. However the process is partly automated. You can find the code in [0_lsms_processing](src/0_lsms_processing/). 

- [0_download_country_codes](src/0_lsms_processing/0_download_country_codes.ipynb): Download the country codes for all Sub Saharian African countries from the WorldBank API to use the same country codes.
- [1_check_lsms_availability](src/0_lsms_processing/1_check_lsms_availability.ipynb): Checks the availability of the LSMS for the given countries.
- [2_consent_lsms_form](src/0_lsms_processing/2_consent_lsms_form.ipynb): Poor mans approach to automate the download. The WorldBank requires to fill a consent form and this file does it for us and downloads the survey files for us. You can download our downloaded surveys from [here](https://drive.google.com/file/d/1IlF66tdPrty5OmGdWGd7iN39KZCV-iKD/view?usp=sharing).
- [3_process_surveys](src/0_lsms_processing/3_process_surveys.ipynb): Preprocesses the RAW survey data. Please find the processing steps in [lib/lsms.py](src/lib/lsms.py). 

After running this code you should have processed survey files in [data/lsms/processed](data/lsms/processed).

### Satellite data and features

To download the data please execute the [0_download_satellite.ipynb](src/1_feature_generation/0_download_satellite.ipynb) notebook. However we recommend you to execute it on Google Colab. For this we have a modified [Colab](src/1_feature_generation/0.1_download_satellite_colab.ipynb) of the notebook, which contains all necessary libs. Since you would need to install Earth Engine locally. You also need a [Google Earth Engine account](https://earthengine.google.com/) to execute the code. Researchers, NGO's and country get free access within a short time. You can download the extracted data from [here](https://drive.google.com/file/d/1HJ3Q6BhmcZsRxb-JjhSkL6zH7hoMj1HB/view?usp=sharing).

After you download the data you can train the CNN using [1_cnn.ipynb](src/1_feature_generation/1_cnn.ipynb). Again we recommend to execute it on Colab for this you can use the [colab](src/1_feature_generation/1.1_cnn colab.ipynb) version. If you don't want to train the network from scratch, you can use the [weights](https://drive.google.com/file/d/1Vt6wC4d0qdbyzJlIILPCaf8zWoMbTzGB/view?usp=sharing).

⚠ Caution: The tfrecords need a lot of RAM (over 28GB)! 

### OSM Features 

The OpenStreetMap Features extraction is straight forward, just execute [2_osm](src/1_feature_generation/2_osm.ipynb). 


All the extracted features can be found in the [data](data/) directory. 

### Evaluation 

After you successfully downloaded, processed and extracted everything you can run the models in [2_evaluation](src/2_evaluation).  
- [0_recent_surveys](src/2_evaluation/0_recent_surveys.ipynb): Evaluation of the recent surveys for each country and optimisation of the prediction model, plot of the features importances
- [1_recent_combined](src/2_evaluation/1_recent_combined.ipynb): Evaluation on combined (pooled) features of the most recent surveys of each country.
- [3_time_travel](src/2_evaluation/3_time_travel.ipynb): Evaluation of the prediction through time. 
- [osm_vis_preprocess](src/2_evaluation/osm_vis_preprocess.ipynb): Visualisation of statistics of the OSM features and impact of PCA on CNN weights
- [plot_cluster_performance](src/2_evaluation/plot_cluster_performance.ipynb): Plot of the influence of splitting the data into non overlapping folds on the performance of the models.
- [1.5_cnn_GMM_test](src/2_evaluation/1.5_cnn_GMM_test.ipynb): Test of the impact of the number of GMM components on the CNN features.
The figures generated in by this code are saved in the dir [figs](figs/).
Note that the goal of our project is to optimize the existing model, so we made sure to keep the base model which, in each part of the code, is trained and evaluated before our best model (so we have each time the graphs of the results with a Ridge Regression and CatBoost separately)

### Other figures

The [3_figs](src/3_figs/) contains the code for some figures presented in the original report of Aamir Shakir such as the cluster repartition and consumption distribution.

### [lib folder](src/lib/) 

The lib folder contains code, which used in the notebooks. Please read the code and the comment to understand in depth there function. Here an overview.

- [clusters_util](src/lib/clusters_util.py): Functions that assure the non overlapping of the satellites images between test and train folds
- [estimator_util](src/lib/estimator_util.py): contains the functions such as the ridge regression, data load for the estimation.
- [lsms](src/lib/lsms.py): Class for processing the surveys.
- [satellite_utils](src/lib/satellite_utils.py): Utils for satellite extraction.
- [tfrecordhelper](src/lib/tfrecordhelper.py): Class for processing tfrecords.


## Work In Progress (WIP)

There are still some parts which parts which are Work In Progress, such as the Tutorial part, to make the work more accessible for NGO's and non tech folks also the website is currently in progress.

## Acknowledgements
- [ohsome API](https://github.com/GIScience/ohsome-py): For extracting OSM Features
- [africa-poverty](https://github.com/sustainlab-group/africa_poverty): For satellite extraction code

### Pretrained models
Pretrained models are found under the src folder. CNN models have an pth extension. The others are discriminated by their name. Those files are the result of a simple pickle serialisation and are to be loaded with the same library

## External packages used
- scipy for sparse matrix representation and graph theoretic operation
- xgboost for the xgboost model
- catboost for the catboost model