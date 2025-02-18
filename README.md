# Thesis

## Run

Start the finance2.ipynb jupyter notebook (can be started with VS Code or you can start a jupyter server).

## Resources

* [FB Prophet paper](https://peerj.com/preprints/3190/)
* [FB Prophet documentation](https://facebook.github.io/prophet/docs/quick_start.html)
* [FB Prophet implementation from scratch in PyMC3](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/)
* [FB Prophet implementation from scratch in PyMC3 - presentation](https://www.youtube.com/watch?v=mIAeSDcM1zg)
* [Multiplicative seasonality in PyMC - inspired by FB Prophet](https://www.pymc.io/projects/examples/en/2022.01.0/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html)
* [Timeseers repo](https://github.com/MBrouns/timeseers)
* [Timeseers presentation](https://www.youtube.com/watch?v=appLxcMLT9Y)
* [Fitting models to short time series](https://robjhyndman.com/hyndsight/short-time-series/)
* [Fitting models to long time series](https://robjhyndman.com/hyndsight/long-time-series/)
* [Modeling short time series with prior knowledge](https://minimizeregret.com/short-time-series-prior-knowledge)
* [Modeling short time series with prior knowledge - PyMC](https://juanitorduz.github.io/short_time_series_pymc/)

## General idea

* The work is done on daily data
* When modeling time series, in order to use seasonalities, it is recommended you have at least two periods of data if you are going to use a certain seasonality (i.e. if you use yearly seasonality, you need to have at least 2 years of data)
* If a long time series exhibits similar seasonality patterns as a short time series, we can learn the seasonality from the long time series and transfer and fine-tune it on the short time series
* We use 80 years of data points from the SMP500 index to learn yearly seasonality patterns about how the markets behave; we transfer and fine-tune these patterns on multiple 3-month windows from various stocks that are present in the SMP500 index

## Dev setup

To pull trace (through git LFS), navigate to root directory of repo:

```sh
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

To install conda (PyMC recommends using conda because of BLAS issues):

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

To install PyMC and requirements:

```sh
sudo apt install build-essential
conda create -c conda-forge -n pymc_env python=3.12 "pymc>=5.20.1" --file requirements.txt
```

If you want to run sampling on GPU:

```sh
nvidia-smi
# see which cuda is installed with previous command
# and replace cuda12 with installed version
pip install -U "jax[cuda12]"
```
