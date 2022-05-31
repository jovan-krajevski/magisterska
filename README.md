# magisterska

## Installation instructions (Ubuntu)

```
# install python (via miniconda)
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh

# create virtual envs
conda create --name py310 python=3.10
conda activate py310
python -m venv .venv
source .venv/bin/activate

# install requirements
make sync_deps
```

## Running instructions

```
# activate virtual env
source .venv/bin/activate

# install requirements
make sync_deps

# run jupyter
jupyter notebook
```
