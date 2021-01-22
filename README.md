# aurora_asi
This project downloads and analyzes the aurora all sky imager (ASI) data.

## Installation
Run these shell commands to install the dependencies into a virtual environment and configure the data paths:

```
# cd into the top directory
python3 -m venv env
source env/bin/activate

python3 -m pip install -e .  # (don't forget the .)
#  or 
pip3 install -r requirements.txt

python3 -m aurora_asi init # and answer the prompts.
```