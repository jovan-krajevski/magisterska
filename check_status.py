import argparse
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(description="check status of model")
parser.add_argument("modelname", action="store")

modelname = parser.parse_args().modelname

path = Path(".") / "models" / f"{modelname}.pkl"
with open(path, "rb") as f:
    stats, model = pickle.load(f)

print(f"Iterations comppleted: {len(stats)}")
