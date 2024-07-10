import yaml
import os

def loadyaml(path):
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.SafeLoader)

def createFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
