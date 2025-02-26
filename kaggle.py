"""
The functions here should only run once on the laptop
"""
# setup 
# web1. https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python
# web2. https://www.kaggle.com/code/donkeys/kaggle-python-api
# from kaggle_secrets import UserSecretsClient
import os, json, subprocess


def init_on_kaggle(username, api_key):
    KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
    os.makedirs(KAGGLE_CONFIG_DIR, exist_ok = True)
    api_dict = {"username":username, "key":api_key}
    with open(f"{KAGGLE_CONFIG_DIR}/kaggle.json", "w", encoding='utf-8') as f:
        json.dump(api_dict, f)
    cmd = f"chmod 600 {KAGGLE_CONFIG_DIR}/kaggle.json"
    output = subprocess.check_output(cmd.split(" "))
    output = output.decode(encoding='UTF-8')
    print(output)

