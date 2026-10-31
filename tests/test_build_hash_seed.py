"""Real interpreter startup checks; no source data/config/build is opened."""
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("flags,rejected", [(["-I"], True), (["-R"], True), (["-s", "-P"], False)])
def test_actual_hash_seed_is_checked_before_config(flags, rejected):
    script = '''
import json,os,sys
from teadata import load_data
class ConfigReached(Exception): pass
def blocked(*args,**kwargs): raise ConfigReached
load_data.load_config=blocked
assert os.environ['PYTHONHASHSEED']=='0'
try:
 load_data.load_repo('must-not-read','must-not-read')
except load_data.ContractError:
 assert sys.flags.hash_randomization==1
 print(json.dumps({'rejected':True}))
except ConfigReached:
 assert sys.flags.hash_randomization==0
 print(json.dumps({'rejected':False}))
else: raise AssertionError('Unexpected build execution')
'''
    result = subprocess.run([sys.executable, *flags, "-c", script],
                            env={**os.environ, "PYTHONHASHSEED": "0"},
                            capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(result.stdout) == {"rejected": rejected}


def test_cli_reexec_uses_an_actual_deterministic_safe_interpreter():
    script = '''
import json,os,runpy,subprocess,sys
from teadata import teadata_config
def blocked(*args,**kwargs): raise AssertionError('Configuration or data opened')
teadata_config.load_config=blocked
def intercept(executable,argv,environment):
 assert sys.flags.hash_randomization==1 and environment['PYTHONHASHSEED']=='0'
 assert argv[1:]==['-s','-P','-m','teadata.load_data']
 assert 'PYTHONPATH' not in environment
 probe='import json,os,sys; print(json.dumps([os.environ["PYTHONHASHSEED"],sys.flags.hash_randomization,sys.flags.no_user_site,sys.flags.safe_path]))'
 child=subprocess.run([executable,*argv[1:3],'-c',probe],env=environment,text=True,capture_output=True,check=True,timeout=30)
 print(child.stdout,end='')
 raise SystemExit(0)
os.execve=intercept
runpy.run_module('teadata.load_data',run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-I", "-c", script],
                            env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "/must-not-import"},
                            capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(result.stdout) == ["0", 0, 1, True]
