# *- coding: utf-8 -*


import numpy as np
import pandas as pd
import random
import json
from pymemcache.client.base import Client
from SqlLink import sql_interactive

import redis


l = redis.Redis(host='127.0.0.1', port=6379)

# l.set(name='test', value='')
print(l.get('sda') is None)
# # print(json.loads(l.get('test'))['a'])
