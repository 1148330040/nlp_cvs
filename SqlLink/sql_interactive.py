# *- coding: utf-8 -*

import yaml
import pymysql
import pandas as pd

config_path = '../Config/config.yml'

with open(config_path) as f:
    config = yaml.load(f)
    mysql_params = config['mysql_db']


def link_db():
    params = {
        'host': mysql_params['host'],
        'port': mysql_params['port'],
        'db': mysql_params['data_base'],

        'user': mysql_params['user'],
        'password': mysql_params['pass_word'],

        'charset': 'UTF8MB4',
    }

    return params


def get_sql_cvs_dataset():
    params = link_db()

    db = pymysql.connect(**params)

    with db.cursor() as cursor:
        sql = "SELECT * FROM tang_cvs.test_data;"
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            dataset = pd.DataFrame(result, columns=['content', 'label', 'content_length',
                                                    'label_length', 'consistent_len'])
            return dataset

        except:
            raise ValueError(
                "Please check the database : mysql.tang_cvs.cvs_dataset"
            )


def get_sql_QA_dataset(dataset_name):
    params = link_db()
    db = pymysql.connect(**params)
    columns = ['industry', 'question_type', 'process', 'answer', 'process_type']

    with db.cursor() as cursor:
        sql = f"SELECT * FROM tang_cvs.{dataset_name};"
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            dataset = pd.DataFrame(result, columns=columns)
            return dataset

        except:
            raise ValueError(
                f"Please check the database : mysql.tang_cvs.{dataset_name}, "
                f"database name need to ds4 or ds5 or ds55"
            )
