import json
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
import awswrangler as wr
import time
import os
from dotenv import load_dotenv
from pathlib import Path
from aws_configs import configs
import boto3
from airflow.sdk import dag, task
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.hooks.base import BaseHook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.operators.s3 import S3DeleteObjectsOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerProcessingOperator

# load env parameters
env_path = Path(os.getcwd()).resolve().parent / '.env'
MODEL_PATH = os.getenv("MODEL_PATH")
ROLE_ARN = os.getenv("ROLE_ARN")
REGION = os.getenv("REGION")
PYTORCH_IMAGE = os.getenv("PYTORCH_IMAGE")


@lru_cache(maxsize=1)
def get_cached_session(conn_id='admin'):
    conn = BaseHook.get_connection(conn_id)
    return boto3.session.Session(
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        region_name=conn.extra_dejson.get("region_name", REGION)
    )

@task
def training_job():
    session = get_cached_session()
    sagemaker_client = session.client('sagemaker', region_name=REGION)

    pytorch_image_uri = PYTORCH_IMAGE

    date_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    response = sagemaker_client.create_training_job(
        TrainingJobName=f"training-pipeline-{date_now}",
        AlgorithmSpecification={
            'TrainingImage': pytorch_image_uri,
            'TrainingInputMode': 'File',
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=[
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f"s3://{MODEL_PATH}/train/pre-processed-data/",
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'application/x-image',
                'InputMode': 'File',
            },
        ],
        OutputDataConfig={
            'S3OutputPath': f"s3://{MODEL_PATH}/weights/",
        },
        ResourceConfig={
            'InstanceType': 'ml.g4dn.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30,
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600,
        },
        HyperParameters={
            'epochs': '10',
            'batch-size': '32',
            'learning-rate': '0.001',
            'sagemaker_program': 'script.py',
            "sagemaker_submit_directory":f"s3://{MODEL_PATH}/code/source.tar.gz"
        }
        )




default_args = {
    'owner': 'aws-test',
    'retries': 5,
    'retry_delay': timedelta(minutes=10)
}

@dag(
    dag_id='data_processing',
    start_date=datetime(2025, 5, 19),
    schedule="@daily",
    default_args=default_args
)
def training_dag():
    train_model = training_job()

    train_model

training_dag()
