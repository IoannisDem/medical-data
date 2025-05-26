import json
import boto3
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
import awswrangler as wr
import time
from aws_configs import configs
import sklearn
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
from pathlib import Path


from airflow.sdk import dag, task
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.hooks.base import BaseHook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.operators.s3 import S3DeleteObjectsOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerProcessingOperator

# load env parameters
env_path = Path(os.getcwd()).resolve().parent / '.env'
RAW_DATA_PATH = os.getenv("RAW_DATA")
BUFFER_PATH = os.getenv("BUFFER")
PRE_PROCESS_PATH = os.getenv("PRE_PROCESS")
MODEL_PATH = os.getenv("MODEL_PATH")
REPORT_PATH = os.getenv("REPORT_PATH")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
ROLE_ARN = os.getenv("ROLE_ARN")
REGION = os.getenv("REGION")


# Cache the login
@lru_cache(maxsize=1)
def get_cached_session(conn_id='admin'):
    conn = BaseHook.get_connection(conn_id)
    return boto3.session.Session(
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        region_name=conn.extra_dejson.get("region_name", REGION)
    )

@task.short_circuit
def new_data():
    session = get_cached_session()
    
    s3 = session.client("s3")

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=RAW_DATA_PATH, Prefix="raw-data/")

    all_objects = []
    for page in page_iterator:
        contents = page.get('Contents', [])
        all_objects.extend(contents)

    img_id = [i["Key"] for i in all_objects if i["Key"].endswith(".jpg")] # images in raw bucket

    try:    
        df = wr.s3.read_parquet(
            f"s3://{RAW_DATA_PATH}/data-registry/", 
            boto3_session=session)
        missing_data = list(set(img_id) - set(df["key"]))

    except:
        print("No data stored in Data registry")# no items detected
        missing_data = img_id

    df = pd.DataFrame(
        {
            "bucket": [RAW_DATA_PATH]*len(missing_data),
            "key": missing_data
        }
    )
    if len(missing_data):
        wr.s3.to_csv(
            df, 
            f"s3://{RAW_DATA_PATH}/data.csv", 
            index=False, 
            header=False, 
            boto3_session=session)

        date_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        wr.s3.to_parquet(
            df, 
            f"s3://{RAW_DATA_PATH}/data-registry/data_{date_now}.parquet", 
            index=False, 
            boto3_session=session)

        return True
    
    else:
        return False
        

@task.short_circuit
def buffer_data(csv_bucket, csv_key, target_arn, target_pre, report_arn):
    session = get_cached_session()
    print("Moving data")

    # must fix below else we are getting hacked
    s3control = session.client('s3control', region_name=REGION)
    s3 = session.client("s3", region_name=REGION)
    etag = s3.head_object(Bucket=csv_bucket, Key=csv_key)['ETag']

    account_id = ACCOUNT_ID

    response = s3control.create_job(
        AccountId=account_id,
        ConfirmationRequired=False,
        Operation={
            'S3PutObjectCopy': {
                'TargetResource': target_arn,
                'TargetKeyPrefix': target_pre,
                'MetadataDirective': 'COPY',
            }
        },
        Report={
            'Bucket': report_arn,
            'Prefix': 'report',
            'Enabled': True,
            'Format': 'Report_CSV_20180820',
            'ReportScope': 'AllTasks'
        },
        Manifest={
            'Spec': {
                'Format': 'S3BatchOperations_CSV_20180820',
                'Fields': ['Bucket', 'Key']
            },
            'Location': {
                'ObjectArn': f'arn:aws:s3:::{csv_bucket}/{csv_key}',
                'ETag': etag
            }
        },
        Priority=1,
        RoleArn=ROLE_ARN,
        Description='Batch copy images',
    )

    job_id = response['JobId']
    job_status = s3control.describe_job(
        AccountId=ACCOUNT_ID,
        JobId=job_id)['Job']['Status']
    
    while job_status != "Complete":
        time.sleep(20)
        job_status = s3control.describe_job(
            AccountId=ACCOUNT_ID,
            JobId=job_id)['Job']['Status']
        
        print(job_status)
    return True

@task.short_circuit
def train_test_split_dataset():
    session = get_cached_session()
    
    s3 = session.client("s3")
    try:
        paginator = s3.get_paginator('list_objects_v2')

        page_iterator = paginator.paginate(Bucket=PRE_PROCESS_PATH, Prefix="pre-processed-data/")

        all_objects = []
        for page in page_iterator:
            contents = page.get('Contents', [])
            all_objects.extend(contents)

        img_id = [i["Key"] for i in all_objects if i["Key"].endswith(".jpg")] # images in raw bucket
        df = pd.DataFrame(
            {
                "bucket": [PRE_PROCESS_PATH]*len(img_id),
                "key": img_id
            }
        )
        df["label"] = df["key"].apply(lambda x: x.split("/")[1])
        train, test = train_test_split(df, test_size=0.2, stratify=df["label"])
        
        train = train.drop(["label"], axis=1).reset_index(drop=True)
        test = test.drop(["label"], axis=1).reset_index(drop=True)

        wr.s3.to_csv(
                train, 
                f"s3://{PRE_PROCESS_PATH}/train.csv", 
                index=False, 
                header=False, 
                boto3_session=session)
        
        wr.s3.to_csv(
                test, 
                f"s3://{PRE_PROCESS_PATH}/test.csv", 
                index=False, 
                header=False, 
                boto3_session=session)
        
        return True
    except:
        return False


@task
def sample_task():
    print("Complete")
    return 1


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
def airflow_dag():
    added_data = new_data() # check if new data are added
    move_data_buffer = buffer_data(
        csv_bucket=RAW_DATA_PATH, 
        csv_key="data.csv", 
        target_arn=f"arn:aws:s3:::{BUFFER_PATH}", 
        target_pre="buffer_data", 
        report_arn=f"arn:aws:s3:::{REPORT_PATH}") # move data to buffer

    # check if the data are added in the buffer folder
    wait_for_file = S3KeySensor(
        task_id="wait_for_new_images",
        bucket_name=BUFFER_PATH,
        bucket_key="buffer_data/raw-data/*",
        aws_conn_id="admin",
        poke_interval=5,
        timeout=600,
        wildcard_match=True,
        mode="poke",
    )
    
    preprocess_task_health = SageMakerProcessingOperator(
        task_id="preprocess_raw_data_healthy", 
        config=configs.processing_cnf("Healthy"),
        wait_for_completion=True,
        aws_conn_id="admin")

    preprocess_task_tumor = SageMakerProcessingOperator(
        task_id="preprocess_raw_data_tumor", 
        config=configs.processing_cnf("Tumor"),
        wait_for_completion=True,
        aws_conn_id="admin")

    # next_task = sample_task()

    del_obj_buffer = S3DeleteObjectsOperator(
        task_id="delete_buffer_folder",
        bucket=BUFFER_PATH,
        prefix="buffer_data/raw-data/",
        aws_conn_id="admin",
        trigger_rule=TriggerRule.ALL_DONE)

    create_datasets = train_test_split_dataset()

    del_obj_train = S3DeleteObjectsOperator(
        task_id="delete_train_folder",
        bucket=MODEL_PATH,
        prefix="train/",
        aws_conn_id="admin",
        trigger_rule=TriggerRule.ALL_DONE)
    
    del_obj_test = S3DeleteObjectsOperator(
        task_id="delete_test_folder",
        bucket=MODEL_PATH,
        prefix="test/",
        aws_conn_id="admin",
        trigger_rule=TriggerRule.ALL_DONE)

    move_data_train = buffer_data(
        csv_bucket=PRE_PROCESS_PATH, 
        csv_key="train.csv", 
        target_arn=f"arn:aws:s3:::{MODEL_PATH}", 
        target_pre="train", 
        report_arn=f"arn:aws:s3:::{REPORT_PATH}"
    ) # move data to buffer

    move_data_test = buffer_data(
        csv_bucket=PRE_PROCESS_PATH, 
        csv_key="test.csv", 
        target_arn=f"arn:aws:s3:::{MODEL_PATH}", 
        target_pre="test", 
        report_arn=f"arn:aws:s3:::{REPORT_PATH}"
    ) # move data to buffer

    wait_for_file
    added_data >> move_data_buffer
    [move_data_buffer, wait_for_file] >> preprocess_task_health
    [move_data_buffer, wait_for_file] >> preprocess_task_tumor
    [preprocess_task_health, preprocess_task_tumor] >> del_obj_buffer >> create_datasets
    create_datasets >> [del_obj_train, del_obj_test]
    [del_obj_train, del_obj_test] >> move_data_train
    [del_obj_train, del_obj_test] >> move_data_test



airflow_dag()
