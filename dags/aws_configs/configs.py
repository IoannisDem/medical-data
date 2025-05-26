from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(os.getcwd()).resolve().parent.parent / '.env'
BUFFER_PATH = os.getenv("BUFFER")
PRE_PROCESS_PATH = os.getenv("PRE_PROCESS")
PROCESSING_IMAGE = os.getenv("PROCESSING_IMAGE")
ROLE_ARN = os.getenv("ROLE_ARN")



def processing_cnf(data_class):
    date_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    processing_config ={
        "ProcessingJobName":f'job-{date_now}-{data_class}',
        "ProcessingInputs":[
            {
                'InputName': 'input_data',
                'AppManaged': False,
                'S3Input': {
                    'S3Uri': f"s3://{BUFFER_PATH}/buffer_data/raw-data/{data_class}/",
                    'LocalPath': "/opt/ml/processing/input",
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3CompressionType': 'None'
                },
            },
        ],
        "ProcessingOutputConfig":{
            'Outputs': [
                {
                    'OutputName': 'string',
                    'S3Output': {
                        'S3Uri': f"s3://{PRE_PROCESS_PATH}/pre-processed-data/{data_class}/",
                        'LocalPath': "/opt/ml/processing/output",
                        'S3UploadMode': 'EndOfJob'
                    },
                    'AppManaged': False
                },
            ],
        },

        "ProcessingResources":{
            'ClusterConfig': {
                "InstanceCount": 1,
                "InstanceType": "ml.t3.medium",
                "VolumeSizeInGB": 1,
            }
        },
        "StoppingCondition":{
            'MaxRuntimeInSeconds': 600
        },
        "AppSpecification":{
            'ImageUri': PROCESSING_IMAGE,
        },
        "RoleArn":ROLE_ARN,
    }
    return processing_config