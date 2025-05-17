FROM public.ecr.aws/amazonlinux/amazonlinux

RUN yum install python3 pip -y
RUN pip3 install boto3 pandas requests torchvision

# Set environment variables
ENV input_path=/opt/ml/processing/input
ENV output_path=/opt/ml/processing/output

# copy local files
COPY ./processing/script.py /script.py

CMD ["python3", "script.py"]
