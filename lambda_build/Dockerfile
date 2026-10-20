
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY scripts/ ${LAMBDA_TASK_ROOT}/scripts/
COPY models/ ${LAMBDA_TASK_ROOT}/models/

CMD ["lambda_function.lambda_handler"]
