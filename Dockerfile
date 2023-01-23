FROM public.ecr.aws/lambda/python:3.8

RUN pip install pipenv

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

COPY eye_model.tflite .

RUN pipenv install --system --deploy

COPY ["lambda.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "lambda:app"]