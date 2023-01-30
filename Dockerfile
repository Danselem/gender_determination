FROM python:3.9-slim

RUN pip install pipenv

RUN pip install tflite-runtime

# RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

COPY eye_model.tflite .

RUN pipenv install --system --deploy

COPY ["lambda.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "lambda:app"]