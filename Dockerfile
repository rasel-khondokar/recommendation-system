FROM python:3.9.4-buster
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
CMD python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000 --workers 4