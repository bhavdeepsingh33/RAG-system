FROM python:3.12

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "st_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# CMD [ "ls" ]