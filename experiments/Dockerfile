FROM python:3.10.9

RUN apt-get update && \
    apt-get install -y graphviz libgraphviz-dev pkg-config build-essential

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY *.py /app/

# run streamlit app at 0.0.0.0:8501
CMD streamlit run app.py --server.port 8501 --server.address 0.0.0.0