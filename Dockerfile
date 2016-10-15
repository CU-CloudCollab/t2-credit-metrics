FROM  python:2.7

RUN pip install boto3 && \
    pip install Cython && \
    pip install flake8 && \
    pip install ipython && \
    pip install pandas && \
    pip install tabulate

  # Add in our working data
  RUN mkdir /home/tools
  COPY t2-metrics.py /home/tools
  WORKDIR /home/tools
