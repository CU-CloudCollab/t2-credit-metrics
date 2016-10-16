FROM  python:2.7

RUN pip install boto3 && \
    pip install Cython && \
    pip install flake8 && \
    pip install ipython && \
    pip install nose && \
    pip install pandas && \
    pip install tabulate

  # Add in our working data
  RUN mkdir /home/tools && \
      mkdir /home/tools/tests

  COPY ["t2_metrics.py", ".flake8rc", "lint_code.sh", "run_tests.sh", "/home/tools/"]
  COPY tests /home/tools/tests

  WORKDIR /home/tools
