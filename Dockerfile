FROM  python:2.7

# Add in our working data
RUN mkdir /home/tools && \
    mkdir /home/tools/tests

# copy in requirements.txt + install
COPY requirements.txt /home/tools
RUN pip install -r /home/tools/requirements.txt

# Copy in the code
COPY ["t2_metrics.py", ".flake8rc", "lint_code.sh", "run_tests.sh", "/home/tools/"]
COPY tests /home/tools/tests

# set working dir
WORKDIR /home/tools
