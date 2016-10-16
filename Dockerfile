FROM  python:2.7

# Add in our working data
RUN mkdir /home/tools && \
    mkdir /home/tools/tests

# Copy in the code
COPY ["t2_metrics.py", ".flake8rc", "lint_code.sh", "requirements.txt", "run_tests.sh", "/home/tools/"]
COPY tests /home/tools/tests

# set working dir
WORKDIR /home/tools

# install requirements
RUN pip install -r requirements.txt
