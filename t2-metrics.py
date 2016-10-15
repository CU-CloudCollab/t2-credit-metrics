"""
Working notes/script for cloudwatch t2 metric anlysis

References on namespaces, metrics and dimensions:
    * https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/aws-namespaces.html
    * https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ec2-metricscollected.html
    * https://boto3.readthedocs.io/en/latest/reference/services/cloudwatch.html

To build and run:
    * docker build -t t2-metrics .
    * docker run -it --rm -v ~/.aws:/root/.aws t2-metrics python t2-metrics.py *instance_id*

To run interactive terminal:
    * docker run -it --rm -v ~/.aws:/root/.aws t2-metrics ipython
"""
import datetime
import os
import sys

import boto3
import pandas

from tabulate import tabulate


def main(argv):
    # Settings
    METRICS = ['CPUUtilization', 'CPUCreditUsage', 'CPUCreditBalance']
    N_DAYS_TO_REPORT = 3
    T2_INSTANCE_DEFAULTS = get_t2_defaults()

    # setup AWS environment
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    cw_client = boto3.client('cloudwatch')
    ec2_client = boto3.client('ec2')

    if len(argv) < 2:
        print 'Please specify instance id (e.g., python t2-metrics.py i-1701a887)'
        sys.exit(1)

    # get metadata about requested instance
    requested_instance = argv[1]
    requested_instance_type = get_instance_type(ec2_client, requested_instance)

    # This only makes sense for t2 instances, so check that
    if requested_instance_type not in T2_INSTANCE_DEFAULTS:
        print 'Instance does not appear to be in the t2 family'
        print 'Instance type: %s' % (requested_instance_type)
        sys.exit(1)

    # get instance defaults
    instance_defaults = T2_INSTANCE_DEFAULTS[requested_instance_type]

    # setup filters
    start_time = datetime.datetime.now() - datetime.timedelta(days=N_DAYS_TO_REPORT)
    end_time = datetime.datetime.now()
    hourly_period = 60*60
    daily_period = 60*60*24
    target_dimension_filter = [
        {
            'Name': 'InstanceId',
            'Value': requested_instance
        }
    ]

    # get a merged set of metrics by day + add computed items
    daily_df = get_merged_metrics_df(METRICS, cw_client, target_dimension_filter, daily_period, start_time, end_time)
    daily_df['Credits_Earned_Per_Day'] = instance_defaults['cpu_credits_per_hour'] * 24
    daily_df['Credit_Net'] = daily_df['Credits_Earned_Per_Day'] - daily_df['CPUCreditUsage_Sum']

    # get a merged set of metrics by hour + add computed items
    hourly_df = get_merged_metrics_df(METRICS, cw_client, target_dimension_filter, hourly_period, start_time, end_time)
    hourly_df['Credits_Earned_Per_Hour'] = instance_defaults['cpu_credits_per_hour']
    hourly_df['Credit_Net'] = hourly_df['Credits_Earned_Per_Hour'] - hourly_df['CPUCreditUsage_Sum']

    print ''
    print 'Instance ID: %s' % (requested_instance)
    print 'Instance Type: %s' % (requested_instance_type)
    print 'Credits Earned per Hour: %s' % (instance_defaults['cpu_credits_per_hour'])
    print 'Maximum Credit Balance: %s' % (instance_defaults['maximum_credit_balance'])

    print ''
    print 'Summary by Hour:'
    print tabulate(hourly_df[['CPUUtilization_Average',
                              'CPUCreditUsage_Sum',
                              'Credit_Net',
                              'CPUCreditBalance_Minimum',
                              'CPUCreditBalance_Maximum',
                              ]], headers='keys', tablefmt='psql')

    print ''
    print 'Summary by day:'
    print tabulate(daily_df[['CPUUtilization_Average',
                             'CPUCreditUsage_Sum',
                             'Credit_Net',
                             'CPUCreditBalance_Minimum',
                             'CPUCreditBalance_Maximum',
                             ]], headers='keys', tablefmt='psql')


def get_t2_defaults():
    """
    Build a base dictionary w/ t2 default values (accumulation rate, base cpu, etc)
    Note: Stubbing out in code as it doesn't appear to be available via api
        values from:  https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/t2-instances.html#t2-instances-cpu-credits

    Returns:
        dict. Dictionary indexed by instance type (e.g., t2.nano)
    """

    # values from:  https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/t2-instances.html#t2-instances-cpu-credits
    defaults = {}

    defaults['t2.nano'] = {
        'initial_cpu_credit': 30,
        'cpu_credits_per_hour': 3,
        'base_cpu_performance': .05,
        'maximum_credit_balance': 72
    }

    defaults['t2.micro'] = {
        'initial_cpu_credit': 30,
        'cpu_credits_per_hour': 6,
        'base_cpu_performance': .1,
        'maximum_credit_balance': 144
    }

    defaults['t2.small'] = {
        'initial_cpu_credit': 30,
        'cpu_credits_per_hour': 12,
        'base_cpu_performance': .2,
        'maximum_credit_balance': 288
    }

    defaults['t2.medium'] = {
        'initial_cpu_credit': 60,
        'cpu_credits_per_hour': 24,
        'base_cpu_performance': .4,
        'maximum_credit_balance': 576
    }

    defaults['t2.large'] = {
        'initial_cpu_credit': 60,
        'cpu_credits_per_hour': 36,
        'base_cpu_performance': .6,
        'maximum_credit_balance': 864
    }

    return defaults


def get_merged_metrics_df(metrics, cw_client, dimension_filters, period, start_time, end_time):
    """
    Get a merged dataframe for a set of specified metrics

    Args:
        * metrics (list of str): List of metric names
            see https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ec2-metricscollected.html
        * cw_client (botocore.client.CloudWatch): Boto3 CloudWatch client
        * dimension_filters (list of dict): List of Name/Value pairs for filtering metrics (e.g., instance id)
            see https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ec2-metricscollected.html
        * period (int): Grain of each observation (in seconds - e.g., 60 is one observation per minute)
        * start_time (datetime): Datetime of first observation in series
        * end_time (datetime): Datetime of last observation in series

    Returns:
        * Pandas.DataFrame.  A concatenated dataframe of metrics.  Each measure will be prefixed by metric name.
            e.g., CPUUtilization_Average, CPUUtilization_Sum
    """
    dataframes = []

    for metric in metrics:
        dataframes.append(
            get_ec2_metric_dataframe(
                cw_client,
                metric,
                dimension_filters,
                period,
                start_time,
                end_time
            )
        )

    return pandas.concat(dataframes, axis=1)


def get_ec2_metric_dataframe(cw_client, metric_name, dimension_filters, period, start_time, end_time):
    """
    Get an EC2 metric and convert it to a pandas dataframe w/ timeseries index

    Args:
        * cw_client (botocore.client.CloudWatch): Boto3 CloudWatch client
        * metric_name (str): Cloudwatch MetricName
            see https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ec2-metricscollected.html
        * dimension_filters (list of dict): List of Name/Value pairs for filtering metrics (e.g., instance id)
            see https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ec2-metricscollected.html
        * period (int): Grain of each observation (in seconds - e.g., 60 is one observation per minute)
        * start_time (datetime): Datetime of first observation in series
        * end_time (datetime): Datetime of last observation in series

    Returns:
        pandas.DataFrame.  A Pandas dataframe of requested metrics
    """
    # get the CW response
    metric_response = cw_client.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName=metric_name,
        Dimensions=dimension_filters,
        Period=period,
        Statistics=['Sum', 'Average', 'Minimum', 'Maximum', 'SampleCount'],
        StartTime=start_time,
        EndTime=end_time
    )

    # Extract datapoints and convert to timeseries dataframe
    metric_df = pandas.DataFrame.from_dict(metric_response['Datapoints'])
    metric_df.set_index(pandas.DatetimeIndex(metric_df['Timestamp']), inplace=True)

    # Clean up columns so we can concat the result
    metric_df.rename(columns={
        'Average': '%s_Average' % (metric_name),
        'Maximum': '%s_Maximum' % (metric_name),
        'Minimum': '%s_Minimum' % (metric_name),
        'SampleCount': '%s_SampleCount' % (metric_name),
        'Sum': '%s_Sum' % (metric_name),
        'Unit': '%s_Unit' % (metric_name),
        'Timestamp': '%s_Timestamp' % (metric_name),
        }, inplace=True)

    return metric_df


def get_instance_type(ec2_client, instance_id):
    """
    Lookup the instance type for a given instance id

    Args:
        * ec2_client (botocore.client.EC2): Boto3 EC2 Client Instance
        * instance_id (str): Instance ID to look up (e.g., i-1701a887)

    Returns:
        * Str.  Instance type description (e.g., t2.medium)
    """
    return ec2_client.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]['InstanceType']


if __name__ == "__main__":
    main(sys.argv)
