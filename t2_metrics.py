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

from collections import OrderedDict
from tabulate import tabulate


def main(argv):
    # Settings
    NAMESPACE = 'AWS/EC2'
    METRICS = ['CPUUtilization', 'CPUCreditUsage', 'CPUCreditBalance']
    N_DAYS_TO_REPORT = 3
    HOURLY_PERIOD = 60*60
    DAILY_PERIOD = 60*60*24
    T2_INSTANCE_DEFAULTS = get_t2_defaults()

    # setup AWS environment
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    cw_client = boto3.client('cloudwatch')
    ec2_client = boto3.client('ec2')

    if len(argv) < 2:
        raise ValueError("Please specify instance id (e.g., python t2-metrics.py i-1701a887")

    # get metadata about requested instance
    requested_instance = argv[1]
    requested_instance_type = get_instance_type(ec2_client, requested_instance)
    requested_instance_launch_dt = get_instance_launch_time(ec2_client, requested_instance)

    # This only makes sense for t2 instances, so check that
    if requested_instance_type not in T2_INSTANCE_DEFAULTS:
        raise ValueError("Instance type %s doesn't appear to be in the t2 family" % (requested_instance_type))

    # get instance defaults
    instance_defaults = T2_INSTANCE_DEFAULTS[requested_instance_type]

    # setup filters
    start_time = datetime.datetime.now() - datetime.timedelta(days=N_DAYS_TO_REPORT)
    end_time = datetime.datetime.now()
    target_dimension_filter = [
        {
            'Name': 'InstanceId',
            'Value': requested_instance
        }
    ]

    # get a merged set of metrics by day + add computed items
    daily_df = get_merged_metric_df(cw_client,
                                    NAMESPACE,
                                    METRICS,
                                    target_dimension_filter,
                                    DAILY_PERIOD,
                                    start_time,
                                    end_time)

    daily_df['Credits_Earned_Per_Day'] = instance_defaults['cpu_credits_per_hour'] * 24
    daily_df['Credit_Net'] = daily_df['Credits_Earned_Per_Day'] - daily_df['CPUCreditUsage_Sum']

    # rename columns to save space on output
    daily_df.rename(columns={
        'CPUUtilization_Average': 'CPU_Average',
        'CPUCreditUsage_Sum': 'Credits_Used',
        'CPUCreditBalance_Minimum': 'CrBalance_Min',
        'CPUCreditBalance_Maximum': 'CrBalance_Max'
    }, inplace=True)

    # get a merged set of metrics by hour + add computed items
    hourly_df = get_merged_metric_df(cw_client,
                                     NAMESPACE,
                                     METRICS,
                                     target_dimension_filter,
                                     HOURLY_PERIOD,
                                     start_time,
                                     end_time)

    hourly_df['Credits_Earned_Per_Hour'] = instance_defaults['cpu_credits_per_hour']
    hourly_df['Credit_Net'] = hourly_df['Credits_Earned_Per_Hour'] - hourly_df['CPUCreditUsage_Sum']
    hourly_df['Burst_TTL_40'] = get_instance_ttl_hours(instance_defaults, .4, hourly_df['CPUCreditBalance_Average'])
    hourly_df['Burst_TTL_60'] = get_instance_ttl_hours(instance_defaults, .6, hourly_df['CPUCreditBalance_Average'])
    hourly_df['Burst_TTL_80'] = get_instance_ttl_hours(instance_defaults, .8, hourly_df['CPUCreditBalance_Average'])
    hourly_df['Burst_TTL_100'] = get_instance_ttl_hours(instance_defaults, 1, hourly_df['CPUCreditBalance_Average'])

    # rename columns to save space on output
    hourly_df.rename(columns={
        'CPUUtilization_Average': 'CPU_Average',
        'CPUCreditUsage_Sum': 'Credits_Used',
        'CPUCreditBalance_Minimum': 'CrBalance_Min',
        'CPUCreditBalance_Maximum': 'CrBalance_Max'
    }, inplace=True)

    # Generate aging table data
    aging = get_credit_aging_dict(hourly_df, requested_instance_launch_dt)

    print ''
    print 'Instance ID: %s' % (requested_instance)
    print 'Instance Type: %s' % (requested_instance_type)
    print 'Launched: %s' % (str(requested_instance_launch_dt))
    print 'Base CPU: %d%%' % (instance_defaults['base_cpu_performance'] * 100)
    print 'Initial Burst Credits: %d' % (instance_defaults['initial_cpu_credit'])
    print 'Burst Credits Earned per Hour: %s' % (instance_defaults['cpu_credits_per_hour'])
    print 'Maximum Credit Balance: %s' % (instance_defaults['maximum_credit_balance'])

    print ''
    print 'Summary by Hour:'
    print tabulate(hourly_df[['CPU_Average',
                              'Credits_Used',
                              'Credit_Net',
                              'CrBalance_Min',
                              'CrBalance_Max',
                              'Burst_TTL_40',
                              'Burst_TTL_60',
                              'Burst_TTL_80',
                              'Burst_TTL_100'
                              ]], headers='keys', tablefmt='psql', floatfmt=".2f")

    print 'Burst Time-to-live (TTL) = Minutes at specified CPU utilization (40, 60, 80, 100) until \
burstable CPU credits will be expended.  Note: This calculation doesn not include initial credits.'

    print ''
    print 'Estimated earned burst credits by age (expire after 24 hours)'
    print tabulate(aging,  headers='keys', tablefmt='psql', floatfmt=".2f")

    print ''
    print 'Summary by day:'
    print tabulate(daily_df[['CPU_Average',
                             'Credits_Used',
                             'Credit_Net',
                             'CrBalance_Min',
                             'CrBalance_Max',
                             ]], headers='keys', tablefmt='psql', floatfmt=".2f")


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


def get_merged_metric_df(cw_client, namespace, metrics, dimension_filters, period, start_time, end_time):
    """
    Get a merged dataframe for a set of specified metrics

    Args:
        * cw_client (botocore.client.CloudWatch): Boto3 CloudWatch client
        * namespace (str): AWS Metric namespace
            see https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/aws-namespaces.html
        * metrics (list of str): List of metric names
            see https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ec2-metricscollected.html
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
            get_metric_df(
                cw_client,
                namespace,
                metric,
                dimension_filters,
                period,
                start_time,
                end_time
            )
        )

    return pandas.concat(dataframes, axis=1)


def get_metric_df(cw_client, namespace, metric_name, dimension_filters, period, start_time, end_time):
    """
    Get an EC2 metric and convert it to a pandas dataframe w/ timeseries index

    Args:
        * cw_client (botocore.client.CloudWatch): Boto3 CloudWatch client
        * namespace (str): AWS Metric namespace
            see https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/aws-namespaces.html
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
        Namespace=namespace,
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


def get_instance_launch_time(ec2_client, instance_id):
    """
    Lookup the launch time for a given instance id

    Args:
        * ec2_client (botocore.client.EC2): Boto3 EC2 Client Instance
        * instance_id (str): Instance ID to look up (e.g., i-1701a887)

    Returns:
        * Datetime.  Instance start time
    """
    return ec2_client.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]['LaunchTime']


def get_instance_ttl_hours(instance_defaults, target_pct, credit_balance):
    """
    Estimate burstable hours given base cpu rate, percentage level, credit balance, and credit aging
    """
    if instance_defaults['base_cpu_performance'] < target_pct:
        return (credit_balance / (target_pct * 60)) * 60

    return '-'


def add_prior_day_totals(hourly_df, column_name):
    """
    For each datapoint, if a value exists for the prior day, add it as a new column.  Will be used for
    better modeling of credit loss.

    REVIEW: there is likely some more functional/idiomatic way to do this with Pandas - not finding it now, though

    Args:
        * hourly_df (Pandas.DataFrame): base hourly_df w/ Credit_Net
        * column_name (str): Name of new column to add

    Returns:
        * Pandas.DataFrame.  Dataframe with additional column
    """
    hourly_df[column_name] = '-'
    for ix, row in hourly_df.iterrows():
        prior_day_index = ix - datetime.timedelta(hours=24)
        if prior_day_index in hourly_df.index:
            hourly_df.ix[ix, column_name] = hourly_df.ix[prior_day_index]['Credit_Net']

    return hourly_df


def get_credit_aging_dict(hourly_df, launch_dt):
    """
    Given an hourly dataframe, generate an ordered dictionary of estimated aging bins
    Purpose: see if an unusual number of credits will be expiring at any point in time

    Note: restarting instance clears out credits -> no accumulation afterward

    Args:
        * hourly_df (Pandas.DataFrame): dataframe for hourly credit net totals
        * launch_dt (datetime.datetime): Datetime for when instance was launched

    Returns:
        * collections.OrderedDict.  Ordered dictionary of aging bins
    """
    current_dt = datetime.datetime.now()
    aging = [OrderedDict([
        ('21-24 Hours (expiring soon)', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(hours=24)) &
                        (hourly_df.index <= current_dt - datetime.timedelta(hours=21)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('17-20 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(hours=20)) &
                        (hourly_df.index <= current_dt - datetime.timedelta(hours=17)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('13-16 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(hours=16)) &
                        (hourly_df.index <= current_dt - datetime.timedelta(hours=13)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('9-12 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(hours=12)) &
                        (hourly_df.index <= current_dt - datetime.timedelta(hours=9)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('5-8 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(hours=8)) &
                        (hourly_df.index <= current_dt - datetime.timedelta(hours=5)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('0-4 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(hours=4)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                     ]['Credit_Net'].sum()),
    ])]

    return aging

if __name__ == "__main__":
    main(sys.argv)
