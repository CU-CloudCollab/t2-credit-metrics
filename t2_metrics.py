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
    if len(argv) < 2:
        raise ValueError("Please specify instance id (e.g., python t2-metrics.py i-1701a887")

    # setup AWS environment and clients
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    aws_clients = {
        'cw_client': boto3.client('cloudwatch'),
        'ec2_client': boto3.client('ec2')
    }

    # Run t2 metrics report.  Placeholder for additional reports by argument
    print t2_metrics_report(argv[1], aws_clients)


def t2_metrics_report(requested_instance, aws_clients, n_days_to_report=5):
    """
    Build input data for requested instance + print the t2 metrics reports

    Args:
        * requested_instance (str): Amazon instance ID
        * aws_clients (dict): EC2 and Cloudwatch Clients
            keys: cw_client, ec2_client
    """
    # Constants/Definitions
    HOURLY_PERIOD = 60*60
    DAILY_PERIOD = 60*60*24

    # Settings
    namespace = 'AWS/EC2'
    metrics = ['CPUUtilization', 'CPUCreditUsage', 'CPUCreditBalance']
    t2_instance_defaults = get_t2_defaults()

    # get metadata about requested instance
    requested_instance_type = get_instance_type(aws_clients['ec2_client'], requested_instance)

    # This only makes sense for t2 instances, so check that
    if requested_instance_type not in t2_instance_defaults:
        raise ValueError("Instance type %s doesn't appear to be in the t2 family" % (requested_instance_type))

    # build instance profile dict
    instance_profile = {
        'instance_id': requested_instance,
        'instance_defaults': t2_instance_defaults[requested_instance_type],
        'instance_type': requested_instance_type,
        'instance_launch_dt': get_instance_launch_time(aws_clients['ec2_client'], requested_instance)
    }

    # setup filters
    start_time = datetime.datetime.now() - datetime.timedelta(days=n_days_to_report)
    end_time = datetime.datetime.now()
    target_dimension_filter = [
        {
            'Name': 'InstanceId',
            'Value': requested_instance
        }
    ]

    # build dataframes
    hourly_df = build_hourly_t2_df(aws_clients['cw_client'], namespace, metrics, target_dimension_filter, HOURLY_PERIOD,
                                   instance_profile['instance_defaults'], start_time, end_time)

    daily_df = build_daily_t2_df(aws_clients['cw_client'], namespace, metrics, target_dimension_filter, DAILY_PERIOD,
                                 instance_profile['instance_defaults'], start_time, end_time)

    # Generate aging table data
    aging_dict = get_credit_aging_dict(hourly_df, instance_profile['instance_launch_dt'])

    # output the report
    return build_t2_metrics_report_output(instance_profile, hourly_df, daily_df, aging_dict)


def build_t2_metrics_report_output(instance_profile, hourly_df, daily_df, aging_dict):
    """
    Build output of t2 metrics report

    Args:
        * instance_profile (dict): Dictionary of information about requested instance
            keys: instance_id, instance_type, instance_defaults, instance_launch_dt
        * hourly_df (pandas.DataFrame): T2 metrics dataframe at hourly grain
        * daily_df (pandas.DataFrame): T2 metrics dataframe at daily grain
        * aging_dict (OrderedDict): Ordered dictionary of credit aging values

    Returns:
        * Str. String value of report
    """
    instance_defaults = instance_profile['instance_defaults']
    report = ''

    # print report
    report += '\n'
    report += 'Instance ID: %s\n' % (instance_profile['instance_id'])
    report += 'Instance Type: %s\n' % (instance_profile['instance_type'])
    report += 'Launched: %s\n' % (str(instance_profile['instance_launch_dt']))
    report += 'Base CPU: %d%%\n' % (instance_defaults['base_cpu_performance'] * 100)
    report += 'Initial Burst Credits: %d\n' % (instance_defaults['initial_cpu_credit'])
    report += 'Burst Credits Earned per Hour: %s\n' % (instance_defaults['cpu_credits_per_hour'])
    report += 'Maximum Credit Balance: %s\n' % (instance_defaults['maximum_credit_balance'])

    report += '\n'
    report += 'Summary by Hour:\n'
    report += tabulate(hourly_df[['CPU_Average',
                                  'Credits_Used',
                                  'Credit_Net',
                                  'CrBalance_Min',
                                  'CrBalance_Max',
                                  'Burst_TTL_40',
                                  'Burst_TTL_60',
                                  'Burst_TTL_80',
                                  'Burst_TTL_100'
                                  ]], headers='keys', tablefmt='psql', floatfmt=".2f") + '\n'

    report += 'Burst Time-to-live (TTL) = Minutes at specified CPU utilization (40, 60, 80, 100) until \
burstable CPU credits will be expended.  Note: This calculation doesn not include initial credits.\n'

    report += '\n'
    report += 'Estimated earned burst credits by age (expire after 24 hours)\n'
    report += tabulate(aging_dict,  headers='keys', tablefmt='psql', floatfmt=".2f") + '\n'

    report += '\n'
    report += 'Summary by day:\n'
    report += tabulate(daily_df[['CPU_Average',
                                 'Credits_Used',
                                 'Credit_Net',
                                 'CrBalance_Min',
                                 'CrBalance_Max',
                                 ]], headers='keys', tablefmt='psql', floatfmt=".2f")

    return report


def build_hourly_t2_df(cw_client, namespace, metrics, target_dimension_filter, hourly_period, instance_defaults,
                       start_time, end_time):
    """
    Get a metrics dataframe at hourly grain

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
        pandas.DataFrame.  Metrics dataframe at hourly grain
    """
    # get a merged set of metrics by hour + add computed items
    hourly_df = get_merged_metric_df(cw_client,
                                     namespace,
                                     metrics,
                                     target_dimension_filter,
                                     hourly_period,
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

    return hourly_df


def build_daily_t2_df(cw_client, namespace, metrics, target_dimension_filter, daily_period, instance_defaults,
                      start_time, end_time):
    """
    Get a metrics dataframe at daily grain

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
        pandas.DataFrame.  Metrics dataframe at daily grain
    """

    # get a merged set of metrics by day + add computed items
    daily_df = get_merged_metric_df(cw_client,
                                    namespace,
                                    metrics,
                                    target_dimension_filter,
                                    daily_period,
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

    return daily_df


def get_instance_ttl_hours(instance_defaults, target_pct, credit_balance):
    """
    Estimate burstable minutes given base cpu rate, percentage level, credit balance, and credit aging

    Args:
        * instance_defaults (dict): Instance defaults for the specified instance
        * target_pct (Decimal): The cpu usage rate for which we should estimate Time-to-live
        * credit_balance (Decimal): The existing credit balance to use for estimating Time-to-live
    """
    if instance_defaults['base_cpu_performance'] < target_pct:
        return (credit_balance / (target_pct * 60)) * 60

    return '-'


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
        ('20-24 Hours (expiring soon)', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(minutes=24*60)) &
                        (hourly_df.index < current_dt - datetime.timedelta(minutes=20*60)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('16-20 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(minutes=20*60)) &
                        (hourly_df.index < current_dt - datetime.timedelta(minutes=16*60)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('12-16 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(minutes=16*60)) &
                        (hourly_df.index < current_dt - datetime.timedelta(minutes=12*60)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('8-12 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(minutes=12*60)) &
                        (hourly_df.index < current_dt - datetime.timedelta(minutes=8*60)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('4-8 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(minutes=8*60)) &
                        (hourly_df.index < current_dt - datetime.timedelta(minutes=4*60)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                      ]['Credit_Net'].sum()),
        ('0-4 Hours', hourly_df.ix[
                        (hourly_df.index >= current_dt - datetime.timedelta(minutes=4*60)) &
                        (hourly_df.index >= launch_dt) &
                        (hourly_df['Credit_Net'] > 0)
                     ]['Credit_Net'].sum()),
    ])]

    return aging


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


if __name__ == "__main__":
    main(sys.argv)
