import collections
import datetime
import unittest

import botocore.session
import pandas
import t2_metrics

from botocore.stub import Stubber
from nose.tools import assert_equals


class TestT2Metrics(unittest.TestCase):
    def setUp(self):
        """
        Setup test responses so we can use across multiple tests
        """
        # get_instance_type() test data
        ec2_client = botocore.session.get_session().create_client('ec2')
        ec2_stubber = Stubber(ec2_client)
        self.stub_ec2_response(ec2_stubber)
        ec2_stubber.activate()
        self.get_instance_type_response = t2_metrics.get_instance_type(ec2_client, '12345')

        # get_instance_launch_time() test data
        self.stub_ec2_response(ec2_stubber)
        ec2_stubber.activate()
        self.get_instance_launch_time_response = t2_metrics.get_instance_launch_time(ec2_client, '12345')

        # get_t2_defaults() test data
        self.t2_defaults = t2_metrics.get_t2_defaults()

        # get_metric_df() test data
        cw_client = botocore.session.get_session().create_client('cloudwatch')
        cw_stubber = Stubber(cw_client)
        self.stub_cloudwatch_response(cw_stubber, 'CPUCreditUsage')
        cw_stubber.activate()
        self.metric_df_response = t2_metrics.get_metric_df(
            cw_client,
            'AWS/EC2',
            'CPUCreditUsage',
            [
                {
                    'Name': 'InstanceId',
                    'Value': 'test-instance'
                }
            ],
            60*60,
            datetime.datetime(2016, 5, 1),
            datetime.datetime(2016, 6, 1)
        )

        # get_merged_metric_df() test data
        cw_client = botocore.session.get_session().create_client('cloudwatch')
        cw_stubber = Stubber(cw_client)
        self.stub_cloudwatch_response(cw_stubber, 'CPUCreditUsage')
        self.stub_cloudwatch_response(cw_stubber, 'CPUCreditBalance')
        cw_stubber.activate()
        self.merged_metric_df_response = t2_metrics.get_merged_metric_df(
            cw_client,
            'AWS/EC2',
            ['CPUCreditUsage', 'CPUCreditBalance'],
            [
                {
                    'Name': 'InstanceId',
                    'Value': 'test-instance'
                }
            ],
            60*60,
            datetime.datetime(2016, 5, 1),
            datetime.datetime(2016, 6, 1)
        )

    def test_get_instance_type(self):
        """
        get_instance_type() should return expected value from stubbed response
        """

        assert_equals(self.get_instance_type_response,
                      't2.micro')

    def test_get_instance_launch_time(self):
        """
        get_instance_launch_time() should return expected value from stubbed response
        """
        assert_equals(self.get_instance_launch_time_response,
                      datetime.datetime(2015, 1, 1))

    def test_t2_defaults_item_count(self):
        """
        get_t2_defaults() should return expected number of items
        """
        assert_equals(len(self.t2_defaults), 5)

    def test_t2_defaults_keys(self):
        """
        get_t2_defaults() should return keys for each t2 instance type
        """
        assert_equals('t2.nano' in self.t2_defaults, True)
        assert_equals('t2.micro' in self.t2_defaults, True)
        assert_equals('t2.small' in self.t2_defaults, True)
        assert_equals('t2.medium' in self.t2_defaults, True)
        assert_equals('t2.large' in self.t2_defaults, True)

    def test_t2_defaults_dict_keys(self):
        """
        get_t2_defaults() instance dictionaries should have expected set of keys
        """
        # Has keys for each value of interest for each instance type
        for key, instance in self.t2_defaults.iteritems():
            assert_equals('initial_cpu_credit' in instance, True)
            assert_equals('cpu_credits_per_hour' in instance, True)
            assert_equals('base_cpu_performance' in instance, True)
            assert_equals('maximum_credit_balance' in instance, True)

    def test_get_metric_df_type(self):
        """
        get_metric_df() should return Pandas DataFrame
        """
        assert_equals(isinstance(self.metric_df_response, pandas.core.frame.DataFrame), True)

    def test_get_metric_df_length(self):
        """
        get_metric_df() should return expected number of rows
        """
        assert_equals(len(self.metric_df_response), 3)

    def test_get_metric_df_prefixes(self):
        """
        get_metric_df() should return measures prefixed with metric name
        """
        # Should prefix measures with metric name
        assert_equals(list(self.metric_df_response.columns.values),
                      ['CPUCreditUsage_Average',
                       'CPUCreditUsage_Maximum',
                       'CPUCreditUsage_Minimum',
                       'CPUCreditUsage_SampleCount',
                       'CPUCreditUsage_Sum',
                       'CPUCreditUsage_Timestamp',
                       'CPUCreditUsage_Unit'
                       ])

    def test_get_metric_df_sums(self):
        """
        get_metric_df() columns checksums should be as expected from test data
        """
        assert_equals(self.metric_df_response['CPUCreditUsage_Average'].sum(), 675)
        assert_equals(self.metric_df_response['CPUCreditUsage_Maximum'].sum(), 684)
        assert_equals(self.metric_df_response['CPUCreditUsage_Minimum'].sum(), 681)
        assert_equals(self.metric_df_response['CPUCreditUsage_SampleCount'].sum(), 672)
        assert_equals(self.metric_df_response['CPUCreditUsage_Sum'].sum(), 678)

    def test_merged_metric_df_type(self):
        """
        get_merged_metric_df() should return Pandas DataFrame
        """
        assert_equals(isinstance(self.merged_metric_df_response, pandas.core.frame.DataFrame), True)

    def test_merged_metric_df_length(self):
        """
        get_merged_metric_df() should return expected number of rows
        """
        assert_equals(len(self.merged_metric_df_response), 3)

    def test_merged_metric_df_columns(self):
        """
        get_merged_metric_df() should return prefixed columns from requested metrics
        """
        assert_equals(list(self.merged_metric_df_response.columns.values),
                      ['CPUCreditUsage_Average',
                       'CPUCreditUsage_Maximum',
                       'CPUCreditUsage_Minimum',
                       'CPUCreditUsage_SampleCount',
                       'CPUCreditUsage_Sum',
                       'CPUCreditUsage_Timestamp',
                       'CPUCreditUsage_Unit',
                       'CPUCreditBalance_Average',
                       'CPUCreditBalance_Maximum',
                       'CPUCreditBalance_Minimum',
                       'CPUCreditBalance_SampleCount',
                       'CPUCreditBalance_Sum',
                       'CPUCreditBalance_Timestamp',
                       'CPUCreditBalance_Unit'
                       ])

    def test_merged_metric_df_index_merge(self):
        """
        get_merged_metric_df() should return observation measures merged on datetime index
        """
        for index, metric_row in self.merged_metric_df_response.iterrows():
            assert_equals(metric_row['CPUCreditUsage_Average'], metric_row['CPUCreditBalance_Average'])
            assert_equals(metric_row['CPUCreditUsage_Maximum'], metric_row['CPUCreditBalance_Maximum'])
            assert_equals(metric_row['CPUCreditUsage_Minimum'], metric_row['CPUCreditBalance_Minimum'])
            assert_equals(metric_row['CPUCreditUsage_SampleCount'], metric_row['CPUCreditBalance_SampleCount'])
            assert_equals(metric_row['CPUCreditUsage_Sum'], metric_row['CPUCreditBalance_Sum'])
            assert_equals(metric_row['CPUCreditUsage_Timestamp'], metric_row['CPUCreditBalance_Timestamp'])
            assert_equals(metric_row['CPUCreditUsage_Unit'], metric_row['CPUCreditBalance_Unit'])

    def test_get_instance_ttl_hours_calc(self):
        """
        get_instance_ttl_hours() should calculate expected values for given inputs
        """
        mock_t2_instance = {
            'initial_cpu_credit': 30,
            'cpu_credits_per_hour': 12,
            'base_cpu_performance': .2,
            'maximum_credit_balance': 288
        }

        assert_equals(t2_metrics.get_instance_ttl_hours(mock_t2_instance, .4, 300), 750)
        assert_equals(t2_metrics.get_instance_ttl_hours(mock_t2_instance, .6, 270), 450)

    def test_get_instance_ttl_hours_unlimited(self):
        """
        get_instance_ttl_hours() should return '-' if base cpu rate is >= target
        """
        mock_t2_instance = {
            'initial_cpu_credit': 30,
            'cpu_credits_per_hour': 12,
            'base_cpu_performance': .6,
            'maximum_credit_balance': 288
        }

        assert_equals(t2_metrics.get_instance_ttl_hours(mock_t2_instance, .4, 300), '-')
        assert_equals(t2_metrics.get_instance_ttl_hours(mock_t2_instance, .6, 270), '-')

    def test_get_credit_aging_dict_calc(self):
        """
        get_credit_aging_dict() should calculate expected values for given inputs
        """

        test_set = [
            {
                'Timestamp': datetime.datetime.now(),
                'Credit_Net': 5
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=239),
                'Credit_Net': 7
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=240),
                'Credit_Net': 9
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=479),
                'Credit_Net': 11
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=480),
                'Credit_Net': 13
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=719),
                'Credit_Net': 15
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=720),
                'Credit_Net': 17
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=959),
                'Credit_Net': 19
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=960),
                'Credit_Net': 21
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=1199),
                'Credit_Net': 23
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=1200),
                'Credit_Net': 25
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=1439),
                'Credit_Net': 27
            }
        ]

        test_df = pandas.DataFrame.from_dict(test_set)
        test_df.set_index(pandas.DatetimeIndex(test_df['Timestamp']), inplace=True)

        test_launch_dt = datetime.datetime.now() - datetime.timedelta(hours=25)
        response = t2_metrics.get_credit_aging_dict(test_df, test_launch_dt)

        assert_equals(isinstance(response[0], collections.OrderedDict), True)
        assert_equals(response[0]['0-4 Hours'], 12)
        assert_equals(response[0]['4-8 Hours'], 20)
        assert_equals(response[0]['8-12 Hours'], 28)
        assert_equals(response[0]['12-16 Hours'], 36)
        assert_equals(response[0]['16-20 Hours'], 44)
        assert_equals(response[0]['20-24 Hours (expiring soon)'], 52)

    def test_get_credit_aging_dict_pre_launch_credits(self):
        """
        get_credit_aging_dict() should not count credits if they occurred prior to last launch
        """

        test_set = [
            {
                'Timestamp': datetime.datetime.now(),
                'Credit_Net': 5
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=239),
                'Credit_Net': 7
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=240),
                'Credit_Net': 9
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=479),
                'Credit_Net': 11
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=480),
                'Credit_Net': 13
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=719),
                'Credit_Net': 15
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=720),
                'Credit_Net': 17
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=959),
                'Credit_Net': 19
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=960),
                'Credit_Net': 21
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=1199),
                'Credit_Net': 23
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=1200),
                'Credit_Net': 25
            },
            {
                'Timestamp': datetime.datetime.now() - datetime.timedelta(minutes=1439),
                'Credit_Net': 27
            }
        ]

        test_df = pandas.DataFrame.from_dict(test_set)
        test_df.set_index(pandas.DatetimeIndex(test_df['Timestamp']), inplace=True)

        test_launch_dt = datetime.datetime.now() - datetime.timedelta(minutes=961)
        response = t2_metrics.get_credit_aging_dict(test_df, test_launch_dt)

        assert_equals(isinstance(response[0], collections.OrderedDict), True)
        assert_equals(response[0]['0-4 Hours'], 12)
        assert_equals(response[0]['4-8 Hours'], 20)
        assert_equals(response[0]['8-12 Hours'], 28)
        assert_equals(response[0]['12-16 Hours'], 36)
        assert_equals(response[0]['16-20 Hours'], 21)  # datapoint > 960 should be excluded
        assert_equals(response[0]['20-24 Hours (expiring soon)'], 0)

    def stub_cloudwatch_response(self, cw_stubber, metric):
        """
        Create a mock response for cloudwatch metrics and add to stubber

        Args:
            * cw_stubber (botocore.stub.Stubber): Stubber instance for cloudwatch client
            * metric (str): Metric name to include in stubbed response
        """
        response = {
            'Label': 'Test Response',
            'Datapoints': [
                {
                    'Timestamp': datetime.datetime(2016, 1, 1),
                    'SampleCount': 123.0,
                    'Average': 124.0,
                    'Sum': 125.0,
                    'Minimum': 126.0,
                    'Maximum': 127.0,
                    'Unit': 'Count'
                },
                {
                    'Timestamp': datetime.datetime(2016, 1, 2),
                    'SampleCount': 224,
                    'Average': 225,
                    'Sum': 226,
                    'Minimum': 227,
                    'Maximum': 228,
                    'Unit': 'Count'
                },
                {
                    'Timestamp': datetime.datetime(2016, 1, 3),
                    'SampleCount': 325,
                    'Average': 326,
                    'Sum': 327,
                    'Minimum': 328,
                    'Maximum': 329,
                    'Unit': 'Count'
                },
            ]
        }

        expected_params = {
            'Namespace': 'AWS/EC2',
            'MetricName': metric,
            'Dimensions': [
                {
                    'Name': 'InstanceId',
                    'Value': 'test-instance'
                }
            ],
            'Period': 3600,
            'Statistics': ['Sum', 'Average', 'Minimum', 'Maximum', 'SampleCount'],
            'StartTime': datetime.datetime(2016, 5, 1),
            'EndTime': datetime.datetime(2016, 6, 1),
        }

        cw_stubber.add_response('get_metric_statistics', response, expected_params)

    def stub_ec2_response(self, ec2_stubber):
        """
        Create a mock response for ec2 metrics and add to stubber

        Args:
            * cw_stubber (botocore.stub.Stubber): Stubber instance for ec2 client
        """
        response = {
            'Reservations': [
                {
                    'ReservationId': 'string',
                    'OwnerId': 'string',
                    'RequesterId': 'string',
                    'Groups': [
                        {
                            'GroupName': 'string',
                            'GroupId': 'string'
                        },
                    ],
                    'Instances': [
                        {
                            'InstanceId': 'string',
                            'ImageId': 'string',
                            'State': {
                                'Code': 123,
                                'Name': 'running'
                            },
                            'PrivateDnsName': 'string',
                            'PublicDnsName': 'string',
                            'StateTransitionReason': 'string',
                            'KeyName': 'string',
                            'AmiLaunchIndex': 123,
                            'ProductCodes': [
                                {
                                    'ProductCodeId': 'string',
                                    'ProductCodeType': 'marketplace'
                                },
                            ],
                            'InstanceType': 't2.micro',
                            'LaunchTime': datetime.datetime(2015, 1, 1),
                            'Placement': {
                                'AvailabilityZone': 'string',
                                'GroupName': 'string',
                                'Tenancy': 'default',
                                'HostId': 'string',
                                'Affinity': 'string'
                            },
                            'KernelId': 'string',
                            'RamdiskId': 'string',
                            'Platform': 'Windows',
                            'Monitoring': {
                                'State': 'enabled'
                            },
                            'SubnetId': 'string',
                            'VpcId': 'string',
                            'PrivateIpAddress': 'string',
                            'PublicIpAddress': 'string',
                            'StateReason': {
                                'Code': 'string',
                                'Message': 'string'
                            },
                            'Architecture': 'x86_64',
                            'RootDeviceType': 'ebs',
                            'RootDeviceName': 'string',
                            'BlockDeviceMappings': [
                                {
                                    'DeviceName': 'string',
                                    'Ebs': {
                                        'VolumeId': 'string',
                                        'Status': 'attached',
                                        'AttachTime': datetime.datetime(2015, 1, 1),
                                        'DeleteOnTermination': True
                                    }
                                },
                            ],
                            'VirtualizationType': 'hvm',
                            'InstanceLifecycle': 'scheduled',
                            'SpotInstanceRequestId': 'string',
                            'ClientToken': 'string',
                            'Tags': [
                                {
                                    'Key': 'string',
                                    'Value': 'string'
                                },
                            ],
                            'SecurityGroups': [
                                {
                                    'GroupName': 'string',
                                    'GroupId': 'string'
                                },
                            ],
                            'SourceDestCheck': True,
                            'Hypervisor': 'ovm',
                            'NetworkInterfaces': [
                                {
                                    'NetworkInterfaceId': 'string',
                                    'SubnetId': 'string',
                                    'VpcId': 'string',
                                    'Description': 'string',
                                    'OwnerId': 'string',
                                    'Status': 'available',
                                    'MacAddress': 'string',
                                    'PrivateIpAddress': 'string',
                                    'PrivateDnsName': 'string',
                                    'SourceDestCheck': True,
                                    'Groups': [
                                        {
                                            'GroupName': 'string',
                                            'GroupId': 'string'
                                        },
                                    ],
                                    'Attachment': {
                                        'AttachmentId': 'string',
                                        'DeviceIndex': 123,
                                        'Status': 'attached',
                                        'AttachTime': datetime.datetime(2015, 1, 1),
                                        'DeleteOnTermination': True
                                    },
                                    'Association': {
                                        'PublicIp': 'string',
                                        'PublicDnsName': 'string',
                                        'IpOwnerId': 'string'
                                    },
                                    'PrivateIpAddresses': [
                                        {
                                            'PrivateIpAddress': 'string',
                                            'PrivateDnsName': 'string',
                                            'Primary': True,
                                            'Association': {
                                                'PublicIp': 'string',
                                                'PublicDnsName': 'string',
                                                'IpOwnerId': 'string'
                                            }
                                        },
                                    ]
                                },
                            ],
                            'IamInstanceProfile': {
                                'Arn': 'string',
                                'Id': 'string'
                            },
                            'EbsOptimized': True,
                            'SriovNetSupport': 'string',
                            'EnaSupport': True
                        },
                    ]
                },
            ]
        }

        expected_params = {'InstanceIds': ['12345']}

        ec2_stubber.add_response('describe_instances', response, expected_params)

if __name__ == '__main__':
    unittest.main()
