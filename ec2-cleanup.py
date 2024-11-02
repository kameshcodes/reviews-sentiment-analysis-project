import boto3
import paramiko
import logging
import os
from botocore.exceptions import NoCredentialsError, ClientError
from typing import List, Tuple

# Ensure log directory exists
LOG_DIR = 'log'
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
LOG_FILE_PATH = os.path.join(LOG_DIR, 'cleanup_log.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()  # Log to console as well
    ]
)

# AWS region and SSH configurations
REGION = 'ap-south-1'  # Specify your region
SSH_KEY_PATH = '/path/to/your/ssh-key.pem'  # Replace with path to your SSH key
EC2_USER = 'ec2-user'  # Replace with your EC2 username

# Docker cleanup commands to be run on each instance
DOCKER_CLEANUP_COMMANDS = """
docker stop $(docker ps -q) || true
docker rm $(docker ps -aq) || true
docker rmi $(docker images -q) --force || true
docker volume rm $(docker volume ls -q) --force || true
docker system prune -af --volumes || true
"""


def get_all_instances(region: str) -> List[Tuple[str, str]]:
    """
    Retrieves all running EC2 instances in the specified region.

    Args:
        region (str): AWS region name.

    Returns:
        List[Tuple[str, str]]: List of instance IDs and their corresponding public IP addresses.
    """
    ec2 = boto3.client('ec2', region_name=region)
    try:
        response = ec2.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        instances = [
            (instance['InstanceId'], instance['PublicIpAddress'])
            for reservation in response['Reservations']
            for instance in reservation['Instances']
            if 'PublicIpAddress' in instance
        ]
        logging.info(f"Found {len(instances)} running instances.")
        return instances
    except (NoCredentialsError, ClientError) as e:
        logging.error(f"Error retrieving instances: {e}")
        return []


def clean_docker_on_instance(ip_address: str) -> None:
    """
    Executes Docker cleanup commands on an EC2 instance via SSH.

    Args:
        ip_address (str): The public IP address of the instance.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname=ip_address, username=EC2_USER, key_filename=SSH_KEY_PATH)
        logging.info(f"Connected to {ip_address}. Cleaning Docker resources...")

        stdin, stdout, stderr = ssh.exec_command(DOCKER_CLEANUP_COMMANDS)
        logging.info(stdout.read().decode())
        errors = stderr.read().decode()
        if errors:
            logging.error(f"Errors during Docker cleanup on {ip_address}: {errors}")
        else:
            logging.info(f"Successfully cleaned Docker resources on {ip_address}.")
    except Exception as e:
        logging.error(f"Failed to clean Docker on {ip_address}: {e}")
    finally:
        ssh.close()


def terminate_instances(instance_ids: List[str], region: str) -> None:
    """
    Terminates specified EC2 instances.

    Args:
        instance_ids (List[str]): List of instance IDs to terminate.
        region (str): AWS region name.
    """
    ec2 = boto3.client('ec2', region_name=region)
    try:
        ec2.terminate_instances(InstanceIds=instance_ids)
        logging.info(f"Terminated instances: {instance_ids}")
    except (NoCredentialsError, ClientError) as e:
        logging.error(f"Error terminating instances: {e}")


def delete_detached_volumes(region: str) -> None:
    """
    Deletes all detached EBS volumes in the specified region.

    Args:
        region (str): AWS region name.
    """
    ec2 = boto3.client('ec2', region_name=region)
    try:
        volumes = ec2.describe_volumes(
            Filters=[{'Name': 'status', 'Values': ['available']}]
        )
        for volume in volumes['Volumes']:
            volume_id = volume['VolumeId']
            ec2.delete_volume(VolumeId=volume_id)
            logging.info(f"Deleted detached volume: {volume_id}")
    except ClientError as e:
        logging.error(f"Error deleting volumes: {e}")


def release_elastic_ips(region: str) -> None:
    """
    Releases all unattached Elastic IPs in the specified region.

    Args:
        region (str): AWS region name.
    """
    ec2 = boto3.client('ec2', region_name=region)
    try:
        addresses = ec2.describe_addresses()
        for address in addresses['Addresses']:
            if 'InstanceId' not in address:  # Only release unattached IPs
                allocation_id = address['AllocationId']
                ec2.release_address(AllocationId=allocation_id)
                logging.info(f"Released Elastic IP: {allocation_id}")
    except ClientError as e:
        logging.error(f"Error releasing Elastic IPs: {e}")


def delete_load_balancers(region: str) -> None:
    """
    Deletes all load balancers in the specified region.

    Args:
        region (str): AWS region name.
    """
    elb = boto3.client('elbv2', region_name=region)
    try:
        load_balancers = elb.describe_load_balancers()
        for lb in load_balancers['LoadBalancers']:
            lb_arn = lb['LoadBalancerArn']
            elb.delete_load_balancer(LoadBalancerArn=lb_arn)
            logging.info(f"Deleted Load Balancer: {lb_arn}")
    except ClientError as e:
        logging.error(f"Error deleting Load Balancers: {e}")


def delete_snapshots(region: str) -> None:
    """
    Deletes all snapshots owned by the user in the specified region.

    Args:
        region (str): AWS region name.
    """
    ec2 = boto3.client('ec2', region_name=region)
    try:
        snapshots = ec2.describe_snapshots(OwnerIds=['self'])['Snapshots']
        for snapshot in snapshots:
            snapshot_id = snapshot['SnapshotId']
            ec2.delete_snapshot(SnapshotId=snapshot_id)
            logging.info(f"Deleted snapshot: {snapshot_id}")
    except ClientError as e:
        logging.error(f"Error deleting snapshots: {e}")


def delete_security_groups(region: str) -> None:
    """
    Deletes all security groups except the default in the specified region.

    Args:
        region (str): AWS region name.
    """
    ec2 = boto3.client('ec2', region_name=region)
    try:
        security_groups = ec2.describe_security_groups()
        for sg in security_groups['SecurityGroups']:
            if sg['GroupName'] != 'default':  # Avoid default group
                try:
                    ec2.delete_security_group(GroupId=sg['GroupId'])
                    logging.info(f"Deleted security group: {sg['GroupId']}")
                except ClientError as e:
                    logging.warning(f"Could not delete security group {sg['GroupId']}: {e}")
    except ClientError as e:
        logging.error(f"Error listing security groups: {e}")


def main() -> None:
    """
    Main function that performs cleanup tasks on EC2 instances, volumes, Elastic IPs,
    load balancers, snapshots, security groups, and S3 buckets in the specified region.
    """
    instances = get_all_instances(REGION)
    if instances:
        for instance_id, ip_address in instances:
            clean_docker_on_instance(ip_address)
        instance_ids = [instance_id for instance_id, _ in instances]
        terminate_instances(instance_ids, REGION)

    # Additional cleanup steps
    delete_detached_volumes(REGION)
    release_elastic_ips(REGION)
    delete_load_balancers(REGION)
    delete_snapshots(REGION)
    delete_security_groups(REGION)


if __name__ == "__main__":
    main()
