"""
Author: Nikola Gavric (nikola.gavric@kristiania.no)
Created on: 06-12-2023
Description: The script logs system and network information into log.csv every second.
"""

import psutil
import time
import csv
from datetime import datetime
import os

FEATURES = [
	'Timestamp',
	'CPU_usage',
	'Num_processes',
	'Interrupts_per_sec',
	'DSK-write',
	'DSK-read',
	'RAM-percentage',
	'Unique_IPs',
	'Num_Sockets',
	'Upload_speed',
	'Download_speed'
]

def write_header(file_path):
	with open(file_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(FEATURES)

def write_to_csv(file_path, data):
	with open(file_path, 'a', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(data)

def get_system_and_network_info(interval):
	# Get initial network information using psutil
	net_io_counters_start = psutil.net_io_counters()

	# Get initial interrupts count
	interrupts_start = psutil.cpu_stats().interrupts
	
	# Get disk usage (write and read per second)
	disk_io_counters_start = psutil.disk_io_counters()
	
	# Sleep for the specified interval
	time.sleep(interval)

	# Get network information again after the interval
	net_io_counters_end = psutil.net_io_counters()

	# Calculate bytes sent and received per second
	bytes_sent_per_sec = (net_io_counters_end.bytes_sent - net_io_counters_start.bytes_sent) / interval
	bytes_received_per_sec = (net_io_counters_end.bytes_recv - net_io_counters_start.bytes_recv) / interval

	# Get disk usage again after the interval
	disk_io_counters_end = psutil.disk_io_counters()

	# Calculate bytes written and read per second
	disk_write_per_sec = (disk_io_counters_end.write_bytes - disk_io_counters_start.write_bytes) / interval
	disk_read_per_sec = (disk_io_counters_end.read_bytes - disk_io_counters_start.read_bytes) / interval

	# Get active connections using psutil
	connections = psutil.net_connections(kind='inet')

	# Extract unique remote IP addresses
	unique_ips = set(connection.raddr.ip for connection in connections if connection.raddr)

	# Get final interrupts count
	interrupts_end = psutil.cpu_stats().interrupts
	interrupts_per_sec = interrupts_end - interrupts_start

	# Get CPU usage percentage
	cpu_usage = psutil.cpu_percent(interval=1)

	# Get number of running processes
	num_processes = len(list(psutil.process_iter()))

	# Get RAM memory usage
	ram_usage = psutil.virtual_memory()

	# Get SWAP memory usage
	swap_usage = psutil.swap_memory()

	return {
		'network_info': {
			'unique_ips': len(unique_ips),
			'active_sockets': len(connections),
			'bytes_sent_per_sec': bytes_sent_per_sec,
			'bytes_received_per_sec': bytes_received_per_sec
		},
		'system_info': {
			'cpu_usage': cpu_usage,
			'num_processes': num_processes,
			'interrupts_per_sec': interrupts_per_sec,
			'disk_write_per_sec': disk_write_per_sec,
			'disk_read_per_sec': disk_read_per_sec,
			'ram_usage': ram_usage.percent				 
		}
	}
	
def main():
	log_file = 'log.csv'
	
	if not os.path.isfile(log_file):
		# Write header if the file is empty
		write_header(log_file)

	interval = 1  # Set the interval in seconds (e.g., 1 second)
	
	print(f"\033[92m\nLogging system and network information in progress\n" \
		  f"Dataset file name:\t\t\t{log_file}\n\n" \
		  f"Press Ctrl+C to exit \033[0m")

	while True:
		info = get_system_and_network_info(interval)

		# Prepare data for writing to CSV
		timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		data = [
			timestamp,
			round(info['system_info']['cpu_usage'], 2),
			info['system_info']['num_processes'],
			info['system_info']['interrupts_per_sec'],
			round(info['system_info']['disk_write_per_sec'], 2),
			round(info['system_info']['disk_read_per_sec'], 2),
			round(info['system_info']['ram_usage'], 2),
			info['network_info']['unique_ips'],
			info['network_info']['active_sockets'],
			round(info['network_info']['bytes_sent_per_sec'], 2),
			round(info['network_info']['bytes_received_per_sec'], 2)
		]

		# Uncomment the following lines if you want to print the data
		# print("Timestamp:", timestamp)
		# print("CPU Usage (%):", data[1])
		# print("Number of Processes:", data[2])
		# print("Interrupts per Second:", data[3])
		# print("Disk Write Speed (Bytes per Second):", data[4])
		# print("Disk Read Speed (Bytes per Second):", data[5])
		# print("RAM Usage (%):", data[6])
		# print("Unique IPs:", data[7])
		# print("Active Sockets:", data[8])
		# print("Upload Speed (Bytes per Second):", data[9])
		# print("Download Speed (Bytes per Second):", data[10])
		# print("----------------------")

		# Write to CSV
		write_to_csv(log_file, data)

if __name__ == "__main__":
	main()
