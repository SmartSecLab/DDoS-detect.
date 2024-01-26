"""
Author: Nikola Gavric (nikola.gavric@kristiania.no)
Created on: 06-12-2023
Description: The script logs system and network information and performs ML inference using random forest.
"""

import os
import csv
from datetime import datetime
import joblib
import psutil
import time


FEATURES = [
	'Timestamp',
	'CPU-usage',
	'Num-processes',
	'Interrupts-per-sec',
	'DSK-write',
	'DSK-read',
	'RAM-percentage',
	'Unique-IPs',
	'Num-sockets',
	'Upload-speed',
	'Download-speed'
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

    interval = 1  # Set the interval in seconds (e.g., 1 second)

    # Load the pre-trained Random Forest model
    random_forest_model = joblib.load('random_forest_classifier.joblib')

    print(f"\033[92m\nDDoS detection occurring every {2*interval} seconds\n" \
          f"Press Ctrl+C to exit \033[0m")

    # Load the LabelEncoder used during training
    label_encoder = joblib.load('label_encoder.joblib')
	
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

        # Feed the features into the Random Forest model for prediction
        features_for_prediction = [data[1:]]  # Exclude the timestamp
        prediction = random_forest_model.predict(features_for_prediction)

        # Map the numeric label back to the original label name
        original_label = label_encoder.inverse_transform(prediction)

        # Print or use the prediction as needed
        print(f"Timestamp: {timestamp}, Predicted Attack Type: {original_label[0]}")

if __name__ == "__main__":
    main()
