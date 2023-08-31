import numpy as np
import queue
import heapq
import matplotlib.pyplot as plt
import csv

class Node:
    def __init__(self, node_id, node_type, position, communication_range):
        self.node_id = node_id
        self.node_type = node_type
        self.position = position
        self.communication_range = communication_range
        self.active = False
        self.covered_relays = set()
        self.relay_cost = 0  

    def distance(node1, node2):
        return np.linalg.norm(node1.position - node2.position)

    def distance_to(self, other_node):
        return np.linalg.norm(self.position - other_node.position)

    def distance_to_sink(self, sink_node):
        return self.distance_to(sink_node)

    def set_primary_path(self, path):
        path_list = [tuple(node) for node in path]
        self.primary_path = set(path_list)

        node_objects_list = [Node("R" + str(i), "Relay", np.array(node), 1) for i, node in enumerate(path_list)]
        self.covered_relays = set(relay_node for relay_node in node_objects_list[1:-1] if relay_node.node_type == 'Relay')

    def __lt__(self, other):
        return self.distance_to_sink(self) < self.distance_to_sink(other)

def can_communicate(node1, node2, threshold=1.0):
    return np.linalg.norm(node1.position - node2.position) <= max(node1.communication_range, node2.communication_range) * threshold

def place_relays_random(num_relays, area_size, communication_range):
    relays = [Node("R" + str(i), "Relay", np.random.rand(2) * area_size, communication_range) for i in range(1, num_relays + 1)]
    return relays

def place_relays_triangular(num_relays, area_size, communication_range):
    relays = []
    num_rows = int(area_size / (communication_range * np.sqrt(3) / 2)) + 1
    num_cols = int(area_size / communication_range)

    max_relays = num_rows * num_cols
    num_relays_to_place = min(num_relays, max_relays)

    diagonal_distance = communication_range * np.sqrt(3)

    for row in range(num_rows):
        for col in range(num_cols):
            if len(relays) >= num_relays_to_place:
                break
            x = col * communication_range
            y = row * communication_range * np.sqrt(3) / 2
            if row % 2 == 1:
                x += communication_range / 2
            position = np.array([x, y])
            position = np.clip(position, 0, area_size)  

            connected_relays = []
            for relay in relays:
                distance_to_relay = np.linalg.norm(relay.position - position)
                if distance_to_relay <= diagonal_distance:
                    connected_relays.append(relay)

            new_relay = Node("R" + str(len(relays) + 1), "Relay", position, communication_range)
            relays.append(new_relay)

            for relay in connected_relays:
                relay.covered_relays.add(new_relay)
                new_relay.covered_relays.add(relay)

    return relays

def place_relays_rectangular(num_relays, area_size, communication_range):
    num_cols = int(area_size / communication_range) + 1
    num_rows = num_cols
    remaining_relays = num_relays % num_cols

    max_relays = num_cols * num_rows + num_cols
    num_relays_to_place = min(num_relays, max_relays)

    relays = []
    for row in range(num_rows):
        for col in range(num_cols):
            if len(relays) >= num_relays_to_place:
                break
            x = col * communication_range
            y = row * communication_range
            position = np.array([x, y])
            position = np.clip(position, 0, area_size)  
            relays.append(Node("R" + str(len(relays) + 1), "Relay", position, communication_range))

    for col in range(remaining_relays):
        if len(relays) >= num_relays_to_place:
            break
        x = col * communication_range
        y = num_rows * communication_range
        position = np.array([x, y])
        position = np.clip(position, 0, area_size)  
        relays.append(Node("R" + str(len(relays) + 1), "Relay", position, communication_range))

    return relays

def find_communication_path_dijkstra(sensor_node, relay_nodes, sink_node, threshold):
    distances = {node: float('inf') for node in relay_nodes | {sink_node}}
    distances[sensor_node] = 0
    previous_nodes = {node: None for node in relay_nodes | {sink_node}}
    priority_queue = [(0, sensor_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == sink_node:
            break

        if current_distance > distances[current_node]:
            continue

        for neighbor in [node for node in relay_nodes | {sink_node} if can_communicate(current_node, node, threshold)]:
            distance_to_neighbor = current_distance + current_node.distance(neighbor) + neighbor.relay_cost  
            if distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = distance_to_neighbor
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance_to_neighbor, neighbor))

    if previous_nodes[sink_node] is None:
        return None

    current_node = sink_node
    path = [current_node]
    while current_node != sensor_node:
        current_node = previous_nodes[current_node]
        path.append(current_node)

    path.reverse()
    return path

def find_secondary_path(sensor_node, relay_nodes, primary_path, sink_node, threshold):
    primary_path_relays = set(tuple(relay) for relay in primary_path[1:-1] if tuple(relay) in relay_nodes)

    distances = {node: float('inf') for node in relay_nodes | {sink_node}}
    distances[sensor_node] = 0
    previous_nodes = {node: None for node in relay_nodes | {sink_node}}
    priority_queue = [(0, sensor_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == sink_node:
            break

        if current_distance > distances[current_node]:
            continue

        for neighbor in [node for node in relay_nodes | {sink_node} if can_communicate(current_node, node, threshold) and node not in primary_path_relays]:
            distance_to_neighbor = current_distance + current_node.distance(neighbor) + neighbor.relay_cost
            if distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = distance_to_neighbor
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance_to_neighbor, neighbor))

    if previous_nodes[sink_node] is None:
        return None

    current_node = sink_node
    path = [current_node]
    while current_node != sensor_node:
        current_node = previous_nodes[current_node]
        path.append(current_node)

    path.reverse()
    return path

def calculate_path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += np.linalg.norm(path[i + 1] - path[i])
    return length

def run_simulation(num_sensors, num_relays, num_sinks, area_size, relay_activation_threshold, relay_placement, sensor_positions, sink_positions):
    if relay_placement == 'random':
        relay_nodes = place_relays_random(num_relays, area_size, 1.0)
    elif relay_placement == 'triangular':
        relay_nodes = place_relays_triangular(num_relays, area_size, 1.0)
    elif relay_placement == 'rectangular':
        relay_nodes = place_relays_rectangular(num_relays, area_size, 1.0)
    else:
        raise ValueError("Invalid relay_placement option. Use 'random', 'triangular', or 'rectangular'.")

    sensor_nodes = [Node("S" + str(i), "Sensor", position, relay_activation_threshold) for i, position in enumerate(sensor_positions, 1)]
    sink_nodes = [Node("G" + str(i), "Sink", position, relay_activation_threshold) for i, position in enumerate(sink_positions, 1)]

    plt.figure(figsize=(10, 10))

    num_connected_sensors = 0
    active_relays = []
    total_primary_path_length = 0
    total_secondary_path_length = 0
    num_relay_connections = 0
    total_communication_cost = 0  


    for sensor_node in sensor_nodes:
        primary_path = find_communication_path_dijkstra(sensor_node, set(relay_nodes), next(iter(sink_nodes)), relay_activation_threshold)
        if primary_path is not None:
            primary_path = np.array([node.position for node in primary_path])
            plt.plot(primary_path[:, 0], primary_path[:, 1], color='darkgreen', linewidth=0.8, label='Primary Path')

            sensor_node.set_primary_path(primary_path)

            used_relays_in_primary_path = []
            for node in primary_path[1:-1]:
                for relay_node in relay_nodes:
                    if np.array_equal(relay_node.position, node) and relay_node not in used_relays_in_primary_path:
                        relay_node.relay_cost = calculate_additional_relay_cost(sensor_node, relay_node)
                        relay_node.active = True
                        active_relays.append(relay_node)
                        used_relays_in_primary_path.append(relay_node)

            sensor_node.active = True
            num_connected_sensors += 1  

            primary_path_cost = calculate_path_cost(primary_path)
            total_communication_cost += primary_path_cost

            total_primary_path_length += primary_path_cost

            secondary_path = find_secondary_path(sensor_node, {relay for relay in relay_nodes if relay not in used_relays_in_primary_path}, primary_path, next(iter(sink_nodes)), relay_activation_threshold)
            if secondary_path is not None:
                secondary_path = np.array([node.position for node in secondary_path])
                plt.plot(secondary_path[:, 0], secondary_path[:, 1], color='darkblue', linewidth=0.8, label='Secondary Path')

                for node in secondary_path[1:-1]:
                    for relay_node in relay_nodes:
                        if np.array_equal(relay_node.position, node):
                            relay_node.relay_cost = calculate_additional_relay_cost(sensor_node, relay_node)
                            relay_node.active = True
                            active_relays.append(relay_node)
                            num_relay_connections += 1

                secondary_path_cost = calculate_path_cost(secondary_path)
                total_communication_cost += secondary_path_cost

                total_secondary_path_length += secondary_path_cost

    num_used_relays = len(set(active_relays))

    for relay_node in relay_nodes:
        if relay_node not in active_relays:
            relay_node.active = False

    for sensor_node in sensor_nodes:
        color = 'blue' if sensor_node.active else 'lightblue'
        plt.scatter(sensor_node.position[0], sensor_node.position[1], color=color, marker='o', s=60, edgecolors='black', linewidth=0.5)
        plt.text(sensor_node.position[0], sensor_node.position[1]+0.1, str(sensor_node.node_id), fontsize=8, ha='center', va='bottom')

    for relay_node in relay_nodes:
        color = 'green' if relay_node.active else 'gray'
        plt.scatter(relay_node.position[0], relay_node.position[1], color=color, marker='o', s=60, edgecolors='black', linewidth=0.5)
        plt.text(relay_node.position[0], relay_node.position[1]+0.1, str(relay_node.node_id), fontsize=8, ha='center', va='bottom')

    for sink_node in sink_nodes:
        plt.scatter(sink_node.position[0], sink_node.position[1], color='red', marker='o', s=180, edgecolors='black', linewidth=0.5)
        plt.text(sink_node.position[0], sink_node.position[1]+0.1, str(sink_node.node_id), fontsize=8, ha='center', va='bottom')

    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.title('Dual-Homed of Multi-Tier WSN with Heterogeneous Nodes')
    plt.axis('equal')

    if num_connected_sensors == 0:
        print("No sensors are connected.")
    else:
        print("Evaluation Metrics:")
        print("Number of Connected Sensors:", num_connected_sensors)
        print("Number of Active Relays:", num_used_relays)
        print("Average Primary Path Length:", total_primary_path_length / num_connected_sensors)
        print("Average Secondary Path Length:", total_secondary_path_length / num_connected_sensors)
        average_relay_connectivity = num_relay_connections / num_used_relays
        print("Average Relay Connectivity:", average_relay_connectivity)
        print("Total Communication Cost:", total_communication_cost)  

    evaluation_metrics = {
        'Number of Connected Sensors': num_connected_sensors,
        'Number of Active Relays': num_used_relays,
        'Average Primary Path Length': total_primary_path_length / num_connected_sensors,
        'Average Secondary Path Length': total_secondary_path_length / num_connected_sensors,
        'Average Relay Connectivity': average_relay_connectivity,
        'Total Communication Cost': total_communication_cost
    }

    return evaluation_metrics

def calculate_additional_relay_cost(sensor_node, relay_node):
    return np.linalg.norm(sensor_node.position - relay_node.position) * 0.1  

def calculate_path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        cost += np.linalg.norm(path[i + 1] - path[i])
    return cost

def compare_placement_methods(num_sensors, num_relays, num_sinks, area_size, relay_activation_threshold):
    methods = ['random', 'triangular', 'rectangular']
    results = []

    sensor_positions = [np.random.rand(2) * area_size for _ in range(num_sensors)]
    sink_positions = [np.random.rand(2) * area_size for _ in range(num_sinks)]

    for method in methods:
        print(f"\nRunning simulation for {method} relay placement method:")
        print("=" * 50)
        result = run_simulation(num_sensors, num_relays, num_sinks, area_size, relay_activation_threshold, method, sensor_positions, sink_positions)
        results.append((method, result))

    return results

# def save_evaluation_metrics_to_csv(results):
#     with open('evaluation_metrics.csv', mode='w', newline='') as file:
#         fieldnames = ['Method', 'Connected Sensors', 'Active Relays', 'Avg Primary Path Length', 'Avg Secondary Path Length', 'Avg Relay Connectivity', 'Total Cost']
#         writer = csv.DictWriter(file, fieldnames=fieldnames)

#         writer.writeheader()
#         for method, result in results:
#             writer.writerow({
#                 'Method': method,
#                 'Connected Sensors': result['Number of Connected Sensors'],
#                 'Active Relays': result['Number of Active Relays'],
#                 'Avg Primary Path Length': result['Average Primary Path Length'],
#                 'Avg Secondary Path Length': result['Average Secondary Path Length'],
#                 'Avg Relay Connectivity': result['Average Relay Connectivity'],
#                 'Total Cost': result['Total Communication Cost']
#             })
            
# Run the simulation
num_sensors = 50 #number of sensors in the simulation
num_sinks = 2 #number of sinks
area_size = 10 #open area size
relay_activation_threshold = 1.01 #relay activation threshold
num_relays = 121 #number of relays that can be used

results = compare_placement_methods(num_sensors, num_relays, num_sinks, area_size, relay_activation_threshold)

# Plotting topologies
plt.figure(figsize=(8, 6))
plt.bar([res[0] for res in results], [res[1]['Average Relay Connectivity'] for res in results])
plt.xlabel('Relay Placement Method')
plt.ylabel('Average Relay Connectivity')
plt.title('Comparison of Relay Placement Methods')
plt.show()
