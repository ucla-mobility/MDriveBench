import carla

# Connect to your running CARLA server (port 2000)
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)

# Read the .xodr file
with open('/data2/marco/CoLMDriver/ucla_v2.xodr', 'r') as f:
    xodr_content = f.read()

# Setup generation parameters
# Note: wall_height=0.0 is better for seeing your waypoints clearly
params = carla.OpendriveGenerationParameters(
    vertex_distance=2.0, 
    max_road_length=50.0, 
    wall_height=0.0, 
    additional_width=0.6, 
    smooth_junctions=True, 
    enable_mesh_visibility=True
)

print("Sending OpenDRIVE data to server...")
client.generate_opendrive_world(xodr_content, params)
print("Map loaded successfully!")
