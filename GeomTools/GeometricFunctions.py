from itertools import combinations
import numpy as np
import trimesh

# geometric functions -----------------------------------------------------

def get_angle_of_normal(normal, axis="Z") -> float:
    
    if axis == "Z":
        # Define a reference vector (e.g., [0, 0, 1] for the positive Z-axis)
        reference_vector = [0,0,1]
    elif axis == "Y":
        # Define a reference vector (e.g., [0, 0, 1] for the positive Z-axis)
        reference_vector = [0,1,0]
    elif axis == "X":
        # Define a reference vector (e.g., [0, 0, 1] for the positive Z-axis)
        reference_vector = [1,0,0]


    # Normalize the normal vector (ensure it has unit length)
    normalized_normal = normal / np.linalg.norm(normal)

    # Calculate the dot product between the normalized normal and the reference vector
    dot_product = np.dot(normalized_normal, reference_vector)

    # Calculate the angle in radians using the arccosine function
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Convert the angle from radians to degrees if needed
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def get_rotated_mesh_in_z_axis(mesh, degrees) -> trimesh.Trimesh:
    # Convert degrees to radians
    radians = np.radians(degrees)
    
    new_mesh = mesh.copy()
    
    
    rotation_matrix = np.array([
    [np.cos(radians), -np.sin(radians), 0, 0],
    [np.sin(radians), np.cos(radians), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

    # Apply the rotation matrix to the vertices of the mesh
    new_mesh.apply_transform(rotation_matrix.T)
    
    return new_mesh

def get_projected_mesh(mesh, plane_point) -> trimesh.Trimesh:
    
    '''Returns a mesh projection of a mesh 
    in a plane normal to Y-Axis defined by a specific point'''
    
    new_mesh = mesh.copy()
    new_mesh.apply_transform(trimesh.transformations.projection_matrix(plane_point, [0,0,1], pseudo=True))
    return new_mesh

def get_projected_meshes(element_list, plane_point) -> list[trimesh.Trimesh]:
    
    '''Returns a mesh projection of a list of api Elements 
    in a plane normal to Y-Axis defined by a specific point'''
    
    if len(element_list)>1:
        
        
        projected_meshes = get_projected_mesh(element_list[0].mesh, plane_point)
    
        for element in element_list[1:]:
            
            if element.mesh is not None:
                projected_meshes +=  get_projected_mesh(element.mesh, plane_point)
            
    else:
        
        projected_meshes = get_projected_mesh(element_list[0].mesh, plane_point)
        
    return projected_meshes

def get_most_distant_point_in_axis(mesh,axis) -> float:
    
    '''Returns the most distant point from a mesh in a specified axis
    -- the axis should be 'Y', 'X' or 'Z' '''
    
    if axis == 'X':
        point = mesh.vertices[np.argmax(mesh.vertices[:, 0])]
    elif axis == 'Y':
        point = mesh.vertices[np.argmax(mesh.vertices[:, 1])]
    elif axis == 'Z':
        point = mesh.vertices[np.argmax(mesh.vertices[:, 2])]
        
    return point

def get_perpendicular_distance_point_to_line(point,path) -> float:
    
    '''Returns the perpendicular distance from a point to a path line'''
    
    max_distance = path.length
    
    # Calculate pairwise distances and find the most distant points
    most_distant_points=None
    
    for point1, point2 in combinations(path.vertices, 2):
        distance = np.linalg.norm(point1 - point2)
        if distance >= max_distance:
            max_distance = distance
            most_distant_points = (point1, point2)
            
    # Calculate the direction vector of the line segment
    line_direction = most_distant_points[1] - most_distant_points[0] 

    # Calculate the point on the line closest to the given point
    closest_point_on_line = most_distant_points[0] + np.dot(point - most_distant_points[0], line_direction) / np.dot(line_direction, line_direction) * line_direction

    # Calculate the perpendicular distance between the closest point on the line and the given point
    distance = np.linalg.norm(closest_point_on_line - point)
    
    return distance

def get_footprint_area(mesh) -> float:
    
    return mesh.section([0,0,1],mesh.bounds[1]-0.1).to_planar()[0].area

def get_depth(mesh, front_mesh) -> float:
    
    # Determine the direction based on the centroids of the front mesh and the mesh's oriented bounding box
    direction_to_other_object = front_mesh.centroid - mesh.bounding_box_oriented.centroid
    
    # Initiate auxiliary variables
    max_dot_product = -float('inf')
    front_face_normal = None

    # Iterate through faces of the oriented bounding box and analizes how aligned they are based on its dot product
    for normal in mesh.bounding_box_oriented.face_normals:
        dot_product = np.dot(normal, direction_to_other_object)
        
        if dot_product > max_dot_product:
            max_dot_product = dot_product
            front_face_normal = normal

    # get rotation of the front face normal in relation to the axis-X        
    rotation_angle = 90 - get_angle_of_normal(front_face_normal, "X")
    
    
    # Rotate the mesh and gets it y component
    return((get_rotated_mesh_in_z_axis(mesh.bounding_box_oriented, rotation_angle).extents)[1])

def get_projected_area_of_element(element) -> float:
            
    return trimesh.path.polygons.projected(element,normal=[0,0,1]).area

def get_projected_area_of_elements(elements) -> float:
    
    return sum([get_projected_area_of_element(el.mesh) for el in elements])

def get_distance_object_to_point(obj,point,axis="Y") -> float: 
      
    axis_dict = {
        "X":0,
        "Y":1,
        "Z":2
    }
        
    # Get the vertices of the path3d object
    path_vertices = obj.vertices

    # Calculate the distance in the y-axis
    # Find the closest vertex on the path to the point
    closest_vertex_idx = np.argmin(np.abs(path_vertices[:, axis_dict[axis]] - point[axis_dict[axis]]))

    # Calculate the y-axis distance between the original point and the closest vertex
    distance = np.abs(point[axis_dict[axis]] - path_vertices[closest_vertex_idx, axis_dict[axis]])
        
    return distance