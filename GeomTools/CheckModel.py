import ifcopenshell
from GeomTools.CheckElement import CheckElement
from .Selector import Selector
import ifcopenshell.util.shape
import ifcopenshell.geom
import multiprocessing
import trimesh
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


VALID_REPRESENTATION_TYPES: list = [
            "Point",
            "PointCloud",
            "Curve",
            "Curve2D",
            "Curve3D",
            "Surface",
            "Surface2D",
            "Surface3D",
            "Advanced Surface",
            "GeometricSet",
            "GeometricCurveSet",
            "SurfaceModel",
            "Tessellation",
            "SolidModel",
            "SweptSolid",
            "Advanced SweptSolid",
            "Brep",
            "AdvancedBrep",
            "CSG",
            "Clipping",
            "BoundingBox",
            "Sectioned Spine",
            "MappedRepresentation"
        ]

    
class CheckModel:

    ''' Class that enable the access to loading and querying models
    
    Example:

    my_model = CheckModel('PathToMy.ifc', load_geometry=True)

    walls = my_model.select('.IfcWall')
    
    '''
    def __init__(self, ifc_path, load_geometry=False) -> None:

        self.ifc: ifcopenshell.file = ifcopenshell.open(ifc_path) # The IfcOpenShell file



        self.__parser: Selector = Selector() # The selector object to enable the queries

        # The geometry settings
        self.__geometry_settings = ifcopenshell.geom.settings()
        
        self.__geometry_settings.set(self.__geometry_settings.USE_WORLD_COORDS, True)

        self.__geometry_settings.set(self.__geometry_settings.APPLY_DEFAULT_MATERIALS, True)

        contexts = [c.id() for c in self.ifc.by_type("IfcGeometricRepresentationContext") if c.ContextIdentifier == "Body"]
        
        self.__geometry_settings.set_context_ids(contexts)

        self.__geometry_settings.set(self.__geometry_settings.BOOLEAN_ATTEMPT_2D, True)


        # The map conversion to enable the meshes of IFC4 files to be correctly georeferenced

        self.map_conversion = self.ifc.by_type('IfcMapConversion')[0] if self.ifc.schema == "IFC4" else None


        # Calculations for georeferencing 

        self.rot_angle = np.round(np.arctan2(self.map_conversion.XAxisOrdinate,self.map_conversion.XAxisAbscissa),13) if self.map_conversion else 0

        self.rot_matrix = trimesh.transformations.rotation_matrix(self.rot_angle,[0,0,1]) 

        self.translation_vector = np.array([self.map_conversion.Eastings,self.map_conversion.Northings,self.map_conversion.OrthogonalHeight]) if self.map_conversion else np.array([0,0,0])

        if self.map_conversion:
            if self.map_conversion.Scale:
                self.scale = self.map_conversion.Scale
            else:
                self.scale = 1
        else:
            self.scale = 1

        self.mesh_dict: dict = dict()

        if load_geometry:
        
            self.mesh_dict = self.__get_mesh_dict()

    def __has_valid_representation(self,element) -> bool:
    
        if element.Representation:
            for representation in element.Representation.Representations:
                    if representation.RepresentationType in VALID_REPRESENTATION_TYPES:
                        return True
                    else:
                        return False                  
        else:
            return False
                           

    def select(self, query: str) -> list:

        query_result: list = self.__parser.parse(self.ifc, query)

        # Verify if the element has a valid representation and create the CheckElement instance for each object in the query result
        elements_with_meshes: list = [CheckElement(element, self.mesh_dict[element.GlobalId]) if (self.__has_valid_representation(element) and element.GlobalId in self.mesh_dict) else CheckElement(element) for element in query_result] if query_result else None

        return (elements_with_meshes)
    
    def __process_shape(self, shape, scale, rot_matrix, translation_vector):
        
        mesh = trimesh.Trimesh(
            vertices=ifcopenshell.util.shape.get_vertices(shape.geometry),
            faces=ifcopenshell.util.shape.get_faces(shape.geometry)
        )

        color = list(shape.geometry.materials[0].diffuse)

        face_colors = np.array([color for _ in range(len(mesh.faces))])

        mesh.visual.face_colors = face_colors

        # Combine transformations: scaling, rotation, and translation
        transform_matrix = np.dot(rot_matrix, np.diag([scale, scale, scale, 1]))
        transform_matrix[:3, 3] += translation_vector
        mesh.apply_transform(transform_matrix)

        return shape.guid, mesh

    def __get_mesh_dict(self) -> dict:

        time_0 = datetime.now()

        print("Loading geometry!")

        iterator = ifcopenshell.geom.iterator(self.__geometry_settings, self.ifc, multiprocessing.cpu_count())

        mesh_dict = {}

        if iterator.initialize():

            with ThreadPoolExecutor() as executor:
                futures = []

                while True:

                    shape = iterator.get()

                    futures.append(executor.submit(self.__process_shape, shape, self.scale, self.rot_matrix, self.translation_vector))
                    
                    if not iterator.next():
                        break

                    for future in as_completed(futures):
                        guid, mesh = future.result()
                        mesh_dict[guid] = mesh

        print("Finished loading geometry!")
        time_1 = datetime.now()
        print("elapsed time: " + str(time_1-time_0))
        return(mesh_dict)
    








