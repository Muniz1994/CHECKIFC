# ------------------------------------------------------------------------------
#  Abstraction Data Processing Classes
#-----------------------------------------------------------------------------

class CheckElement:

    ''' Class created to give the mesh processing capabilities of Trimesh to IfcOpenShell elements'''

    def __init__(self, ifcOS_element, mesh=None):

        self.ifc = ifcOS_element
        self.mesh = mesh
        
    def __getattr__(self, attr):
        
        ''' The class uses this magic method to associate the IfcOpenShell elements' attributes to the CheckElement object'''
        
        return getattr(self.ifc,attr)
        