from GeomTools.CheckModel import CheckModel
import trimesh

# load model
my_model = CheckModel("TestFile/BuildingTeste3.ifc")


# get examples of objects that collide
space = my_model.select('#2VqlKGzpbEYfMhQ83V14GE')[0]
wall = my_model.select('#3dUPquTnvBDe7uKbalFHNI')[0]

# get examples of objects that don't coolide
wall_a = my_model.select('#3dUPquTnvBDe7uKbalFHP5')[0]
wall_b = my_model.select('#3dUPquTnvBDe7uKbalFHVg')[0]


c_manager = trimesh.collision.CollisionManager() # create manager that contains all of the objects that need to check collision

c_manager.add_object("space",space.mesh)
c_manager.add_object("wall",wall.mesh)
c_manager.add_object("wall_a",wall_a.mesh)
c_manager.add_object("wall_b",wall_b.mesh)

# docs to collision manager https://trimesh.org/trimesh.collision.html
# it also has some distance and more complex collision methods that might help you

# the collision amongst all the objects inserted in the collision manager will be checked
result = c_manager.in_collision_internal(return_names=True)


print(result)