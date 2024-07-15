import pyrender

def show_object_list(obj_list):
    scene = pyrender.Scene()

    for obj in obj_list:
        if obj.mesh:
            scene.add(pyrender.Mesh.from_trimesh(obj.mesh, smooth=False))

    pyrender.Viewer(scene, use_raymond_lighting=True)