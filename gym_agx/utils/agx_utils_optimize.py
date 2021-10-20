
import agx
import agxCollide
import agxOSG
import agxSDK
import agxRender
import agxModel
import agxUtil
import agxIO


def createConvexDecomposition(geometry):

    remove_shapes = []
    add_shapes = []
    num_vertices_render_data = 0
    num_vertices_mesh_data = 0

    shapes = geometry.getShapes()
    for s in shapes:
        mesh = s.asMesh()
        if not mesh:
            continue

        data = mesh.getMeshData()
        render_data = mesh.getRenderData()

        vertices = data.getVertices()
        indices = data.getIndices()

        print("Create convex decomposition")
        print("Trimesh num vertices:", len(vertices))

        result = agxCollide.ConvexRefVector()
        resolution = 10
        assert agxUtil.createConvexDecomposition(vertices, indices, result, resolution) > 0
        convex_num_vertices = 0

        result[0].setRenderData(render_data)
        num_vertices_render_data += len(render_data.getVertexArray())

        for c in result:
            convex_num_vertices += len(c.getMeshData().getVertices())
            add_shapes.append(c)

        remove_shapes.append(s)

    print("Convex num vertices: ", convex_num_vertices)
    num_vertices_mesh_data += convex_num_vertices

    for remove in remove_shapes:
        print("Removing shape", remove)
        geometry.remove(remove)

    for add in add_shapes:
        geometry.add(add)

    return num_vertices_render_data, num_vertices_mesh_data


def reduceMesh(geometry):

    remove_shapes = []
    add_shapes = []

    render_data_size = 0
    mesh_data_size = 0

    shapes = geometry.getShapes()
    for s in shapes:
        mesh = s.asMesh()
        if not mesh:
            continue

        data = mesh.getMeshData()

        vertices = data.getVertices()
        indices = data.getIndices()

        out_vertices = agx.Vec3Vector()
        out_indices = agx.UInt32Vector()

        remove_shapes.append(s)
        render_data = s.getRenderData()

        timer = agx.Timer(True)
        print("Before: {} # vertices", len(vertices))
        # Now do the actual reduction
        assert agxUtil.reduceMesh(
            vertices, indices,
            out_vertices, out_indices,
            0.3,
            7.0)

        print("After: {} # vertices", len(out_vertices))
        print("Reduction ratio: {}%", 100*(len(vertices) - len(out_vertices))/len(vertices))
        print("Time to reduce:", timer.getTime())
        options = agxCollide.Trimesh.NO_WARNINGS + agxCollide.Trimesh.REMOVE_DUPLICATE_VERTICES
        new_mesh = agxCollide.Trimesh(out_vertices, out_indices, "", options)
        new_mesh.setRenderData(render_data)

        nv = len(render_data.getVertexArray())
        ni = len(render_data.getIndexArray())
        nn = len(render_data.getNormalArray())
        nt = len(render_data.getTexCoordArray())
        nc = len(render_data.getColorArray())

        size = (nv * 3 * 8) + (ni * 4) + (nn * 3 * 8) + (nt * 2 * 8) + (nc * 4 * 8)
        render_data_size += size
        print("Render data size: {} Mb".format(size / 1E6))

        nvm = len(out_vertices)
        nim = len(out_indices)
        size_mesh = (nvm * 4 * 8) + (nim * 4)
        mesh_data_size += size_mesh
        print("Mesh data size: {} Mb".format(size_mesh / 1E6))
        print("#meshVertices-renderVertices:", nvm - nv)

        add_shapes.append(new_mesh)

    for remove in remove_shapes:
        print("Removing shape", remove)
        geometry.remove(remove)

    for add in add_shapes:
        geometry.add(add)

    return render_data_size, mesh_data_size


