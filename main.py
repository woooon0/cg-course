from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

g_cam_theta = 90 
g_cam_pi = 90   
g_cmove_offset = glm.vec3(0.,0.,0.)
g_l_xyz = glm.vec3(0.,0.,0.) # initial lookat point
g_c_u = glm.vec3()
g_c_v = glm.vec3()
g_c_w = glm.vec3()
g_m_left = False
g_init_point = list([])
g_cmove_x = 0.
g_cmove_y = 0.
g_y_change = 0.
g_x_change = 0.
g_dist = 1
g_u_change = glm.vec3()
g_v_change = glm.vec3()
g_is_rev = False
g_rev_change = glm.vec3()
g_mod_toggle = 0
g_models = []
g_bvhroot = 0
g_motion_active = False
g_current_frame = 0
g_num_frame = 0
g_frame_time = 0
g_motion_data = []
g_ispreloaded = False
g_bonesize = 0.05


g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_normal;
out vec3 vout_surface_pos;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;  // interpolated normal

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 lightmove;
uniform vec3 material_color;


void main()
{
 // light and material properties
    vec3 light_pos = lightmove;
    vec3 light_color = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.5*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = vec3(1,1,1);  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''
g_cubev_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 
layout (location = 2) in vec3 vin_color;

out vec3 vout_normal;
out vec3 vout_surface_pos;
out vec3 vout_color;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;
    vout_color = vin_color;
    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_cubef_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;  // interpolated normal
in vec3 vout_color;  

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 lightmove;


void main()
{
 // light and material properties
    vec3 light_pos = lightmove;
    vec3 light_color = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.5*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = vout_color;
    vec3 material_diffuse = vout_color;
    vec3 material_specular = vec3(1,1,1);  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''




def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def prepare_cube_vao():
    # 36 vertices: [x, y, z, nx, ny, nz]
    cube_vertices = np.array([
        # -Z face
        -0.4, -0.4, -0.4,  0, 0, -1,1,0,0,
         0.4, -0.4, -0.4,  0, 0, -1,1,0,0,
         0.4,  0.4, -0.4,  0, 0, -1,1,0,0,
         0.4,  0.4, -0.4,  0, 0, -1,1,0,0,
        -0.4,  0.4, -0.4,  0, 0, -1,1,0,0,
        -0.4, -0.4, -0.4,  0, 0, -1,1,0,0,
        # +Z face
        -0.4, -0.4, 0.4,   0, 0, 1,0,1,0,
         0.4, -0.4, 0.4,   0, 0, 1,0,1,0,
         0.4,  0.4, 0.4,   0, 0, 1,0,1,0,
         0.4,  0.4, 0.4,   0, 0, 1,0,1,0,
        -0.4,  0.4, 0.4,   0, 0, 1,0,1,0,
        -0.4, -0.4, 0.4,   0, 0, 1,0,1,0,
        # -X face
        -0.4,  0.4,  0.4, -1, 0, 0,0,0,1,
        -0.4,  0.4, -0.4, -1, 0, 0,0,0,1,
        -0.4, -0.4, -0.4, -1, 0, 0,0,0,1,
        -0.4, -0.4, -0.4, -1, 0, 0,0,0,1,
        -0.4, -0.4,  0.4, -1, 0, 0,0,0,1,
        -0.4,  0.4,  0.4, -1, 0, 0,0,0,1,
        # +X face
         0.4,  0.4,  0.4, 1, 0, 0,1,1,0,
         0.4,  0.4, -0.4, 1, 0, 0,1,1,0,
         0.4, -0.4, -0.4, 1, 0, 0,1,1,0,
         0.4, -0.4, -0.4, 1, 0, 0,1,1,0,
         0.4, -0.4,  0.4, 1, 0, 0,1,1,0,
         0.4,  0.4,  0.4, 1, 0, 0,1,1,0,
        # -Y face
        -0.4, -0.4, -0.4, 0, -1, 0,0,1,1,
         0.4, -0.4, -0.4, 0, -1, 0,0,1,1,
         0.4, -0.4,  0.4, 0, -1, 0,0,1,1,
         0.4, -0.4,  0.4, 0, -1, 0,0,1,1,
        -0.4, -0.4,  0.4, 0, -1, 0,0,1,1,
        -0.4, -0.4, -0.4, 0, -1, 0,0,1,1,
        # +Y face
        -0.4, 0.4, -0.4, 0, 1, 0,1,0,1,
         0.4, 0.4, -0.4, 0, 1, 0,1,0,1,
         0.4, 0.4,  0.4, 0, 1, 0,1,0,1,
         0.4, 0.4,  0.4, 0, 1, 0,1,0,1,
        -0.4, 0.4,  0.4, 0, 1, 0,1,0,1,
        -0.4, 0.4, -0.4, 0, 1, 0,1,0,1,
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)
    # position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * cube_vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * cube_vertices.itemsize, ctypes.c_void_p(3 * cube_vertices.itemsize))
    glEnableVertexAttribArray(1)
    # color
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * cube_vertices.itemsize, ctypes.c_void_p(6 * cube_vertices.itemsize))
    glEnableVertexAttribArray(2)
    glBindVertexArray(0)
    return vao

def prepare_vao_lines(grid_size=10, grid_spacing=1.0):
    vertices = []

    for z in range(-grid_size, grid_size + 1):
        vertices.extend([-grid_size * grid_spacing, 0.0, z * grid_spacing, 0.0, 1.0, 0.0]) 
        vertices.extend([grid_size * grid_spacing, 0.0, z * grid_spacing, 0.0, 1.0, 0.0])  

    for x in range(-grid_size, grid_size + 1):
        vertices.extend([x * grid_spacing, 0.0, -grid_size * grid_spacing, 0.0, 1.0, 0.0])  
        vertices.extend([x * grid_spacing, 0.0, grid_size * grid_spacing, 0.0, 1.0, 0.0])   

    # Convert to numpy array
    vertices = np.array(vertices, dtype=np.float32)

    # Create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  
    glBindVertexArray(VAO)      


    VBO = glGenBuffers(1)   
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  

    # Copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW) 
    # Configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)
    # Configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)
    return VAO, len(vertices) // 6 


class Model:
    def __init__(self, vertices, indices,name):
        self.index_count = len(indices)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.name = name
        self.obj = None
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER,vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24,  ctypes.c_void_p(3*np.dtype(np.float32).itemsize))
        glEnableVertexAttribArray(1)
    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES,0, self.index_count)

def load_obj(path):
    vertices = []
    normals = []
    indices = []
    normindices = []
    indice3 = 0
    indice4 = 0
    indicemore = 0

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.extend(vertex)  
            elif line.startswith('vn '):
                parts = line.strip().split()
                normal = list(map(float, parts[1:4]))
                normals.extend(normal)
            elif line.startswith('f '):  
                parts = line.strip().split()[1:]
                if "//" in parts[0]:
                    face = [int(p.split('//')[0]) - 1 for p in parts]
                    nface = [int(p.split('//')[1]) - 1 for p in parts]  
                    if len(face) == 3:
                        indices.extend(face)
                        normindices.extend(nface)
                        indice3 += 1
                    elif len(face) == 4:  
                        indices.extend([face[0], face[1], face[2]])  
                        indices.extend([face[0], face[2], face[3]])  
                        normindices.extend([nface[0], nface[1], nface[2]])  
                        normindices.extend([nface[0], nface[2], nface[3]])  
                        indice4 += 2
                    elif len(face) > 4:  
                        for i in range(1, len(face) - 1):
                            indices.extend([face[0], face[i], face[i + 1]])
                            normindices.extend([nface[0], nface[i], nface[i + 1]])
                            indicemore += 1
                elif "/" in parts[0]:
                    face = [int(p.split('/')[0]) - 1 for p in parts]
                    nface = [int(p.split('/')[2]) - 1 for p in parts]  
                    if len(face) == 3:
                        indices.extend(face)
                        normindices.extend(nface)
                        indice3 += 1
                    elif len(face) == 4:
                        indices.extend([face[0], face[1], face[2]])
                        indices.extend([face[0], face[2], face[3]])
                        normindices.extend([nface[0], nface[1], nface[2]])
                        normindices.extend([nface[0], nface[2], nface[3]])
                        indice4 += 2
                    elif len(face) > 4:
                        for i in range(1, len(face) - 1):
                            indices.extend([face[0], face[i], face[i + 1]])
                            normindices.extend([nface[0], nface[i], nface[i + 1]])
                            indicemore += 1
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    normindices = np.array(normindices, dtype=np.uint32)
    print("faces: ", indice3 + indice4 + indicemore)
    print("face with 3 vertices: ", indice3)
    print("face with 4 vertices: ", indice4)
    print("face with more than 4 vertices: ", indicemore)
    return vertices, normals, indices, normindices


def give_obj(joint,name,obj): 
    if joint.name == name:
        joint.obj = obj
        return 0
    for child in joint.children:
        give_obj(child,name,obj)

def appendmodel(file,name,rootnode):
    vertices, normals, indices, normindices = load_obj(file)
    vn = []
    for i in range(len(indices)):
        vn.extend(vertices[3*indices[i]:3*indices[i]+3])
        vn.extend(normals[3*normindices[i]:3*normindices[i]+3])
    vertices = np.array(vn,dtype=np.float32)
    indices = np.array(indices,dtype=np.uint32)
    model = Model(vertices, indices,name)
    give_obj(rootnode,name,model)




def obj1_callback():
    global g_models, g_bvhroot, g_motion_data, g_current_frame, g_motion_active
    bvhfile = os.path.join('jump.bvh')
    g_bvhroot,rootnum = parse_bvh(bvhfile)
    print("File name : " + bvhfile)
    print("Number of frames : " + str(g_num_frame))
    print("FPS : "+str(1/g_frame_time))
    print("Number of joints : " + str(rootnum))
    print("List of all joint names : ")
    g_bvhroot.printall()
    g_current_frame = 0
    g_motion_active = False
    print("bvh loaded")

    # file = os.path.join('hip.obj')
    # appendmodel(file,'hip')
    # file = os.path.join('abdomen.obj')
    # appendmodel(file,'abdomen')
    # file = os.path.join('chest.obj')
    # appendmodel(file,'chest')
    file = os.path.join('rFoot.obj')
    appendmodel(file,'rFoot',g_bvhroot)
    file = os.path.join('lFoot.obj')
    appendmodel(file,'lFoot',g_bvhroot)    
    file = os.path.join('rShin.obj')
    appendmodel(file,'rShin',g_bvhroot)
    file = os.path.join('lShin.obj')
    appendmodel(file,'lShin',g_bvhroot)    
    # file = os.path.join('lThigh.obj')
    # appendmodel(file,'lThigh')    
    # file = os.path.join('rThigh.obj')
    # appendmodel(file,'rThigh')    
    # file = os.path.join('lButtock.obj')
    # appendmodel(file,'lButtock')    
    # file = os.path.join('rButtock.obj')
    # appendmodel(file,'rButtock')    
    setup_all_obj_transforms(g_bvhroot)

class Joint:
    def __init__(self, name):
        self.name = name
        self.offset = np.array((0,0,0), dtype=float)
        self.channels = []
        self.children = []
        self.parent = None
        self.channel_indices = []
        self.bone_local_transform = None 
        self.obj_local_transform = None
        self.obj = None

    def add_child(self, joint):
        joint.parent = self
        self.children.append(joint)

    def printall(self, level=0):
        print("  " * level + self.name)
        for child in self.children:
            if child.name != "End Site":
                child.printall(level + 1)

def set_bone_local_transform(joint):
    if joint.parent is not None:
        start = np.zeros(3)
        end = joint.offset
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            joint.bone_local_transform = glm.mat4(1.0)
            return
        center = (start + end) / 2  # bone의 중간 위치

        y = np.array([0, 1, 0], dtype=np.float32)
        direction_n = direction / length
        axis = np.cross(y, direction_n)
        angle = np.arccos(np.clip(np.dot(y, direction_n), -1, 1))
        R = glm.mat4(1.0)
        if np.linalg.norm(axis) > 1e-6 and angle > 1e-6:
            R = glm.rotate(glm.mat4(1.0), angle, glm.vec3(*axis))
        S = glm.scale(glm.mat4(1.0), glm.vec3(g_bonesize, length, g_bonesize))
        T = glm.translate(glm.mat4(1.0), glm.vec3(*center))
        joint.bone_local_transform = T * R * S
    else:
        joint.bone_local_transform = glm.mat4(1.0)
def set_obj_local_transform(joint):
    if joint.parent is not None:
        start = np.zeros(3)
        end = joint.offset
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            joint.obj_local_transform = glm.mat4(1.0)
            return
        center = (start + end) / 2  # bone의 중간 위치

        y = np.array([0, 1, 0], dtype=np.float32)
        direction_n = direction / length
        axis = np.cross(y, direction_n)
        angle = np.arccos(np.clip(np.dot(y, direction_n), -1, 1))
        R = glm.mat4(1.0)
        if np.linalg.norm(axis) > 1e-6 and angle > 1e-6:
            R = glm.rotate(glm.mat4(1.0), angle, glm.vec3(*axis))
        T = glm.translate(glm.mat4(1.0), glm.vec3(*center))
        joint.obj_local_transform = T * R
    else:
        joint.obj_local_transform = glm.mat4(1.0)

def setup_all_bone_transforms(joint):  #본용
    set_bone_local_transform(joint)
    for child in joint.children:
        setup_all_bone_transforms(child)

def setup_all_obj_transforms(joint):  #obj용
    set_obj_local_transform(joint)
    for child in joint.children:
        setup_all_obj_transforms(child)

def draw_bone_cube(parent_transform, joint, loc_MVP, cube_vao, MVP):
    M = parent_transform * joint.bone_local_transform
    M = MVP * M
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(M))
    glBindVertexArray(cube_vao)
    glDrawArrays(GL_TRIANGLES, 0, 36)

def draw_bvh_cubes(joint, parent_transform, frame_data, loc_MVP, cube_vao, MVP):
    T = glm.translate(glm.mat4(1), glm.vec3(*joint.offset))
    if frame_data is not None and joint.channels:
        for i, j in enumerate(joint.channels):
            value = frame_data[joint.channel_indices[i]]
            if j == "XPOSITION" or j == "Xposition":
                T = glm.translate(T, glm.vec3(value, 0, 0))
            elif j == "YPOSITION" or j == "Yposition":
                T = glm.translate(T, glm.vec3(0, value, 0))
            elif j == "ZPOSITION" or j == "Zposition":
                T = glm.translate(T, glm.vec3(0, 0, value))
            elif j == "XROTATION" or j == "Xrotation":
                T = glm.rotate(T, glm.radians(value), glm.vec3(1, 0, 0))
            elif j == "YROTATION" or j == "Yrotation":
                T = glm.rotate(T, glm.radians(value), glm.vec3(0, 1, 0))
            elif j == "ZROTATION" or j == "Zrotation":
                T = glm.rotate(T, glm.radians(value), glm.vec3(0, 0, 1))
    world_transform = parent_transform * T
    if joint.parent is not None:
        if joint.parent.children[0] == joint:
            draw_bone_cube(parent_transform, joint, loc_MVP, cube_vao, MVP)

    for child in joint.children:
        draw_bvh_cubes(child, world_transform, frame_data, loc_MVP, cube_vao, MVP)




def draw_bvh_obj(parent_transform,joint,loc_MVP,MVP):
    M = parent_transform * joint.obj_local_transform
    M = MVP * M
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(M))
    joint.obj.draw()


def draw_bvh_objects(joint, parent_transform, frame_data, loc_MVP, MVP):
    T = glm.translate(glm.mat4(1), glm.vec3(*joint.offset))
    if frame_data is not None and joint.channels:
        for i, j in enumerate(joint.channels):
            value = frame_data[joint.channel_indices[i]]
            if j == "XPOSITION" or j == "Xposition":
                T = glm.translate(T, glm.vec3(value, 0, 0))
            elif j == "YPOSITION" or j == "Yposition":
                T = glm.translate(T, glm.vec3(0, value, 0))
            elif j == "ZPOSITION" or j == "Zposition":
                T = glm.translate(T, glm.vec3(0, 0, value))
            elif j == "XROTATION" or j == "Xrotation":
                T = glm.rotate(T, glm.radians(value), glm.vec3(1, 0, 0))
            elif j == "YROTATION" or j == "Yrotation":
                T = glm.rotate(T, glm.radians(value), glm.vec3(0, 1, 0))
            elif j == "ZROTATION" or j == "Zrotation":
                T = glm.rotate(T, glm.radians(value), glm.vec3(0, 0, 1))
    world_transform = parent_transform * T
    if joint.parent is not None:
        if joint.parent.children[0] == joint:
            if joint.obj is not None:
                draw_bvh_obj(parent_transform, joint, loc_MVP, MVP)

    for child in joint.children:
        draw_bvh_objects(child, world_transform, frame_data, loc_MVP, MVP)



 








def parse_bvh(path):
    global g_motion_data, g_frame_time,g_num_frame
    with open(path, 'r') as f:
        lines = iter(f.readlines())

    stack = []
    rootnode = None
    channel_index = 0
    jointnum = 0
    joints = []

    for line in lines:
        words = line.strip().split()
        if not words:
            continue
        if words[0] in ('ROOT', 'JOINT'):
            jointnum += 1
            joint_name = words[1]
            joint = Joint(joint_name)
            if stack:
                stack[-1].add_child(joint)
                
            stack.append(joint)
            if rootnode is None:
                rootnode = joint
            joints.append(joint)
        elif words[0] == 'End':
            joint = Joint('End Site')
            stack[-1].add_child(joint)
            stack.append(joint)
        elif words[0] == '{':
            continue
        elif words[0] == '}':
            stack.pop()
        elif words[0] == 'OFFSET':
            stack[-1].offset = [float(w) for w in words[1:]]
        elif words[0] == 'CHANNELS':
            channels = words[2:]
            stack[-1].channels = channels
            stack[-1].channel_indices = list(range(channel_index, channel_index + len(channels)))
            channel_index += len(channels)
        elif words[0] == 'Frames:':
            g_num_frame = int(words[1])
        elif words[0] == 'Frame' and words[1] == 'Time:':
            g_frame_time = float(words[2])
            break

    g_motion_data = [list(map(float, next(lines).strip().split())) for _ in range(g_num_frame)]
    setup_all_bone_transforms(rootnode)
    return rootnode,jointnum


def bvhdrop_callback(window, paths):
    global  g_bvhroot, g_motion_data, g_current_frame, g_motion_active
    bvhfile = paths[0]
    g_bvhroot,rootnum = parse_bvh(bvhfile)

    print("File name : " + bvhfile)
    print("Number of frames : " + str(g_num_frame))
    print("FPS : "+str(1/g_frame_time))
    print("Number of joints : " + str(rootnum))
    print("List of all joint names : ")
    g_bvhroot.printall()
    g_current_frame = 0
    g_motion_active = False
    print("bvh loaded")


def button_callback(window, button, action, mod):
    global g_init_point, g_m_left,g_y_change,g_dist,g_x_change,g_cmove_offset,g_v_change,g_u_change,g_rev_toggle,g_rev_change,g_mod_toggle
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            if glfwGetKey(window,GLFW_KEY_LEFT_ALT)==GLFW_PRESS and glfwGetKey(window,GLFW_KEY_LEFT_CONTROL)==GLFW_PRESS:
                g_mod_toggle = 1
            elif glfwGetKey(window,GLFW_KEY_LEFT_ALT)==GLFW_PRESS and glfwGetKey(window,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS:
                g_mod_toggle = 2
            elif glfwGetKey(window,GLFW_KEY_LEFT_ALT)==GLFW_PRESS:
                g_mod_toggle = 3
            g_init_point = list(glfwGetCursorPos(window))
            g_m_left = True
            if g_is_rev == True:
                g_rev_toggle = True
            else:
                g_rev_toggle = False
        elif action==GLFW_RELEASE:
            g_mod_toggle = 0
            g_m_left = False
            g_y_change = 0.
            g_x_change = 0.
            g_v_change = glm.vec3()
            g_u_change = glm.vec3()
            g_rev_change = glm.vec3()




def cursor_callback(window, xpos, ypos):
    global g_dist,g_y_change,g_cam_theta,g_cam_pi,g_x_change,g_cmove_offset,g_v_change,g_u_change,g_is_rev,g_rev_change
    if g_mod_toggle == 1:
        if g_m_left ==True:
            if g_dist + 0.0075*(ypos-g_init_point[1]) > 0.01:
                g_dist = g_dist-g_y_change
                g_y_change = 0.01*(ypos-g_init_point[1])
                g_dist = g_dist + g_y_change
    elif g_mod_toggle ==2:
        if g_m_left ==True:
            if g_rev_toggle == True:
                g_cmove_offset = g_cmove_offset - g_rev_change
            else:
                g_cmove_offset = g_cmove_offset - g_v_change - g_u_change
            g_v_change = g_c_v*0.00085*(ypos-g_init_point[1])*g_dist
            g_u_change = g_c_u*0.00085*(g_init_point[0]-xpos)*g_dist
            g_rev_change = g_u_change + g_v_change

            if g_rev_toggle == True:
                g_rev_change =  glm.mat3x3(1.,0.,0.,
                                  0.,-1.,0.,
                                  0.,0.,1.)*g_rev_change
            g_cmove_offset = g_cmove_offset + g_rev_change

    elif g_mod_toggle == 3:
        if g_m_left ==True:
            g_cam_theta = g_cam_theta - g_y_change
            g_y_change = 0.14*(g_init_point[1]-ypos)
            g_cam_theta = g_cam_theta + g_y_change
            if(g_rev_toggle == False):
                g_cam_pi = g_cam_pi - g_x_change
                g_x_change = 0.15*(xpos-g_init_point[0])
                g_cam_pi = g_cam_pi + g_x_change
            else:
                g_cam_pi = g_cam_pi - g_x_change
                g_x_change = 0.15*(g_init_point[0]-xpos)
                g_cam_pi = g_cam_pi + g_x_change
            if g_cam_theta >180:
                if g_is_rev == False:
                    g_is_rev = True
                    g_cmove_offset = g_cmove_offset * glm.mat3x3(-1,0,0,
                       0,1,0,
                       0,0,-1)
                if g_cam_theta >360:
                    g_cam_theta = g_cam_theta - 360
                    if g_is_rev == False:
                        g_cmove_offset = g_cmove_offset * glm.mat3x3(-1,0,0,
                           0,1,0,
                           0,0,-1)
            elif g_cam_theta <0:
                if g_is_rev == False:
                    g_is_rev = True
                    g_cmove_offset = g_cmove_offset * glm.mat3x3(-1,0,0,
                       0,1,0,
                       0,0,-1)
                    g_cam_theta = g_cam_theta + 360
            else:
                if g_is_rev == True:
                    g_cmove_offset = g_cmove_offset * glm.mat3x3(-1,0,0,
                       0,1,0,
                       0,0,-1)
                    g_is_rev = False
                



# def scroll_callback(window, xoffset, yoffset):
#     print('mouse wheel scroll: %d, %d'%(xoffset, yoffset))

def key_callback(window, key, scancode, action, mods):
    global g_motion_active,g_bonesize,g_ispreloaded,g_bvhroot,g_motion_data,g_current_frame,g_models
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    if key==GLFW_KEY_SPACE and action==GLFW_PRESS:
        g_motion_active = not g_motion_active
    if key==GLFW_KEY_X and action==GLFW_PRESS:
        g_bonesize = g_bonesize + 0.05
    if key==GLFW_KEY_C and action==GLFW_PRESS:
        if g_bonesize > 0.05:
            g_bonesize = g_bonesize - 0.05
    if key==GLFW_KEY_1 and action==GLFW_PRESS:
        g_ispreloaded = not g_ispreloaded
        if g_ispreloaded:
            obj1_callback()
        else:
            g_bvhroot = None
            g_motion_data = []
            g_current_frame = 0
            g_motion_active = False
            g_models.clear()
            print("bvh unloaded")
        

            
            
                


def main():
    # initialize glfw
    global g_c_u,g_c_v,g_cmove_offset,g_current_frame,g_bonesize
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1280, 1280, '2023008413', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks

    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetDropCallback(window, bvhdrop_callback)
    # glfwSetScrollCallback(window, scroll_callback)
    glfwSetKeyCallback(window, key_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    cubes_program = load_shaders(g_cubev_shader_src, g_cubef_shader_src)

    # get uniform locations
    loc_MVP = glGetUniformLocation(shader_program, 'MVP')
    loc_M = glGetUniformLocation(shader_program, 'M')
    loc_view_pos = glGetUniformLocation(shader_program, 'view_pos')
    loc_lightmov = glGetUniformLocation(shader_program, 'lightmove')
    loc_materialcolor = glGetUniformLocation(shader_program, 'material_color')
    
    # prepare vaos
    vao_frame, grid_vertices = prepare_vao_lines(100,1)
    cube_vao = prepare_cube_vao()
    # vao_3dsquare = prepare_vao_3d_square()

    begin_time = glfwGetTime()
    elap_time = 0


    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)


        # projection matrix
        P = glm.perspective(45.,1,0.01,100000)
        #P = glm.ortho(-5,5, -5,5, -100,100)

        upvec = glm.vec3(0,1,0)
        theta = np.radians(g_cam_theta)
        pi = np.radians(g_cam_pi)
        lookpoint = g_l_xyz + g_cmove_offset
        corigin = g_dist*glm.vec3(np.sin(theta)*np.cos(pi),np.cos(theta),np.sin(theta)*np.sin(pi)) + g_cmove_offset
        lightpos = glm.vec3(0,100,0)

        #get v, u vector
        w = (corigin - lookpoint)/np.linalg.norm(corigin-lookpoint)

        # 평행x
        if np.linalg.norm(np.cross(upvec,w))!=0:
            g_c_u = np.cross(upvec,w)/np.linalg.norm(np.cross(upvec,w))
        else:
            g_c_u = upvec
        g_c_v = np.cross(w,g_c_u)


        if g_is_rev == True:
            crev = g_dist*glm.vec3(np.sin(theta)*np.cos(pi+np.pi),np.cos(theta),np.sin(theta)*np.sin(pi+np.pi)) + g_cmove_offset
            V = glm.lookAt(crev, lookpoint, -upvec)
            I = glm.rotate(glm.mat4(1.0), glm.radians(180),glm.vec3(0,1,0))
            corigin = g_dist*glm.vec3(np.sin(theta)*np.cos(pi),np.cos(theta),np.sin(theta)*np.sin(pi)) -g_cmove_offset 
        else:
            V = glm.lookAt(corigin, lookpoint, upvec)
            I = glm.mat4()


        # current frame: P*V*I (now this is the world frame)
        MVP = P*V*I
        M = glm.mat4()

        glUseProgram(shader_program)
        glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(loc_M, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(loc_view_pos, corigin.x,corigin.y,corigin.z)
        glUniform3f(loc_lightmov,lightpos.x,lightpos.y,lightpos.z)
        glUniform3f(loc_materialcolor,1,0,0)
        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, grid_vertices)

        now = glfwGetTime()
        elap_time += now-begin_time
        begin_time = now
        if g_motion_active:
            if elap_time >= g_frame_time:
                g_current_frame = g_current_frame + 1
                elap_time = 0
                if g_num_frame == g_current_frame:
                    g_current_frame = 0
                print(g_current_frame)

        glUseProgram(cubes_program)
        if g_ispreloaded:
            glUseProgram(shader_program)
            frame_data = g_motion_data[g_current_frame] if g_motion_active else None
            draw_bvh_objects(g_bvhroot, glm.mat4(1.0), frame_data, loc_MVP,MVP)
        elif g_bvhroot:
            frame_data = g_motion_data[g_current_frame] if g_motion_active else None
            draw_bvh_cubes(g_bvhroot, glm.mat4(1.0), frame_data, loc_MVP,cube_vao, MVP)


        # T = glm.translate(lookpoint)
        # MVP2 = MVP*T

        # glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP2))
        # glBindVertexArray(vao_3dsquare)
        # glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_INT,None)



        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
