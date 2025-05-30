from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_cam_theta = 90 
g_cam_pi = 90   
g_cmove_offset = glm.vec3(0.,0.,0.)
g_l_xyz = glm.vec3(0.,0.,0.) # initial lookat point
g_c_u = glm.vec3()
g_c_v = glm.vec3()
g_c_w = glm.vec3()
g_m_left = False
g_init_point = list([])
g_model_offset_x = 0.
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
    vec3 light_ambient = 0.1*light_color;
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

# def prepare_vao_3d_square():
#     # prepare vertex data (in main memory)
    
#     vertices = glm.array(glm.float32,
#         # position        # color
#          0.25, -0.25, 0.25,  0.7, 0.7, 0.7, # x-axis start
#          0.25, -0.25, -0.25,  0.7, 0.7, 0.7, # x-axis end 
#          -0.25, -0.25, -0.25,  0.7, 0.7, 0.7, # x-axis start
#          -0.25, -0.25, 0.25,  0.7, 0.7, 0.7, # x-axis end 
#          0.25, 0.25, 0.25,  0.7, 0.7, 0.7, # x-axis start
#          0.25, 0.25, -0.25,  0.7, 0.7, 0.7, # x-axis end 
#          -0.25, 0.25, -0.25,  0.7, 0.7, 0.7, # x-axis start
#          -0.25, 0.25, 0.25,  0.7, 0.7, 0.7, # x-axis end 
          
#     )
#     indexes = glm.array(glm.uint32,
                        
#          0,1,2,
#          0,2,3,
#          0,5,4,
#          0,5,1,
#          0,7,4,
#          0,7,3,
#          6,4,5,
#          6,4,7,
#          6,1,2,
#          6,1,5,
#          6,3,7,
#          6,3,2,

         
#     )    


#     # create and activate VAO (vertex array object)
#     VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
#     glBindVertexArray(VAO)      # activate VAO
#     EBO = glGenBuffers(1)
#     # create and activate VBO (vertex buffer object)
#     VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
#     glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object
#     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
#     # copy vertex data to VBO
#     glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
#     glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexes.nbytes,indexes.ptr, GL_STATIC_DRAW)

#     # configure vertex positions
#     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
#     glEnableVertexAttribArray(0)

#     # configure vertex colors
#     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
#     glEnableVertexAttribArray(1)

#     return VAO
    
def prepare_vao_lines(grid_size=10, grid_spacing=1.0):
    vertices = []

    # Generate lines parallel to the X-axis
    for z in range(-grid_size, grid_size + 1):
        vertices.extend([-grid_size * grid_spacing, 0.0, z * grid_spacing, 0.0, 1.0, 0.0])  # Start point
        vertices.extend([grid_size * grid_spacing, 0.0, z * grid_spacing, 0.0, 1.0, 0.0])   # End point

    # Generate lines parallel to the Z-axis
    for x in range(-grid_size, grid_size + 1):
        vertices.extend([x * grid_spacing, 0.0, -grid_size * grid_spacing, 0.0, 1.0, 0.0])  # Start point
        vertices.extend([x * grid_spacing, 0.0, grid_size * grid_spacing, 0.0, 1.0, 0.0])   # End point

    # Convert to numpy array
    vertices = np.array(vertices, dtype=np.float32)

    # Create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # Create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # Activate VAO

    # Create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # Create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # Activate VBO as a vertex buffer object

    # Copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)  # Allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # Configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)

    # Configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    return VAO, len(vertices) // 6  # Return VAO and number of vertices


class Model:
    def __init__(self, vertices, indices):
        self.index_count = len(indices)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
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

def drop_callback(window, paths):
    global g_models,g_model_offset_x
    name = paths[0].split("\\")[-1]
    print(name)
    vertices, normals, indices, normindices = load_obj(paths[0])
    vn = []
    for i in range(len(indices)):
        vn.extend(vertices[3*indices[i]:3*indices[i]+3])
        vn.extend(normals[3*normindices[i]:3*normindices[i]+3])
    vertices = np.array(vn,dtype=np.float32)
    indices = np.array(indices,dtype=np.uint32)
    for i in range(0, len(vertices), 6):  # x좌표 +2D
        vertices[i] += g_model_offset_x
    model = Model(vertices, indices)
    g_models.append(model)

    g_model_offset_x += 2.0
    


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
            elif glfwGetKey(window,GLFW_KEY_Q)==GLFW_PRESS:
                g_mod_toggle = 1
            elif glfwGetKey(window,GLFW_KEY_W)==GLFW_PRESS:
                g_mod_toggle = 2
            elif glfwGetKey(window,GLFW_KEY_E)==GLFW_PRESS:
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
    
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)

            
            
                


def main():
    # initialize glfw
    global g_c_u,g_c_v,g_cmove_offset
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
    glfwSetDropCallback(window, drop_callback)
    # glfwSetScrollCallback(window, scroll_callback)
    # glfwSetKeyCallback(window, key_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    loc_MVP = glGetUniformLocation(shader_program, 'MVP')
    loc_M = glGetUniformLocation(shader_program, 'M')
    loc_view_pos = glGetUniformLocation(shader_program, 'view_pos')
    loc_lightmov = glGetUniformLocation(shader_program, 'lightmove')
    loc_materialcolor = glGetUniformLocation(shader_program, 'material_color')
    
    # prepare vaos
    vao_frame, grid_vertices = prepare_vao_lines(100,1)
    # vao_3dsquare = prepare_vao_3d_square()

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

        lightpos = glm.vec3(0,1000,0)



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

        # view matrix


        # current frame: P*V*I (now this is the world frame)
        MVP = P*V*I
        M = glm.mat4()

        MVP1 = MVP

        glUseProgram(shader_program)
        glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP1))
        glUniformMatrix4fv(loc_M, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(loc_view_pos, corigin.x,corigin.y,corigin.z)
        glUniform3f(loc_lightmov,lightpos.x,lightpos.y,lightpos.z)
        glUniform3f(loc_materialcolor,1,0,0)
        for model in g_models:
            model.draw()
        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, grid_vertices)
        

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
