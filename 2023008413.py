from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_cam_theta = 90 
g_cam_pi = -90    ## 0,0,-1 == initial camera point
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

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
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

def prepare_vao_3d_square():
    # prepare vertex data (in main memory)
    
    vertices = glm.array(glm.float32,
        # position        # color
         0.25, -0.25, 0.25,  0.7, 0.7, 0.7, # x-axis start
         0.25, -0.25, -0.25,  0.7, 0.7, 0.7, # x-axis end 
         -0.25, -0.25, -0.25,  0.7, 0.7, 0.7, # x-axis start
         -0.25, -0.25, 0.25,  0.7, 0.7, 0.7, # x-axis end 
         0.25, 0.25, 0.25,  0.7, 0.7, 0.7, # x-axis start
         0.25, 0.25, -0.25,  0.7, 0.7, 0.7, # x-axis end 
         -0.25, 0.25, -0.25,  0.7, 0.7, 0.7, # x-axis start
         -0.25, 0.25, 0.25,  0.7, 0.7, 0.7, # x-axis end 
          
    )
    indexes = glm.array(glm.uint32,
                        
         0,1,2,
         0,2,3,
         0,5,4,
         0,5,1,
         0,7,4,
         0,7,3,
         6,4,5,
         6,4,7,
         6,1,2,
         6,1,5,
         6,3,7,
         6,3,2,

         
    )    


    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO
    EBO = glGenBuffers(1)
    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexes.nbytes,indexes.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO
    
def prepare_vao_lines():
    # prepare vertex data (in main memory)
    
    vertices = glm.array(glm.float32,
        # position        # color
         -5.0, 0.0, 5,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, 5,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, 4,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, 4,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, 3,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, 3,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, 2,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, 2,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, 1,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, 1,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, 0,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, 0,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, -1,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, -1,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, -2,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, -2,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, -3,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, -3,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, -4,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, -4,  0.5, 0.5, 0.5, # x-axis end 
         -5.0, 0.0, -5,  0.5, 0.5, 0.5, # x-axis start
         5.0, 0.0, -5,  0.5, 0.5, 0.5, # x-axis end 

         5, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         5, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end 
         4, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         4, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end
         3, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         3, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end 
         2, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         2, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end 
         1, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         1, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end
         0, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         0, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end 
         -1, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         -1, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end 
         -2, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         -2, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end
         -3, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         -3, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end 
         -4, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         -4, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end 
         -5, 0.0, -5.0,  0.5, 0.5, 0.5, # z-axis start
         -5, 0.0, 5.0,  0.5, 0.5, 0.5, # z-axis end 
          
    )


    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

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
            g_y_change = 0.15*(g_init_point[1]-ypos)
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

# def key_callback(window, key, scancode, action, mods):
#     global g_cam_height,g_cam_ang
#     if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
#         glfwSetWindowShouldClose(window, GLFW_TRUE)

            
            
                


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
    # glfwSetScrollCallback(window, scroll_callback)
    # glfwSetKeyCallback(window, key_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_frame = prepare_vao_lines()
    vao_3dsquare = prepare_vao_3d_square()




    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix
        P = glm.perspective(45.,1,0.01,100)
        #P = glm.ortho(-5,5, -5,5, -100,100)


        upvec = glm.vec3(0,1,0)

        theta = np.radians(g_cam_theta)

        pi = np.radians(g_cam_pi)

        lookpoint = g_l_xyz + g_cmove_offset

        corigin = g_dist*glm.vec3(np.sin(theta)*np.cos(pi),np.cos(theta),np.sin(theta)*np.sin(pi)) + g_cmove_offset


        #get v, u vector
        w = (corigin - lookpoint)/np.linalg.norm(corigin-lookpoint)

        # 평행x
        if np.linalg.norm(np.cross(upvec,w))!=0:
            g_c_u = np.cross(upvec,w)/np.linalg.norm(np.cross(upvec,w))
        g_c_v = np.cross(w,g_c_u)


        if g_is_rev == True:
            crev = g_dist*glm.vec3(np.sin(theta)*np.cos(pi+np.pi),np.cos(theta),np.sin(theta)*np.sin(pi+np.pi)) + g_cmove_offset

            V = glm.lookAt(crev, lookpoint, -upvec)
        else:
            V = glm.lookAt(corigin, lookpoint, upvec)
        # view matrix


        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I

        MVP1 = MVP

        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP1))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 44)

        T = glm.translate(lookpoint)
        MVP2 = MVP*T

        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP2))
        glBindVertexArray(vao_3dsquare)
        glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_INT,None)



        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
