import torch
import glfw
from OpenGL.GL import *
import numpy as np
import time



from imgui_bundle import imgui,implot
# import imgui
# from imgui.integrations.glfw import GlfwRenderer
import glm


from .IMGUIfunctions import IMGUI
from .FrameBuffer import FrameBuffer
from .Camera import Camera
from .MainWindow import MainWindow

class GUI(IMGUI):
    
    def __init__(self,network, width=1000, height=600):
        self.width,self.height = width,height
        self.network = network
        self.network.simulate_iteration()
        self.network.clear_recorder()
        # for ng in network.NeuronGroups:
        #     # self.tensor = getattr(ng,'trace')
        #     self.tensor = ng.trace
        #     print(self.tensor.shape)
        #     # self.tensor= network.NeuronGroups[1].trace
        #     self.tensorSize = self.tensor.shape[0]
        #     self.tensor=torch.reshape(self.tensor,find_numbers_close_to_sqrt(self.tensorSize))
        #     self.tensorHeight,self.tensorWidth = self.tensor.shape
        # for sn in network.SynapseGroups:
        #     print(sn)
        
        IMGUI.__init__(self,width, height)
        ## Vertex shader source code


        import os
        vShaderFile = open(os.path.dirname(os.path.abspath(__file__)).replace("\\","/")+"/default.vs")
        # vShaderFile = open("./visualizer/default.vs")
        self.VERTEX_SHADER = vShaderFile.read()
        vShaderFile.close()
        ## Fragment shader source code
        fShaderFile = open(os.path.dirname(os.path.abspath(__file__)).replace("\\","/")+"/default.fs")
        # fShaderFile = open("./visualizer/default.fs")
        self.FRAGMENT_SHADER = fShaderFile.read()
        fShaderFile.close()
    def initializeOpenGL(self):
        def find_numbers_close_to_sqrt(c):
            sqrt_c = int(c ** 0.5)
            a = sqrt_c
            b = c // a
            while a * b != c:
                a -= 1
                b = c // a
            return b, a
        self.shows = []
        

        ##  Callback function for window resize
        # def framebuffer_size_callback(window, width, height):
        #     glViewport(0, 0, width, height)
        #     self.windowWidth = width
        #     self.windowHeigh = height
        #     glUniform1f(self.uniform_location_size_data, self.windowWidth/8.5)
        

        


        ## Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        ## Create a GLFW window
        self.glfw_window = glfw.create_window(self.width, self.height, "OpenGL Window", None, None)
        if not self.glfw_window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        ## Make the window's context current
        glfw.make_context_current(self.glfw_window)

        ## disable vsync
        glfw.swap_interval(0)

        c1=imgui.create_context()
        _ = implot.create_context()
        implot.set_imgui_context(c1)



        self.io = imgui.get_io()
        self.io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard  # Enable Keyboard Controls
        self.io.config_flags |= imgui.ConfigFlags_.docking_enable  # Enable docking
        self.io.config_flags |= imgui.ConfigFlags_.viewports_enable # Enable Multi-Viewport / Platform Windows


        # self.impl = GlfwRenderer(self.glfw_window)
        import ctypes

        # You need to transfer the window address to imgui.backends.glfw_init_for_opengl
        # proceed as shown below to get it.
        window_address = ctypes.cast(self.glfw_window, ctypes.c_void_p).value
        imgui.backends.glfw_init_for_opengl(window_address, True)
        glsl_version = "#version 330"
        imgui.backends.opengl3_init(glsl_version)

        ## Set the callback function for window resize
        # glfw.set_framebuffer_size_callback(self.glfw_window, framebuffer_size_callback)








        
        ## Create and compile the vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, self.VERTEX_SHADER)
        glCompileShader(vertex_shader)
        ## Create and compile the fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, self.FRAGMENT_SHADER)
        glCompileShader(fragment_shader)
        ## Create the shader program and link the shaders
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        ## Delete the shaders (they are no longer needed)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        ## find uniform locations
        self.uniform_location_isdata = glGetUniformLocation(self.shader_program, "uIsdata")
        self.uniform_location_z = glGetUniformLocation(self.shader_program, "uZ")
        self.uniform_location_size = glGetUniformLocation(self.shader_program, "uSize")
        self.uniform_location_size_data = glGetUniformLocation(self.shader_program, "uSizeData")
        self.uniform_location_locx = glGetUniformLocation(self.shader_program, "uLocx")
        self.uniform_location_locy = glGetUniformLocation(self.shader_program, "uLocy")
        self.uniform_location_loc_mac_x = glGetUniformLocation(self.shader_program, "uLocMacanx")
        self.uniform_location_loc_mac_y = glGetUniformLocation(self.shader_program, "uLocMacany")
        self.uniform_location_projection = glGetUniformLocation(self.shader_program,"projection")
        self.uniform_location_view = glGetUniformLocation(self.shader_program,"view")
        self.uniform_location_model = glGetUniformLocation(self.shader_program,"model")
        self.uniform_location_isplan = glGetUniformLocation(self.shader_program,"uPlan")
        self.uniform_location_isNewWindow = glGetUniformLocation(self.shader_program,"uisNewWindow")

        self.vaos = glGenVertexArrays(len(self.network.NeuronGroups))
        self.vbos = glGenBuffers(len(self.network.NeuronGroups))
        self.colors = glGenTextures(len(self.network.NeuronGroups))
        self.cuda_images = []
        self.tensors = []
        self.tensorHeights = []
        self.tensorWidths = []
        for n in range(len(self.network.NeuronGroups)):
            self.shows.append(1)
            tensor = self.network.NeuronGroups[n].trace
            # tensor = self.network.NeuronGroups[n].spikes
            print("NeuronGroup:{0} shape:{1}".format(str(n),tensor.shape))
            tensorSize = tensor.shape[0]
            tensor=torch.reshape(tensor,find_numbers_close_to_sqrt(tensorSize))
            self.tensors.append(tensor)
            tensorHeight,tensorWidth = tensor.shape
            self.tensorHeights.append(tensorHeight)
            self.tensorWidths.append(tensorWidth)
            ## example : 
            # vertices = np.array([
            #         #positions                     // texture coords
            #         -0.5,  0.33, 0.0,      0.25,0.33,     #0,0
            #         0.0,  0.33, 0.0,      0.50,0.33,     #1.0, 
            #         0.5,  0.33, 0.0,      0.75,0.33,     #0.0, 
            #         -0.5, -0.33, 0.0,      0.25,0.66,     #1.0, 
            #         0.0, -0.33, 0.0,      0.50,0.66,     #0.0, 
            #         0.5, -0.33, 0.0,      0.75,0.66,     #1.0, 
            
            # ], dtype=np.float32)
            
            vertices = np.zeros((tensorHeight*tensorWidth*5+5),dtype=np.float32)
            index=0
            for i in range(tensorWidth):
                for j in range(tensorHeight):
                    # vertices[index+0] = -1+1/4+(i+1)*2*(6/8)/(tensorWidth+1)
                    vertices[index+0] = -1+(i+1)*2/(tensorWidth+1)
                    vertices[index+1] = -1+(j+1)*2/(tensorHeight+1)
                    vertices[index+2] = 0.0
                    vertices[index+3] = (i+1)/(tensorWidth+1)
                    vertices[index+4] = (j+1)/(tensorHeight+1)
                    index +=5
            vertices[index+0] = 7/8
            vertices[index+1] = 0.0
            vertices[index+2] = -1.0
            vertices[index+3] = -1+(0+1)/(tensorWidth+1)
            vertices[index+4] = -1+(0+1)/(tensorHeight+1)

            #!!! Genarate texture that taxture will have cuda tensor values
            glBindTexture(GL_TEXTURE_2D, self.colors[n])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA32F, tensorWidth, tensorHeight, 0, GL_RGBA, GL_FLOAT, None)
            glBindTexture(GL_TEXTURE_2D, 0)


            if self.network.device == 'cpu':pass
            else:
                from cuda import cudart as cu
                err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
                # !!! Register
                err, cuda_image = cu.cudaGraphicsGLRegisterImage(
                    self.colors[n],
                    GL_TEXTURE_2D, 
                    cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
                    )
                
                self.cuda_images.append(cuda_image)

            glBindVertexArray(self.vaos[n])
            glBindBuffer(GL_ARRAY_BUFFER, self.vbos[n])
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0+3*4))


        ## maximize at start
        # glfw.maximize_window(glfw_window)
        ## points shape change from square to circle
        glEnable(GL_PROGRAM_POINT_SIZE)
        self.set_enable_smooth = True

        glUseProgram(self.shader_program)
        glUniform1f(self.uniform_location_size, 5.0)

        # if self.tensorHeight*self.tensorWidth<=100:
        #     glUniform1f(self.uniform_location_size, 40.0)
        # elif self.tensorHeight*self.tensorWidth<=200:
        #     glUniform1f(self.uniform_location_size, 25.0)
        # elif self.tensorHeight*self.tensorWidth<=10000:
        #     glUniform1f(self.uniform_location_size, 5.0)
        # else:
        #     glUniform1f(self.uniform_location_size, 0.25)
        #     self.set_enable_smooth = False
        # glUniform1f(self.uniform_location_size_data, self.windowWidth/8.5)


        self.MainWindow = MainWindow(self.width, self.height, self)
        print("INITGL DONE")

        self.renderOpenGL()
    def renderOpenGL(self):
        # io = imgui.get_io()
        # io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard  # Enable Keyboard Controls
        # # io.config_flags |= imgui.ConfigFlags_.nav_enable_gamepad # Enable Gamepad Controls
        # io.config_flags |= imgui.ConfigFlags_.docking_enable  # Enable docking
        # io.config_flags |= imgui.ConfigFlags_.viewports_enable # Enable Multi-Viewport / Platform Windows
        
        # io.config_viewports_no_auto_merge = True
        # io.config_viewports_no_task_bar_icon = True

        # Setup Dear ImGui style
        imgui.style_colors_dark()

        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        ## Render loop
        lastTime = time.time()
        lastTime2 = lastTime
        frameNumber = 0
        iterationPerSecond = 0
        self.model = glm.mat4(1.0)
        self.model = glm.translate(self.model, glm.vec3(0.0, 0.0, 0.0))
        self.Position  = glm.vec3(0.0, 0.0, 1.5)
        self.Orientation = glm.vec3(0.0, 0.0, -1.0)
        self.Up = glm.vec3(0.0, 1.0, 0.0)
        self.speed = 0.005
        self.Zoom = 95.0
        self.camera = Camera(self)
        # glfw.swap_interval(1)
        self.initIcon()
        self.show_demo_window = True
        self.show_demo_window2 = True

        while not glfw.window_should_close(self.glfw_window):
            self.Begin()

            currentTime = time.time()
            timeDiff = currentTime - lastTime
            timeDiff2 = currentTime - lastTime2
            frameNumber += 1
            # tensor += 0.0001

            if self.maxIterationInSecond !=0 and timeDiff >= 1.0 / self.maxIterationInSecond:
                self.network.simulate_iteration()
                self.iteration += 1
                if self.network.device == 'cpu':pass
                else:torch.cuda.synchronize()
                self.oneVertexTrace = np.roll(self.oneVertexTrace, -1)
                self.oneVertexTrace[-1] = self.tensors[self.selectedGroup][self.selectedY][self.selectedX].item()
                iterationPerSecond += 1
                self.ignoreBeforeOneVertexTrace -= 1
                lastTime = currentTime

            if timeDiff2 >= 1.0 / 1:
                glfw.set_window_title(self.glfw_window, "IterationsPerSecond: "+str(iterationPerSecond)+" FPS: "+str(int((1.0 / timeDiff2) * frameNumber)))
                frameNumber = 0
                iterationPerSecond = 0
                lastTime2 = currentTime
            
            self.RenderGUI()


            imgui.render()

            self.MainWindow.OnUpdate()

            for w in range(len(self.NeuronWindows)):
                self.NeuronWindows[w].OnUpdate()

            self.End()
            

        ## Cleanup
        glDeleteProgram(self.shader_program)
        for n in range(len(self.network.NeuronGroups)):
            glDeleteBuffers(1, [self.vbos[n]])
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.terminate()