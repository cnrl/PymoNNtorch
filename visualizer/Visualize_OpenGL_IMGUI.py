import torch
import glfw
from OpenGL.GL import *
import numpy as np
from cuda import cudart as cu
import time



from imgui_bundle import imgui,implot
# import imgui
# from imgui.integrations.glfw import GlfwRenderer
import glm


from .IMGUIfunctions import IMGUI
from .FrameBuffer import FrameBuffer

class GUI(IMGUI):
    
    def __init__(self,network, width=1000, heigh=600):
        
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
        for sn in network.SynapseGroups:
            print(sn)

        # self.shows = []
        # for i in range(len(self.network.NeuronGroups)):
        #     self.shows.append(1)
        # self.windows = []
        IMGUI.__init__(self,width, heigh)
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
        self.windows = []
        self.windowsTensors = []
        

        ##  Callback function for window resize
        def framebuffer_size_callback(window, width, height):
            glViewport(0, 0, width, height)
            self.windowWidth = width
            self.windowHeigh = height
            glUniform1f(self.uniform_location_size_data, self.windowWidth/8.5)
        

        


        ## Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        ## Create a GLFW window
        self.window = glfw.create_window(self.windowWidth, self.windowHeigh, "OpenGL Window", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        ## Make the window's context current
        glfw.make_context_current(self.window)

        ## disable vsync
        glfw.swap_interval(0)

        c1=imgui.create_context()
        _ = implot.create_context()
        implot.set_imgui_context(c1)



        self.io = imgui.get_io()
        self.io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard  # Enable Keyboard Controls
        # io.config_flags |= imgui.ConfigFlags_.nav_enable_gamepad # Enable Gamepad Controls
        self.io.config_flags |= imgui.ConfigFlags_.docking_enable  # Enable docking
        self.io.config_flags |= imgui.ConfigFlags_.viewports_enable # Enable Multi-Viewport / Platform Windows


        # self.impl = GlfwRenderer(self.window)
        import ctypes

        # You need to transfer the window address to imgui.backends.glfw_init_for_opengl
        # proceed as shown below to get it.
        window_address = ctypes.cast(self.window, ctypes.c_void_p).value
        imgui.backends.glfw_init_for_opengl(window_address, True)
        glsl_version = "#version 330"
        imgui.backends.opengl3_init(glsl_version)

        ## Set the callback function for window resize
        # glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)








        
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
        # glfw.maximize_window(window)
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
        def cameraKeyboard():
            if (glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS):
                self.Position += self.speed * self.Orientation
                # print(Position)

            if (glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS):
                self.Position += self.speed * -glm.normalize(glm.cross(self.Orientation, self.Up))
            
            if (glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS):
            
                self.Position += self.speed * -self.Orientation
            
            if (glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS):
            
                self.Position += self.speed * glm.normalize(glm.cross(self.Orientation, self.Up))
            
            if (glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS):
            
                self.Position += self.speed * self.Up
            
            if (glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS):
            
                self.Position += self.speed * -self.Up

            if (glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS) or (glfw.get_key(self.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS):
                self.speed = 0.0005*8
            if (glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS) or (glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS):
                self.speed = 0.0005*4
            elif (glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.RELEASE) and (glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.RELEASE) and (glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.RELEASE) and (glfw.get_key(self.window, glfw.KEY_RIGHT_CONTROL) == glfw.RELEASE):
                self.speed = 0.0005

            if (glfw.get_key(self.window, glfw.KEY_P) == glfw.PRESS):
                self.model = glm.scale(self.model,glm.vec3(glm.vec2(1.00001),1.0))
                # model = glm.scale(model,glm.vec3(glm.vec2(1.05001),1.0))
                self.Zoom -= 0.1
            if (glfw.get_key(self.window, glfw.KEY_O) == glfw.PRESS):
                self.model = glm.scale(self.model,glm.vec3(glm.vec2(0.999900009999),1.0))
                # model = glm.scale(model,glm.vec3(glm.vec2(0.89999),1.0))
                self.Zoom += 0.1
                # model = glm.scale(model,glm.vec3(glm.vec2(0.999),1.0))

            if (glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS):
                # model = glm.rotate(model,-speed,glm.vec3(1,0,0))
                self.model = glm.rotate(self.model,-self.speed,glm.normalize(glm.cross(self.Orientation, self.Up)))
            if (glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS):
                # model = glm.rotate(model,speed,glm.vec3(1,0,0))
                self.model = glm.rotate(self.model,self.speed,glm.normalize(glm.cross(self.Orientation, self.Up)))
            if (glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS):
                self.model = glm.rotate(self.model,self.speed,glm.normalize(self.Up))
            if (glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS):
                self.model = glm.rotate(self.model,-self.speed,glm.normalize(self.Up)) 

            if ((glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS) or (glfw.get_key(self.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)) and (glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS):
                self.model = glm.mat4(1.0)
                self.model = glm.translate(self.model, glm.vec3(0.0, 0.0, 0.0))
                self.Position  = glm.vec3(0.0, 0.0, 1.5)
                self.Orientation = glm.vec3(0.0, 0.0, -1.0)
                self.Up = glm.vec3(0.0, 1.0, 0.0)
                self.speed = 0.005
                self.Zoom = 95.0
            # glfw.get_cursor_pos
            # Orientation = glm.vec3(0.0, 0.0, -1.0)


            # projection = glm.ortho(-1,1, -1,1, -1000.0, 1000.0)
            # view = glm.lookAt(Position, Position + Orientation, Up) 
            try:
                self.projection = glm.perspective(glm.radians(self.Zoom), self.windowWidth/self.windowHeigh, 0.1, 100000.0)
                # self.projection = glm.perspective(glm.radians(self.Zoom), 1000/600, 0.1, 100000.0)
            except:
                pass
                # self.projection = glm.perspective(glm.radians(self.Zoom), 1.5, 0.1, 100000.0)

            self.view = glm.lookAt(self.Position, self.Position + self.Orientation, self.Up) 
            # projection = glm.perspective(glm.radians(45.0), 1.0, 0.1, 100.0)
            # view = glm.lookAt(Position, Position + Orientation, Up)

            # imgui.set_next_window_size(500, 500)
            # with imgui.begin("Camera:"):
            #     imgui.text("position:")
            #     imgui.text(str(Position))
            #     imgui.text("projection:")
            #     imgui.text(str(projection))
            #     imgui.text("view:")
            #     imgui.text(str(view))
            #     imgui.text("model:")
            #     imgui.text(str(model))

            glUniformMatrix4fv (self.uniform_location_projection, 1, GL_FALSE, glm.value_ptr(self.projection))
            glUniformMatrix4fv (self.uniform_location_view, 1, GL_FALSE, glm.value_ptr(self.view))
            glUniformMatrix4fv (self.uniform_location_model, 1, GL_FALSE, glm.value_ptr(self.model))

        def addQuad(z):
            glUniform1f(self.uniform_location_z, z)
            glUniform1f(self.uniform_location_isdata, 0.0)
            # glUseProgram(0)
            glUniform1f(self.uniform_location_isplan,1.0)
            glLineWidth(2.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            # glEnable(GL_CULL_FACE)
            # glCullFace(GL_BACK)
            # glFrontFace(GL_CW)  
            # glColor3f(1, 1, 0)
            glBegin(GL_QUADS)
            glVertex3f(-0.99, -0.99,0)
            glVertex3f(-0.99, 0.99,0)
            glVertex3f(0.99, 0.99,0)
            glVertex3f(0.99, -0.99,0)
            glEnd()
            # glFrontFace(GL_CCW )
            # glUseProgram(self.shader_program)
            glUniform1f(self.uniform_location_isplan,0.0)
            # glDisable(GL_BLEND)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        def show_simple_window():

            viewport = imgui.get_main_viewport();
            imgui.set_next_window_pos(viewport.work_pos);
            imgui.set_next_window_size(viewport.work_size);
            imgui.set_next_window_viewport(viewport.id_);
            imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0);
            imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0);
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0,0.0));
            window_flags = imgui.WindowFlags_.menu_bar | imgui.WindowFlags_.no_docking
            window_flags |= imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move;
            window_flags |= imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_nav_focus;
            # print("4444:",window_flags)
            
            window_flags |= imgui.WindowFlags_.no_background
            # print(window_flags)
            imgui.begin("DockSpace Demo", True, window_flags);
            imgui.pop_style_var();
            imgui.pop_style_var(2);
            dockspace_id = imgui.get_id("MyDockSpace");
            # print("cccc:",dockspace_id)
            imgui.dock_space(dockspace_id, imgui.ImVec2(0.0, 0.0), 8);
            imgui.end()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
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
        # glfw.swap_interval(1)
        self.initIcon()
        show_demo_window = True
        show_demo_window2 = True

        self.sceneBuffer = FrameBuffer(self.windowWidth, self.windowHeigh)

        while not glfw.window_should_close(self.window):

            glfw.make_context_current(self.window)
            cameraKeyboard()
            glfw.poll_events()

            imgui.backends.opengl3_new_frame()
            imgui.backends.glfw_new_frame()

            if self.darkMod:
                glClearColor(0.15, 0.16, 0.21, 1.0)
            else:
                glClearColor(0.62, 0.64, 0.70 , 1.0)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    
            # glUniform1f(self.uniform_location_loc_mac_x, -1+(self.selectedX+1)*2/(self.tensorWidth+1))
            # glUniform1f(self.uniform_location_loc_mac_y, -1+(self.selectedY+1)*2/(self.tensorHeight+1))

            # self.impl.process_inputs()
            
            imgui.new_frame()

            show_simple_window()
            if show_demo_window:
                show_demo_window = imgui.show_demo_window(show_demo_window)
            if show_demo_window2:
                show_demo_window2 = implot.show_demo_window(show_demo_window2)

            # print("@@@@@@@2")
            currentTime = time.time()
            timeDiff = currentTime - lastTime
            timeDiff2 = currentTime - lastTime2
            frameNumber += 1
            # tensor += 0.0001
            if self.maxIterationInSecond !=0 and timeDiff >= 1.0 / self.maxIterationInSecond:
                self.network.simulate_iteration()
                self.iteration += 1
                if self.network.device == 'cpu':pass
                else:
                    torch.cuda.synchronize()
                self.oneVertexTrace = np.roll(self.oneVertexTrace, -1)
                self.oneVertexTrace[-1] = self.tensors[self.selectedGroup][self.selectedY][self.selectedX].item()
                iterationPerSecond += 1
                self.ignoreBeforeOneVertexTrace -= 1
                lastTime = currentTime

            if timeDiff2 >= 1.0 / 1:
                glfw.set_window_title(self.window, "IterationsPerSecond: "+str(iterationPerSecond)+" FPS: "+str(int((1.0 / timeDiff2) * frameNumber)))
                frameNumber = 0
                iterationPerSecond = 0
                lastTime2 = currentTime
            
            self.renderGUI()


            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0,0))
            imgui.begin("Scene")
            
            width_scene_imgui = imgui.get_content_region_avail().x
            height_scene_imgui = imgui.get_content_region_avail().y
            if  self.windowWidth != width_scene_imgui or self.windowHeigh != height_scene_imgui:
                self.windowHeigh=height_scene_imgui
                self.windowWidth=width_scene_imgui
                # del (self.sceneBuffer)
                self.sceneBuffer.rescaleFramebuffer(int(width_scene_imgui),int(height_scene_imgui))
                glViewport(0, 0, int(self.windowWidth), int(self.windowHeigh))
                # del(self.sceneBuffer)
                # self.sceneBuffer = FrameBuffer(int(width_scene_imgui),int(height_scene_imgui))
                print("ww: ",self.windowWidth,"hh: ",self.windowHeigh)
                # glViewport(0, 0, int(self.windowWidth), int(self.windowHeigh))

            imgui.image(
                self.sceneBuffer.texture, 
                # imgui.ImVec2(self.windowWidth,self.windowHeigh), 
                imgui.get_content_region_avail(),
                imgui.ImVec2(0, 1), 
                imgui.ImVec2(1, 0)
            )
            # imgui.end_child()
            imgui.end()
            imgui.pop_style_var()
            imgui.render()
            # imgui.begin("SSS")
            # imgui.button("NSSS")
            # imgui.end()

            glBindFramebuffer(GL_FRAMEBUFFER,self.sceneBuffer.fbo)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            
            for n in range(len(self.network.NeuronGroups)):

                if not self.shows[n]:continue


                # X=((torch.reshape(tensor, (1,self.tensorHeights[n]*self.tensorWidths[n]))).squeeze(0)).reshape(self.tensorHeights[n]*self.tensorWidths[n],1)
                X=((torch.reshape(self.tensors[n], (1,self.tensorHeights[n]*self.tensorWidths[n]))).squeeze(0)).reshape(self.tensorHeights[n]*self.tensorWidths[n],1)

                if self.network.device == 'cpu':
                    tens2 = torch.zeros([self.tensorHeights[n]*self.tensorWidths[n],1], dtype=torch.float)
                    X2=torch.cat((X,tens2),1)
                    X3=torch.cat((tens2,X2),1)
                    tens3 = torch.ones([self.tensorHeights[n]*self.tensorWidths[n],1], dtype=torch.float)
                    X4=torch.cat((X3,tens3),1)
                    tensor2=X4

                    glBindTexture(GL_TEXTURE_2D, self.colors[n])
                    # print("n:",n,"XXXXXXXXXXXXXX:",tensor2)

                    # glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA32F, tensorWidth, tensorHeight, 0, GL_RGBA, GL_FLOAT, None)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.tensorWidths[n], self.tensorHeights[n], 0, GL_RGBA, GL_FLOAT, tensor2.numpy())
                    


                else:
                    tens2 = torch.zeros([self.tensorHeights[n]*self.tensorWidths[n],1], dtype=torch.float, device=torch.device('cuda:0'))
                    X2=torch.cat((X,tens2),1)
                    X3=torch.cat((tens2,X2),1)
                    tens3 = torch.ones([self.tensorHeights[n]*self.tensorWidths[n],1], dtype=torch.float, device=torch.device('cuda:0'))
                    X4=torch.cat((X3,tens3),1)
                    tensor2=X4
                    (err,) = cu.cudaGraphicsMapResources(1, self.cuda_images[n], cu.cudaStreamLegacy)
                    err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_images[n], 0, 0)
                    (err,) = cu.cudaMemcpy2DToArrayAsync(
                        array,
                        0,
                        0,
                        tensor2.data_ptr(),
                        4*4*self.tensorWidths[n],
                        4*4*self.tensorWidths[n],
                        self.tensorHeights[n],
                        cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        cu.cudaStreamLegacy,
                    )
                    (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_images[n], cu.cudaStreamLegacy)


                # glEnable(GL_BLEND)
                # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                # glBlendEquation(GL_FUNC_ADD)
                # glBlendColor(1.000, 0.012, 0.012, 1.000)
                # glDisable(GL_POINT_SMOOTH)
                # glDisable(GL_PROGRAM_POINT_SIZE)
                
                glBindVertexArray(self.vaos[n])
                glBindBuffer(GL_ARRAY_BUFFER, self.vbos[n])
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0))
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0+3*4))


                glBindTexture(GL_TEXTURE_2D, self.colors[n])
                #!
                glUniform1f(self.uniform_location_z, -1/2*n) 
                # glBegin(GL_LINE_STRIP) 
                if self.selectedX != -1 and self.selectedY != -1: 
                    # glDrawArrays(GL_POINTS, 0, tensorHeight*tensorWidth+1)
                    ## for big number like million if points will be like square specify more details 
                    ## last vertex show value of seleceted index and show that in circle shape
                    # if self.set_enable_smooth:
                    #     glEnable(GL_POINT_SMOOTH)
                    # else:
                    #     glDisable(GL_POINT_SMOOTH)
                    glUniform1f(self.uniform_location_isdata, 0.0)
                    glEnable(GL_POINT_SMOOTH)
                    glDrawArrays(GL_POINTS, 0, self.tensorHeights[n]*self.tensorWidths[n])
                    glEnable(GL_POINT_SMOOTH)
                    # glUniform1f(self.uniform_location_isdata, 1.0)
                    # glDrawArrays(GL_POINTS, self.tensorHeights[n]*self.tensorWidths[n],1)    
                    # print("@@@@@@@@@@@")
                else: 
                    glUniform1f(self.uniform_location_isdata, 0.0)
                    glDrawArrays(GL_POINTS, 0, self.tensorHeights[n]*self.tensorWidths[n])


                #!
                addQuad(-1/2*n)
                
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
            if self.io.config_flags & imgui.ConfigFlags_.viewports_enable > 0:
                backup_current_context = glfw.get_current_context()
                imgui.update_platform_windows()
                imgui.render_platform_windows_default()
                glfw.make_context_current(backup_current_context)

            # self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

            # imgui.end_frame()


            for w in range(len(self.windows)):
                glfw.make_context_current(self.windows[w])
                glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
                glEnable(GL_PROGRAM_POINT_SIZE)
                # glEnable(GL_DEPTH_TEST)
                # glEnable(GL_BLEND)
                # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)



                glUseProgram(self.shader_program)
                glUniform1f(self.uniform_location_size, 5.0)
                glUniform1f(self.uniform_location_isNewWindow,1.0)
                if self.darkMod:
                    glClearColor(0.15, 0.16, 0.21, 1.0)
                else:
                    glClearColor(0.62, 0.64, 0.70 , 1.0)
                n = self.windowsTensors[w]
                X=((torch.reshape(self.tensors[n], (1,self.tensorHeights[n]*self.tensorWidths[n]))).squeeze(0)).reshape(self.tensorHeights[n]*self.tensorWidths[n],1)
                if self.network.device == 'cpu':pass
                  
                else:
                    tens2 = torch.zeros([self.tensorHeights[n]*self.tensorWidths[n],1], dtype=torch.float, device=torch.device('cuda:0'))
                    X2=torch.cat((X,tens2),1)
                    X3=torch.cat((tens2,X2),1)
                    tens3 = torch.ones([self.tensorHeights[n]*self.tensorWidths[n],1], dtype=torch.float, device=torch.device('cuda:0'))
                    X4=torch.cat((X3,tens3),1)
                    tensor2=X4
                    (err,) = cu.cudaGraphicsMapResources(1, self.cuda_images[n], cu.cudaStreamLegacy)
                    err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_images[n], 0, 0)
                    (err,) = cu.cudaMemcpy2DToArrayAsync(
                        array,
                        0,
                        0,
                        tensor2.data_ptr(),
                        4*4*self.tensorWidths[n],
                        4*4*self.tensorWidths[n],
                        self.tensorHeights[n],
                        cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        cu.cudaStreamLegacy,
                    )
                    (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_images[n], cu.cudaStreamLegacy)
                glBindVertexArray(0)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbos[n])
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0))
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0+3*4))
                glBindTexture(GL_TEXTURE_2D, self.colors[n])
                #!
                glUniform1f(self.uniform_location_z,0.0) 
                # glBegin(GL_LINE_STRIP) 
                if self.selectedX != -1 and self.selectedY != -1: 
                    # glDrawArrays(GL_POINTS, 0, tensorHeight*tensorWidth+1)
                    ## for big number like million if points will be like square specify more details 
                    ## last vertex show value of seleceted index and show that in circle shape
                    # if self.set_enable_smooth:
                    #     glEnable(GL_POINT_SMOOTH)
                    # else:
                    #     glDisable(GL_POINT_SMOOTH)
                    glUniform1f(self.uniform_location_isdata, 0.0)
                    glEnable(GL_POINT_SMOOTH)
                    glDrawArrays(GL_POINTS, 0, self.tensorHeights[n]*self.tensorWidths[n])
                    glEnable(GL_POINT_SMOOTH)
                    # glUniform1f(self.uniform_location_isdata, 1.0)
                    # glDrawArrays(GL_POINTS, self.tensorHeights[n]*self.tensorWidths[n],1) 
                else: 
                    glUniform1f(self.uniform_location_isdata, 0.0)
                    glDrawArrays(GL_POINTS, 0, self.tensorHeights[n]*self.tensorWidths[n])


                glfw.swap_buffers(self.windows[w])
            glUniform1f(self.uniform_location_size, 5.0)
            glUniform1f(self.uniform_location_isNewWindow,0.0)


            

        ## Cleanup
        glDeleteProgram(self.shader_program)
        for n in range(len(self.network.NeuronGroups)):
            glDeleteBuffers(1, [self.vbos[n]])
        # self.impl.shutdown()
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.terminate()