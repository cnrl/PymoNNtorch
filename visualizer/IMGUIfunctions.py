from imgui_bundle import imgui,implot


from OpenGL.GL import *

import numpy as np

import glfw
from PIL import Image

from .FrameBuffer import FrameBuffer
from .Camera import Camera
from .NeuronWindow import NeuronWindow

class IMGUI:
    def __init__(self,width=1000, heigh=600):
        # self.windowWidth = width
        # self.windowHeigh = heigh
        self.selectedGroup = 0
        self.selectedX = 0
        self.selectedY = 0
        self.inputValue = 0.0
        self.applyToAll = False
        self.maxIterationInSecond = 0
        self.iteration = 1
        self.oneVertexTrace =np.zeros(100)
        self.ignoreBeforeOneVertexTrace = 100
        self.darkMod = True
        self.debugging = ""
        self.NeuronWindows = []
    def initIcon(self):
        def opentIcon(location):
            img = Image.open(location)
            nrComponents = len(img.getbands())

            format = GL_RED if nrComponents == 1 else \
                    GL_RGB if nrComponents == 3 else \
                    GL_RGBA 
            # print(img)
            textureID = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, textureID)
            glTexImage2D(GL_TEXTURE_2D, 0, format, img.width, img.height, 0, format, GL_UNSIGNED_BYTE, img.tobytes())
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            img.close()
            return textureID
        


        glBindVertexArray(0)


        import os
        self.eyetextureID = [opentIcon(os.path.dirname(os.path.abspath(__file__)).replace("\\","/")+"/icons/icons8-eye-96.png") for _ in range(len(self.network.NeuronGroups))]
        self.closeEyetextureID = [opentIcon(os.path.dirname(os.path.abspath(__file__)).replace("\\","/")+"/icons/icons8-hide-96.png") for _ in range(len(self.network.NeuronGroups))]
        self.arrowtextureID = [opentIcon(os.path.dirname(os.path.abspath(__file__)).replace("\\","/")+"/icons/arrow-up-right-1-16.png") for _ in range(len(self.network.NeuronGroups))]
        self.settingtextureID = [opentIcon(os.path.dirname(os.path.abspath(__file__)).replace("\\","/")+"/icons/icons8-info-100.png") for _ in range(len(self.network.NeuronGroups))]

        # self.eyetextureID = [opentIcon("./visualizer/icons/icons8-eye-96(2).png") for _ in range(len(self.network.NeuronGroups))]
        # self.closeEyetextureID = [opentIcon("./visualizer/icons/icons8-hide-96.png") for _ in range(len(self.network.NeuronGroups))]
        # self.arrowtextureID = [opentIcon("./visualizer/icons/arrow-up-right-1-16.png") for _ in range(len(self.network.NeuronGroups))]
        # self.settingtextureID = [opentIcon("./visualizer/icons/icons8-info-100.png") for _ in range(len(self.network.NeuronGroups))]

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        # glBindVertexArray(self.vao)
        # glBindTexture(GL_TEXTURE_2D, self.color)




    def Begin(self):
        glfw.poll_events()

        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()

        if self.darkMod:
            glClearColor(0.15, 0.16, 0.21, 1.0)
        else:
            glClearColor(0.62, 0.64, 0.70 , 1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        imgui.new_frame()

        self.EnableDoking()
        if self.show_demo_window:
            self.show_demo_window = imgui.show_demo_window(self.show_demo_window)
        if self.show_demo_window2:
            self.show_demo_window2 = implot.show_demo_window(self.show_demo_window2)
            
    def End(self):
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        if self.io.config_flags & imgui.ConfigFlags_.viewports_enable > 0:
            backup_current_context = glfw.get_current_context()
            imgui.update_platform_windows()
            imgui.render_platform_windows_default()
            glfw.make_context_current(backup_current_context)
        glfw.swap_buffers(self.glfw_window)

    def RenderGUI(self):
        # imgui.set_next_window_position(0, 18)
        # imgui.set_next_window_size(self.windowWidth/3, self.windowHeigh/2)
        imgui.begin("Debugging:")
        # imgui.text(self.debugging)
        for i in range(len(self.network.NeuronGroups)):
            if (imgui.button("N"+str(i))):
                self.debugging=""
                self.debugging += str(self.network.NeuronGroups[i].trace)
            imgui.same_line()
        imgui.separator()

        changed, self.debugging = imgui.input_text_multiline(
            'Message:',
            self.debugging,
            size = imgui.ImVec2(0, 0),
            )
        imgui.end()



        # imgui.set_next_window_position(0, 18)
        # imgui.set_next_window_size(self.windowWidth/8, self.windowHeigh/3-18)
        # flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
        imgui.begin("Iteration:")
            # imgui.text("N:"+str(self.tensorWidth*self.tensorHeight))
        imgui.text("Iteration:"+str(self.iteration))
        imgui.text("Speed :")
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            if imgui.begin_tooltip():
                imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                imgui.text_unformatted("Maximum iteration per second")
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
        _, self.maxIterationInSecond = imgui.input_int(' ', self.maxIterationInSecond)


        imgui.separator()
        imgui.text("N:"+str(sum([len(ng.trace) for ng in self.network.NeuronGroups])))
        imgui.text("Group:")
        changed0, self.selectedGroup = imgui.input_int('  ', self.selectedGroup)
        if self.selectedGroup <= -1 : self.selectedGroup = len(self.network.NeuronGroups)- 1
        if self.selectedGroup >= len(self.network.NeuronGroups) : self.selectedGroup = 0
        imgui.text("ind:")
        changed1, self.selectedXY = imgui.input_int('   ', self.selectedY*self.tensorWidths[self.selectedGroup]+self.selectedX)
        if  changed1:
            self.selectedY = self.selectedXY // self.tensorWidths[self.selectedGroup]
            self.selectedX = self.selectedXY % self.tensorWidths[self.selectedGroup]
            # self.oneVertexTrace =np.zeros(100)
            # self.ignoreBeforeOneVertexTrace = 100
            # print("22")
        imgui.text("row:")
        changed2, self.selectedY = imgui.input_int('    ', self.selectedY)
        imgui.text("col:")
        changed3, self.selectedX = imgui.input_int('     ', self.selectedX)
        if changed0 or changed1 or changed2 or changed3:
            self.oneVertexTrace =np.zeros(100)
            self.ignoreBeforeOneVertexTrace = 100
        # if (imgui.button("  Trace\nindividualy",self.windowWidth/9,50)):pass

        if self.selectedY <= -1:self.selectedY = self.tensorHeights[self.selectedGroup]- 1
        if self.selectedY >= self.tensorHeights[self.selectedGroup] : self.selectedY = 0 
        if self.selectedX <= -1 : self.selectedX = self.tensorWidths[self.selectedGroup]-1 
        if self.selectedX >= self.tensorWidths[self.selectedGroup] : self.selectedX = 0 
        imgui.end()



        # if imgui.begin_main_menu_bar():
        #     if imgui.begin_menu('Setting', True):
        #         if imgui.begin_menu('Style', True):
        #             if imgui.menu_item('Dark', None, False, True)[0]:
        #                 self.darkMod = True
        #                 imgui.style_colors_dark()
        #             if imgui.menu_item('Light', None, False, True)[0]:
        #                 self.darkMod = False
        #                 imgui.style_colors_light()
        #             imgui.end_menu()
        #         imgui.end_menu()
        # imgui.end_main_menu_bar()


        imgui.begin("NeuronGroups:")
        for i in range(len(self.network.NeuronGroups)):
            imgui.text("NeuronGroup {0}".format(str(i)))
            # imgui.same_line()
            if self.shows[i]:
                # if i>2:

                if (imgui.image_button("hide"*(i+1),self.eyetextureID[i],imgui.ImVec2(25,25))):
                    self.shows[i]=0
                    pass
            else:
                # if (imgui.button("<",25,25)):
                if (imgui.image_button("show"*(i+1),self.closeEyetextureID[i],imgui.ImVec2(25,25))):
                    # print(i)
                    self.shows[i]=1
                    pass
            imgui.same_line(spacing=1.0)
            # if (imgui.button("+",25,25)):
            if (imgui.image_button("new_window"*(i+1),self.arrowtextureID[i],imgui.ImVec2(25,25))):
                newWindow = NeuronWindow(800,600,self,i)
                self.NeuronWindows.append(newWindow)
            imgui.same_line(spacing=1.0)
            if (imgui.image_button("info"*(i+1),self.settingtextureID[i],imgui.ImVec2(25,25))):
                pass
        imgui.end()


        imgui.begin("Information:")
        imgui.text("Value:")
        imgui.text(str(self.tensors[self.selectedGroup][self.selectedY][self.selectedX].item()))          
        imgui.end()

        imgui.begin("Change:")
        imgui.text("New value:")
        _, self.inputValue = imgui.input_float("##", self.inputValue)
        if (imgui.button("Change",imgui.ImVec2(100,25))):
            self.tensors[self.selectedGroup][self.selectedY][self.selectedX]=self.inputValue
        _,self.applyToAll = imgui.checkbox("Change value \n if has been \n selected", self.applyToAll)
        if self.applyToAll:
            self.tensors[self.selectedGroup][self.selectedY][self.selectedX]=self.inputValue 
        imgui.end()


        imgui.begin("Trace")
        xxxx = np.arange(0, 101, 1.0)
        if implot.begin_plot("Plot"):
            implot.setup_axes("iteration", "trace G:%s, R:%s, C:%s"%(self.selectedGroup,self.selectedY,self.selectedX));
            # print("x:",x)
            # print("y:",self.oneVertexTrace)
            # implot.setup_axes_limits(0, 100, 0, 500);

            implot.plot_line("Trace",xxxx, self.oneVertexTrace,101)
            implot.set_next_marker_style(implot.Marker_.circle, 2, implot.get_colormap_color(1), -1, implot.get_colormap_color(1));
            implot.plot_scatter("Trace",xxxx, self.oneVertexTrace,100)
            # implot.plot_line("y2", x, y2)
            implot.end_plot()
        imgui.end()

        self.ImGuiAddWindow(self.MainWindow,"Scene",False)
        for w in range(len(self.NeuronWindows)): 
            isClose = self.ImGuiAddWindow(self.NeuronWindows[w],"Scene %s N:%s"%(str(w),self.NeuronWindows[w].NeuronIndex),True)
            if not isClose:
                self.NeuronWindows.pop(w)
                break
    def EnableDoking(self):
            viewport = imgui.get_main_viewport()
            imgui.set_next_window_pos(viewport.work_pos)
            imgui.set_next_window_size(viewport.work_size)
            imgui.set_next_window_viewport(viewport.id_)
            imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
            imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0,0.0))
            window_flags = imgui.WindowFlags_.menu_bar | imgui.WindowFlags_.no_docking
            window_flags |= imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
            window_flags |= imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_nav_focus
            window_flags |= imgui.WindowFlags_.no_background
            imgui.begin("DockSpace Demo", True, window_flags)
            imgui.pop_style_var()
            imgui.pop_style_var(2)
            dockspace_id = imgui.get_id("MyDockSpace")
            imgui.dock_space(dockspace_id, imgui.ImVec2(0.0, 0.0), 8)
            imgui.end()
    def ImGuiAddWindow(self,window,text,P_open = False):
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0,0))
        if P_open:
            window.show,isClose = imgui.begin(text,p_open=True)
        else:
            window.show,isClose = imgui.begin(text)
        if imgui.is_window_focused():
            window.cameraInput()
        width_window_imgui,height_window_imgui = imgui.get_content_region_avail()
        if  window.width != width_window_imgui or window.height != height_window_imgui:
            window.width, window.height = width_window_imgui, height_window_imgui
            window.frameBuffer.rescaleFramebuffer(int(width_window_imgui),int(height_window_imgui))
        imgui.image(
            window.frameBuffer.texture, 
            imgui.ImVec2(width_window_imgui,height_window_imgui),
            imgui.ImVec2(0, 1), 
            imgui.ImVec2(1, 0)
        )
        imgui.end()
        imgui.pop_style_var()
        return isClose