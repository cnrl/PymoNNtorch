from imgui_bundle import imgui,implot


from OpenGL.GL import *

import numpy as np

import glfw
from PIL import Image

from .FrameBuffer import FrameBuffer

class IMGUI:
    def __init__(self,width=1000, heigh=600):
        self.windowWidth = width
        self.windowHeigh = heigh
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
        self.frameBufferWindows = []

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




        
    def renderGUI(self):
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
            # pass
            # if main_menu_bar.opened:
                # first menu dropdown
                # with imgui.begin_menu('Setting', True) as setting_menu:
                #     if setting_menu.opened:
                #         with imgui.begin_menu('Style', True) as style_menu:
                #             if style_menu.opened:
                #                 if imgui.menu_item('Dark', None, False, True)[0]:
                #                     self.darkMod = True
                #                     imgui.style_colors_dark()
                #                 if imgui.menu_item('Light', None, False, True)[0]:
                #                     self.darkMod = False
                #                     imgui.style_colors_light()

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
                print("!!",i)

                newWindow = FrameBuffer(800, 600)
                self.frameBufferWindows.append(newWindow)

                #pervious method (multi glfw window)
                # newWindow = glfw.create_window(800, 600, "NeuronGroup "+str(i), None, self.window)
                # if not newWindow:
                #     glfw.terminate()
                #     raise Exception("New window creation failed")
                # glfw.make_context_current(newWindow)
                # def framebuffer_size_callback(window, width, height):
                #     glViewport(0, 0, width, height)
                # glfw.set_framebuffer_size_callback(newWindow, framebuffer_size_callback)
                # glfw.swap_interval(0)
                # self.windows.append(newWindow)
                self.windowsTensors.append(i)
                # glfw.make_context_current(self.window)
            imgui.same_line(spacing=1.0)
            if (imgui.image_button("info"*(i+1),self.settingtextureID[i],imgui.ImVec2(25,25))):
                pass
            # if (imgui.button("x",25,25)):
            #     pass
                # glfw.destroy_window(self.window2)
            # try:
            #     if glfw.window_should_close(self.window2):
            #         glfw.destroy_window(self.window2)
            # except:
            #     pass

            for w in range(len(self.windows)):
                try:
                    if glfw.window_should_close(self.windows[w]):
                            glfw.destroy_window(self.windows[w])
                            self.windows.pop(w)
                            self.windowsTensors.pop(w)
                except:
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
            implot.setup_axes("iteration", "trace G:%s, R:%s, C:%s"%(self.selectedGroup,self.selectedX,self.selectedY));
            # print("x:",x)
            # print("y:",self.oneVertexTrace)
            # implot.setup_axes_limits(0, 100, 0, 500);

            implot.plot_line("Trace",xxxx, self.oneVertexTrace,101)
            implot.set_next_marker_style(implot.Marker_.circle, 2, implot.get_colormap_color(1), -1, implot.get_colormap_color(1));
            implot.plot_scatter("Trace",xxxx, self.oneVertexTrace,100)
            # implot.plot_line("y2", x, y2)
            implot.end_plot()
        imgui.end()
    def XX(self):

            # glUniform1f(self.uniform_location_locx, 1/(self.tensorWidth+1)*(self.selectedX+1))
            # glUniform1f(self.uniform_location_locy, 1/(self.tensorHeight+1)*(self.selectedY+1))
            imgui.set_next_window_position(0, 18)
            imgui.set_next_window_size(self.windowWidth/8, self.windowHeigh/3-18)
            flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
            with imgui.begin("Iteration:",flags=flags):
                    # imgui.text("N:"+str(self.tensorWidth*self.tensorHeight))
                    imgui.text("Iteration:"+str(self.iteration))


                    
                        # imgui.end_tooltip();

                    imgui.text("Speed :")
                    imgui.same_line()
                    imgui.text_disabled("(?)");
                    if imgui.is_item_hovered():
                        with imgui.begin_tooltip():
                            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
                            imgui.text_unformatted("Maximum iteration per second")
                            imgui.pop_text_wrap_pos()
                    _, self.maxIterationInSecond = imgui.input_int(' ', self.maxIterationInSecond)


                    imgui.separator()
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
                    # if self.tensorWidth*self.tensorHeight<=20000:
                    #     pass
                    #     # for i in range(self.tensorHeight):
                    #     #     for j in range(self.tensorWidth):
                    #     #         if i == self.selectedY and j==self.selectedX:
                    #     #             style.colors[imgui.COLOR_BUTTON] = (0.03, 0.07, 0.22, 1.0)
                    #     #             style.colors[imgui.COLOR_TEXT] = (1.0, 0.0, 0.0, 1.0)
                    #     #         else:
                    #     #             style.colors[imgui.COLOR_BUTTON] = (0.13, 0.27, 0.42, 1.0)
                    #     #             style.colors[imgui.COLOR_TEXT] = (1.0, 1.0, 1.0, 1.0)
                    #     #         if (imgui.button("v_"+str(i*self.tensorHeight+j),100,25)):
                    #     #             self.oneVertexTrace =np.zeros(100)
                    #     #             self.ignoreBeforeOneVertexTrace = 100
                    #     #             self.selectedX = j
                    #     #             self.selectedY = i
                    # else:
                    #     pass
                    #     # imgui.text("Generate over \n 20000 button \n in python take \n alot of time\n and drop fps")     


                    imgui.new_line()
            imgui.set_next_window_position(0, self.windowHeigh/3)
            imgui.set_next_window_size(self.windowWidth/8, 2*self.windowHeigh/3)
            flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
            # style.colors[imgui.COLOR_BUTTON] = (0.13, 0.27, 0.42, 1.0)
            # style.colors[imgui.COLOR_TEXT] = (1.0, 1.0, 1.0, 1.0)

            imgui.set_next_window_position(self.windowWidth/8, 18)
            imgui.set_next_window_size(self.windowWidth-2*self.windowWidth/8, self.windowHeigh)
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.01)
            flags = imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS 
            flags = imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS|imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
            with imgui.begin("output",flags=flags):
                pass
                # if imgui.core.is_window_focused() and imgui.core.is_window_hovered():
                #     if imgui.is_mouse_down():
                #         first_mouse_x, first_mouse_y = imgui.get_mouse_pos()
                #         if (first_mouse_x >= self.windowWidth/8 and first_mouse_x <= 7*self.windowWidth/8) and (first_mouse_y >= 0 and first_mouse_y <= self.windowHeigh):
                #             self.oneVertexTrace =np.zeros(100)
                #             self.ignoreBeforeOneVertexTrace = 100
                #             mouse_x = first_mouse_x
                #             mouse_y = first_mouse_y
                #             self.selectedX = math.ceil(math.floor((mouse_x - 1/8*self.windowWidth) / ((6/8)*self.windowWidth/(2*(self.tensorWidth+1)))) / 2)-1
                #             if self.selectedX <= -1: self.selectedX = 0
                #             if self.selectedX >= self.tensorWidth: self.selectedX = self.tensorWidth-1
                #             self.selectedY =  self.tensorHeight - math.ceil(math.floor(mouse_y/(self.windowHeigh/(2*(self.tensorHeight+1)))) / 2)
                #             if self.selectedY <= -1: self.selectedY = 0
                #             if self.selectedY >= self.tensorHeight: self.selectedY = self.tensorHeight-1

                # draw_list = imgui.get_window_draw_list()
                # thicknes = 5
                # if self.tensorHeight*self.tensorWidth<=100:
                #     size = 30
                # elif self.tensorHeight*self.tensorWidth<=200:
                #     size = 25
                # elif self.tensorHeight*self.tensorWidth<=10000:
                #     thicknes = 2
                #     size = 5
                # else:
                #     thicknes = 2
                #     size = 4
                # posCircleX = 1/8 * self.windowWidth + (6/8) * self.windowWidth / (self.tensorWidth+1) * (self.selectedX+1)
                # posCircleY = self.windowHeigh/(self.tensorHeight+1) * (self.tensorHeight-self.selectedY)
                # draw_list.add_circle(posCircleX, posCircleY, size, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 100.0),100, thicknes)
                ## invisible button
                # style.colors[imgui.COLOR_BUTTON] = (0.03, 0.07, 0.22, 0.0)
                # style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.03, 0.07, 0.22, 0.0)
                # style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.03, 0.07, 0.22, 0.0)
                # if (imgui.button("",width_window-2*width_window/8,(97/100)*self.windowHeigh)):
            imgui.pop_style_var(1)

            imgui.set_next_window_position(7*self.windowWidth/8, 2*self.windowHeigh/3)
            imgui.set_next_window_size(self.windowWidth/8, self.windowHeigh/3)
            flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
            
            imgui.set_next_window_size(self.windowWidth/4, self.windowHeigh/3)
            # with imgui.begin("Trace"):#,flags=flags):
            #     draw_list = imgui.get_window_draw_list()
            #     thicknes = 1
            #     size = 1
            #     posCircleX ,posCircleY = imgui.get_window_position()
            #     sizeX ,sizeY = imgui.get_window_size()
            #     posCircleX ,posCircleY = posCircleX, posCircleY + sizeY/2 
            #     distance_in_y = sizeY/25
            #     text_position = (posCircleX ,posCircleY)
            #     imgui.set_next_window_size(0, 0)

            #     draw_list.add_text(posCircleX+5,posCircleY-10*distance_in_y,
            #                 imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.3), "+10")
            #     draw_list.add_text(posCircleX+5,posCircleY-5*distance_in_y,
            #                 imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.3), "+5")
            #     draw_list.add_text(posCircleX+5,posCircleY-0*distance_in_y,
            #                 imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.3), "0")
            #     draw_list.add_text(posCircleX+5,posCircleY+5*distance_in_y,
            #                 imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.3), "-5")
            #     draw_list.add_text(posCircleX+5,posCircleY+10*distance_in_y,
            #                 imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.3), "-10")

            #     draw_list.add_line( posCircleX, posCircleY-10*distance_in_y,
            #                         posCircleX+sizeX, posCircleY-10*distance_in_y, 
            #                         imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.1), thicknes)
            
            #     draw_list.add_line( posCircleX, posCircleY-5*distance_in_y,
            #                         posCircleX+sizeX,posCircleY-5*distance_in_y, 
            #                         imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.1), thicknes)
            #     draw_list.add_line( posCircleX, posCircleY-0*distance_in_y,
            #                         posCircleX+sizeX,posCircleY-0*distance_in_y,
            #                         imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.1), thicknes)
            #     draw_list.add_line( posCircleX, posCircleY+5*distance_in_y,
            #                         posCircleX+sizeX,posCircleY+5*distance_in_y, 
            #                         imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.1), thicknes)
            #     draw_list.add_line( posCircleX, posCircleY+10*distance_in_y,
            #                         posCircleX+sizeX, posCircleY+10*distance_in_y,
            #                         imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.1), thicknes)
                
                
            #     draw_list.add_circle_filled(posCircleX+(99)*sizeX/102, posCircleY-distance_in_y*self.tensor[self.selectedY][self.selectedX], 0.5, imgui.get_color_u32_rgba(*imgui.get_style().colors[imgui.COLOR_TEXT]))

            #     ignore = 0
            #     for i in range(1,len(self.oneVertexTrace)):
            #         draw_list.add_text(posCircleX+i*sizeX/102-3,posCircleY, imgui.get_color_u32_rgba(*(imgui.get_style().colors[imgui.COLOR_TEXT][:3]),0.3),str("'"))
            #         if self.ignoreBeforeOneVertexTrace <= ignore:
                        
            #             # print("No: ",self.ignoreBeforeOneVertexTrace )

            #             # draw_list.add_circle(posCircleX+(i-1)*sizeX/102, posCircleY-distance_in_y*self.oneVertexTrace[i-1], size, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),100, thicknes)
            #             # draw_list.add_circle(posCircleX+i*sizeX/102, posCircleY-distance_in_y*self.oneVertexTrace[i], size, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),100, thicknes)
            #             # print(imgui.get_style().colors[imgui.COLOR_TEXT])
            #             draw_list.add_line(posCircleX+(i-1)*sizeX/102, posCircleY-distance_in_y*self.oneVertexTrace[i-1],
            #                             posCircleX+(i)*sizeX/102, posCircleY-distance_in_y*self.oneVertexTrace[i],
            #                                 imgui.get_color_u32_rgba(*imgui.get_style().colors[imgui.COLOR_TEXT]), thicknes)
                        
            #         else:
            #             ignore += 1
