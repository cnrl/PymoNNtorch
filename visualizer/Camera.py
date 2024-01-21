import glm
import glfw
from OpenGL.GL import *
from imgui_bundle import imgui,implot


class Camera:
    def __init__(self,upper):
        self.upper = upper
        self.model = glm.mat4(1.0)
        self.model = glm.translate(self.model, glm.vec3(0.0, 0.0, 0.0))
        self.Position  = glm.vec3(0.0, 0.0, 1.5)
        self.Orientation = glm.vec3(0.0, 0.0, -1.0)
        self.Up = glm.vec3(0.0, 1.0, 0.0)
        self.speed = 0.01
        self.Zoom = 95.0
    def OnUpdateCamera(self):
        # projection = glm.ortho(-1,1, -1,1, -1000.0, 1000.0)
        # view = glm.lookAt(Position, Position + Orientation, Up) 
        self.projection = glm.perspective(glm.radians(self.Zoom), self.width/self.height, 0.1, 100000.0)

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

        glUniformMatrix4fv (self.upper.uniform_location_projection, 1, GL_FALSE, glm.value_ptr(self.projection))
        glUniformMatrix4fv (self.upper.uniform_location_view, 1, GL_FALSE, glm.value_ptr(self.view))
        glUniformMatrix4fv (self.upper.uniform_location_model, 1, GL_FALSE, glm.value_ptr(self.model))

        
    def cameraInput(self):
        if (imgui.is_key_down(imgui.Key.w) ):
            self.Position += self.speed * self.Orientation
        if (imgui.is_key_down(imgui.Key.a)):
            self.Position += self.speed * -glm.normalize(glm.cross(self.Orientation, self.Up))
        if (imgui.is_key_down(imgui.Key.s)):
            self.Position += self.speed * -self.Orientation
        if (imgui.is_key_down(imgui.Key.d)):
            self.Position += self.speed * glm.normalize(glm.cross(self.Orientation, self.Up))
        if (imgui.is_key_down(imgui.Key.e)):
            self.Position += self.speed * self.Up
        if (imgui.is_key_down(imgui.Key.q)):
            self.Position += self.speed * -self.Up

        if (imgui.is_key_down(imgui.Key.left_ctrl) or (imgui.is_key_down(imgui.Key.right_ctrl))):
            self.speed = 0.0005*8
        elif (imgui.is_key_down(imgui.Key.left_shift) or (imgui.is_key_down(imgui.Key.right_shift))):
            self.speed = 0.0005*4
        else:
            self.speed = 0.0005

        if (imgui.is_key_down(imgui.Key.p)):
            self.model = glm.scale(self.model,glm.vec3(glm.vec2(1.00000000001),1.0))
            self.Zoom -= 0.1
        if (imgui.is_key_down(imgui.Key.o)):
            self.model = glm.scale(self.model,glm.vec3(glm.vec2(0.999900009999),1.0))
            self.Zoom += 0.1

        if (imgui.is_key_down(imgui.Key.up_arrow)):
            self.model = glm.rotate(self.model,-self.speed,glm.normalize(glm.cross(self.Orientation, self.Up)))
            # self.Orientation=glm.rotate(self.Up,-self.speed,glm.normalize(glm.cross(self.Orientation, self.Up)))
            # self.Orientation.x+= self.speed
        if (imgui.is_key_down(imgui.Key.down_arrow)):
            self.model = glm.rotate(self.model,self.speed,glm.normalize(glm.cross(self.Orientation, self.Up)))
        if (imgui.is_key_down(imgui.Key.right_arrow)):
            self.model = glm.rotate(self.model,self.speed,glm.normalize(self.Up))
        if (imgui.is_key_down(imgui.Key.left_arrow)):
            self.model = glm.rotate(self.model,-self.speed,glm.normalize(self.Up)) 

        if ((imgui.is_key_down(imgui.Key.left_ctrl) or imgui.is_key_down(imgui.Key.right_ctrl)) and 
                                                       imgui.is_key_down(imgui.Key.r)):
            self.model = glm.mat4(1.0)
            self.Position  = glm.vec3(0.0, 0.0, 1.5)
            self.Orientation = glm.vec3(0.0, 0.0, -1.0)
            self.Up = glm.vec3(0.0, 1.0, 0.0)
            self.speed = 0.005
            self.Zoom = 95.0
        # self.OnUpdateCamera(width,height)