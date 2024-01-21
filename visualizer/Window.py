from OpenGL.GL import *
from .FrameBuffer import FrameBuffer
from .Camera import Camera
class Window(Camera):
    def __init__(self,width,height,upper):
        Camera.__init__(self,upper)
        self.width,self.height = width,height
        self.frameBuffer = FrameBuffer(width,height)
        self.show = True
    def OnUpdate(self):
        if not self.show:return
        if self.width == 0 or self.height == 0:return
        self.OnUpdateCamera()
        glBindFramebuffer(GL_FRAMEBUFFER,self.frameBuffer.fbo)
        glViewport(0, 0, int(self.width), int(self.height))
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.Draw()
        glBindFramebuffer(GL_FRAMEBUFFER,0)
    def Draw(self):pass
    