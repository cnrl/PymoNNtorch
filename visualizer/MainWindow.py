from .Window import Window
from OpenGL.GL import *
import torch

class MainWindow(Window):
    def __init__(self,width, height,upper):
        Window.__init__(self,width, height,upper)
    def Draw(self):
        for n in range(len(self.upper.network.NeuronGroups)):
            if not self.upper.shows[n]:continue
            X=((torch.reshape(self.upper.tensors[n], (1,self.upper.tensorHeights[n]*self.upper.tensorWidths[n]))).squeeze(0)).reshape(self.upper.tensorHeights[n]*self.upper.tensorWidths[n],1)
            if self.upper.network.device == 'cpu':
                tens2 = torch.zeros([self.upper.tensorHeights[n]*self.upper.tensorWidths[n],1], dtype=torch.float)
                X2=torch.cat((X,tens2),1)
                X3=torch.cat((tens2,X2),1)
                tens3 = torch.ones([self.upper.tensorHeights[n]*self.upper.tensorWidths[n],1], dtype=torch.float)
                X4=torch.cat((X3,tens3),1)
                tensor2=X4
                glBindTexture(GL_TEXTURE_2D, self.upper.colors[n])
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.upper.tensorWidths[n], self.upper.tensorHeights[n], 0, GL_RGBA, GL_FLOAT, tensor2.numpy())
            else:
                from cuda import cudart as cu
                tens2 = torch.zeros([self.upper.tensorHeights[n]*self.upper.tensorWidths[n],1], dtype=torch.float, device=torch.device('cuda:0'))
                X2=torch.cat((X,tens2),1)
                X3=torch.cat((tens2,X2),1)
                tens3 = torch.ones([self.upper.tensorHeights[n]*self.upper.tensorWidths[n],1], dtype=torch.float, device=torch.device('cuda:0'))
                X4=torch.cat((X3,tens3),1)
                tensor2=X4
                (err,) = cu.cudaGraphicsMapResources(1, self.upper.cuda_images[n], cu.cudaStreamLegacy)
                err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.upper.cuda_images[n], 0, 0)
                (err,) = cu.cudaMemcpy2DToArrayAsync(
                    array,
                    0,
                    0,
                    tensor2.data_ptr(),
                    4*4*self.upper.tensorWidths[n],
                    4*4*self.upper.tensorWidths[n],
                    self.upper.tensorHeights[n],
                    cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    cu.cudaStreamLegacy,
                )
                (err,) = cu.cudaGraphicsUnmapResources(1, self.upper.cuda_images[n], cu.cudaStreamLegacy)
            glBindVertexArray(self.upper.vaos[n])
            glBindBuffer(GL_ARRAY_BUFFER, self.upper.vbos[n])
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0+3*4))
            glBindTexture(GL_TEXTURE_2D, self.upper.colors[n])
            glUniform1f(self.upper.uniform_location_z, -1/2*n) 
            # glBegin(GL_LINE_STRIP) 
            if self.upper.selectedGroup == n and self.upper.selectedX != -1 and self.upper.selectedY != -1:
                glUniform1f(self.upper.uniform_location_loc_mac_x, -1+(self.upper.selectedX+1)*2/(self.upper.tensorWidths[n]+1))
                glUniform1f(self.upper.uniform_location_loc_mac_y, -1+(self.upper.selectedY+1)*2/(self.upper.tensorHeights[n]+1)) 
            else:
                glUniform1f(self.upper.uniform_location_loc_mac_x, 0)
                glUniform1f(self.upper.uniform_location_loc_mac_y, 0) 
            ## for big number like million if points will be like square specify more details 
            ## last vertex show value of seleceted index and show that in circle shape
            # if self.set_enable_smooth:
            #     glEnable(GL_POINT_SMOOTH)
            # else:
            #     glDisable(GL_POINT_SMOOTH)
            glUniform1f(self.upper.uniform_location_isdata, 0.0)
            glDrawArrays(GL_POINTS, 0, self.upper.tensorHeights[n]*self.upper.tensorWidths[n])


            #!
            self.addQuad(-1/2*n)
        
    def addQuad(self,z_value):
        glUniform1f(self.upper.uniform_location_z, z_value)
        glUniform1f(self.upper.uniform_location_isdata, 0.0)
        # glUseProgram(0)
        glUniform1f(self.upper.uniform_location_isplan,1.0)
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
        glUniform1f(self.upper.uniform_location_isplan,0.0)
        # glDisable(GL_BLEND)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)