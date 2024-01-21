#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float uZ;
uniform float uSize;
uniform float uSizeData;
uniform float uLocx;
uniform float uLocy;
uniform float uLocMacanx;
uniform float uLocMacany;
uniform float uPlan;

uniform float uisNewWindow;


uniform float uIsdata;

out vec2 TexCoord;
out float Isdata_in_all;
out float Isdata;
out float Isplane;
void main()
{
    TexCoord = aTexCoord;
    Isplane = uPlan;
    if (uIsdata == 1.0){
        gl_Position =  vec4(aPos.x, aPos.y, 0.0, 1.0);
        //gl_PointSize = uSizeData;
        gl_PointSize = uSizeData;
        TexCoord = vec2(uLocx,uLocy);
        Isdata = 1.0;

    }
    else{
        vec4 result = projection * view * model * vec4(aPos.x, aPos.y, uZ, 1.0);
        if (uisNewWindow >= 0.5){
            result = vec4(aPos.x, aPos.y, 0.0, 1.0);
            gl_Position = result;
            gl_PointSize = uSize;
        }
        else{
            gl_Position = result;
            gl_PointSize = uSize/result.z;

        }

        if (aPos.x==uLocMacanx && aPos.y==uLocMacany){
            Isdata_in_all = 1.0;
            gl_PointSize = 2*uSize/result.z;
        }
        else{
            Isdata_in_all = 0.0;
        
        }



        //gl_PointSize = uSizeData;
        Isdata = 0.0;
    }
}