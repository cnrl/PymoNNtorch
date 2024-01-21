#version 330 core

out vec4 FragColor;

in vec2 TexCoord;
in float Isdata;
in float Isdata_in_all;
in float Isplane;

uniform sampler2D ourTexture;

void main()
{
    if (Isplane>=0.5){
        FragColor = vec4(1.0f, 1.0f, 1.0f, 0.4f);

    }
    else{
        if (Isdata >= 0.5){
            if (texture(ourTexture, TexCoord).g >= 1.0){
                FragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
            }
            else{
                FragColor = vec4(texture(ourTexture, TexCoord).ggg, 1.0f);
            }
        }
        else{ 
            if (Isdata_in_all >= 0.5){
                //float dist = length(gl_PointCoord  - vec2(0.4));
                //float alpha = smoothstep(0.4, 0.7, dist);
                //if (dist > 0.6)
                //   discard;
                //if (dist < 0.5)
                //   discard;
                //FragColor = vec4(dist,dist, dist ,1.0f);
                //FragColor = vec4(alpha,alpha, alpha ,1.0f);
                //FragColor = vec4(0.0, 0.0, 0.0, 1.0);
                if (texture(ourTexture, TexCoord).g>=1.0){
                    FragColor = vec4(0.56f, 0.93f, 0.56f, 1.0f);
                    //FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
                }
                else{
                    FragColor = vec4(0.94f, 0.5f, 0.5f, 1.0f);
                }
                //FragColor = vec4(texture(ourTexture, TexCoord).ggg, 0.5f);
            }
            else{
                if (texture(ourTexture, TexCoord).g>=1.0)
                    FragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
                else
                    FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
                    //FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }
        }
    }
}