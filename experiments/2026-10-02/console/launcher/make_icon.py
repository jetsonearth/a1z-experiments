#!/usr/bin/env python3
"""A1Z Console app icon: dot-matrix "A1Z" on an OLED-black macOS squircle, one red status dot (Nothing style)."""
import sys
from PIL import Image,ImageDraw
S=2048;k=S/1024  # draw at 2x, downsample for anti-aliasing
G={'A':['.###.','#...#','#...#','#####','#...#','#...#','#...#'],
   '1':['..#..','.##..','..#..','..#..','..#..','..#..','.###.'],
   'Z':['#####','....#','...#.','..#..','.#...','#....','#####']}
img=Image.new('RGBA',(S,S),(0,0,0,0));d=ImageDraw.Draw(img)
x0,y0,w=100*k,100*k,824*k  # macOS icon grid: 824 px body in a 1024 canvas
d.rounded_rectangle([x0,y0,x0+w,y0+w],radius=185*k,fill=(10,10,10,255))
d.rounded_rectangle([x0+3*k,y0+3*k,x0+w-3*k,y0+w-3*k],radius=182*k,outline=(38,38,38,255),width=int(3*k))
n=19;p=38*k;g0=x0+(w-n*p)/2+p/2
lit=set()
for gi,ch in enumerate('A1Z'):
    for r,row in enumerate(G[ch]):
        for c,v in enumerate(row):
            if v=='#':lit.add((1+gi*6+c,6+r))
red=(15,15)
def inside(cx,cy,m=60*k):  # keep the dim grid away from the rounded corners
    rr=185*k;lx=min(max(cx,x0+rr),x0+w-rr);ly=min(max(cy,y0+rr),y0+w-rr)
    return (cx-lx)**2+(cy-ly)**2<=(rr-m)**2
for i in range(n):
    for j in range(n):
        cx,cy=g0+i*p,g0+j*p
        if (i,j)==red:r,col=0.42*p,(215,25,33,255)
        elif (i,j) in lit:r,col=0.40*p,(255,255,255,255)
        elif inside(cx,cy):r,col=0.11*p,(46,46,46,255)
        else:continue
        d.ellipse([cx-r,cy-r,cx+r,cy+r],fill=col)
img.resize((1024,1024),Image.LANCZOS).save(sys.argv[1] if len(sys.argv)>1 else 'icon.png')
