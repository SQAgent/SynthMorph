# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:53:04 2025

@author: Administrator
"""
# Abaqus script, used for performing finite element modeling and calculation
# Note: Must be run in Abaqus CAE noGUI mode



import math as m
import os
import shutil

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from mesh import *
executeOnCaeStartup()
Mdb()
NumCpu = 64
NumGpu = 0


# Subroutine = '/home/shangqing/sqdata/model/wmh/uhyper_gent.for'
# output_path = 'D:/abaqus/Temp/'
# path = 'D:/abaqus/Temp/Dataset_Picture/'
# file_path=path+'test1.txt'

import sys
print(len(sys.argv))
if len(sys.argv) != 10:
    print("useage: abq2022 cae noGUI=LLM_CAE_FE_2.py -- <output_path> <file_path>")
    sys.exit(1)
output_path = sys.argv[-2]
file_path = sys.argv[-1]
Subroutine = '/home/shangqing/sqdata/model/wmh/uhyper_gent.for'
if output_path:
    output_path = os.path.abspath(output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    os.chdir(output_path)


def normalize_subroutine(path):
    if not path:
        return None
    root, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in ('.f', '.o'):
        return path
    if ext == '.for':
        alt = root + '.f'
        if os.path.isfile(path) and not os.path.exists(alt):
            shutil.copy(path, alt)
        if os.path.exists(alt):
            return alt
    return None


Subroutine = normalize_subroutine(Subroutine)


def read_txt(file_path, cols):

    col_x, col_y = cols[0], cols[1]
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    
    x_values = list(map(float, lines[col_x].strip().split('\t')))
    y_values = list(map(float, lines[col_y].strip().split('\t')))
    
    
    coordinates = []
    for i, x_val, y_val in zip(range(len(x_values)), x_values, y_values):
        if i%10==0:
            coordinates.append([x_val, y_val])
    
    
    coordinates.append([x_values[0], y_values[0]])
    
    return coordinates

with open(file_path, 'r') as file:
    lines = file.readlines()
    line_count = len(lines)




M = mdb.models['Model-1']
s = M.ConstrainedSketch(name='__profile__', 
                        sheetSize=200.0)
s.setPrimaryObject(option=STANDALONE)
for i in range(int(line_count/2.0)):
    col=[2*i,2*i+1]
    line = read_txt(file_path,col)
    s.Spline(points=line)
    # print(i)

p = M.Part(name='Part-1', 
            dimensionality=THREE_D, 
            type=DEFORMABLE_BODY)
p.BaseSolidExtrude(sketch=s, 
                    depth=4.2)
s.unsetPrimaryObject()
del M.sketches['__profile__']



s = M.ConstrainedSketch(name='__profile__', 
                        sheetSize=200.0)
s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=(10.0/30, 10.0/30), 
            point2=(30.0+(10.0/30), 30.0+(10.0/30)))
p = M.Part(name='Part-2', 
            dimensionality=THREE_D, 
            type=DEFORMABLE_BODY)
p.BaseSolidExtrude(sketch=s, 
                    depth=4.2)
s.unsetPrimaryObject()
del M.sketches['__profile__']




#assamble
a = mdb.models['Model-1'].rootAssembly

p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=ON)
p = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-1', part=p, dependent=ON)


a.InstanceFromBooleanCut(name='Part-3', 
                          instanceToBeCut=a.instances['Part-2-1'], 
                          cuttingInstances=(a.instances['Part-1-1'], ), 
                          originalInstances=SUPPRESS)


# =============================================================================
# 
# =============================================================================
a = mdb.models['Model-1'].rootAssembly
a.translate(instanceList=('Part-1-1', 'Part-2-1', 'Part-3-1'), 
              vector=(-10.0/30-30.0/2, -10.0/30-30.0/2, 30.0/(2*m.tan(m.pi/12))-4.2))


a.RadialInstancePattern(instanceList=('Part-3-1', ), 
                        point=(0.0, 0.0, 0.0), 
                        axis=(0.0, 1.0, 0.0), 
                        number=12, 
                        totalAngle=360.0)

a.InstanceFromBooleanMerge(name='layer_1', instances=(
    a.instances['Part-3-1'], a.instances['Part-3-1-rad-2'], 
    a.instances['Part-3-1-rad-3'], a.instances['Part-3-1-rad-4'], 
    a.instances['Part-3-1-rad-5'], a.instances['Part-3-1-rad-6'], 
    a.instances['Part-3-1-rad-7'], a.instances['Part-3-1-rad-8'], 
    a.instances['Part-3-1-rad-9'], a.instances['Part-3-1-rad-10'], 
    a.instances['Part-3-1-rad-11'], a.instances['Part-3-1-rad-12'], ), 
    originalInstances=SUPPRESS, domain=GEOMETRY)



a.LinearInstancePattern(instanceList=('layer_1-1', ), 
                        direction1=(1.0, 0.0, 0.0), #invalid
                        direction2=(0.0, 1.0, 0.0), 
                        number1=1, #invalid
                        number2=6, 
                        spacing1=130.624, #invalid
                        spacing2=30.0)


a.InstanceFromBooleanMerge(name='Layer_All', 
                            instances=(a.instances['layer_1-1'], 
                                      a.instances['layer_1-1-lin-1-2'], 
                                      a.instances['layer_1-1-lin-1-3'], 
                                      a.instances['layer_1-1-lin-1-4'], 
                                      a.instances['layer_1-1-lin-1-5'], 
                                      a.instances['layer_1-1-lin-1-6'], ), 
                            originalInstances=SUPPRESS, 
                            domain=GEOMETRY)
session.viewports['Viewport: 1'].view.setProjection(projection=PERSPECTIVE)

p = mdb.models['Model-1'].parts['Layer_All']


session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.graphicsOptions.setValues(backgroundStyle=SOLID, 
    backgroundColor='#FFFFFF')
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 1'].setValues(origin=(215.217697143555, 0.0), 
    width=294.951019287109)
session.viewports['Viewport: 1'].setValues(width=160.0)
session.viewports['Viewport: 1'].setValues(height=220.0)


session.printOptions.setValues(vpDecorations=OFF)

filename = os.path.join(output_path, 'model')
p = mdb.models['Model-1'].parts['Layer_All']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].restore()
session.pngOptions.setValues(imageSize=(1200, 1200))
session.printToFile(fileName=filename, format=PNG, canvasObjects=(
    session.viewports['Viewport: 1'], ))