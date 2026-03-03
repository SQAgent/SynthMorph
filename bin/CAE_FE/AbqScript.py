# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:53:04 2025

@author: Administrator
"""
import os
import math as m
oneapi = "/home/shangqing/intel/oneapi"
os.environ["ONEAPI_ROOT"] = oneapi
os.environ["PATH"] = os.path.join(oneapi, "compiler/latest/linux/bin/intel64") + ":" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = os.path.join(oneapi, "compiler/latest/linux/compiler/lib/intel64_lin") + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from mesh import *
executeOnCaeStartup()
Mdb()
NumCpu = 24
NumGpu = 0


Subroutine = '/home/shangqing/sqdata/model/wmh/uhyper_gent.f'
# file_path = "/home/shangqing/sqdata/project/sqagents/bin/CAE_FE/debug/test.txt"

# 默认在 test.txt 同级目录下生成结果

import sys
print(len(sys.argv))
if len(sys.argv) != 9:
    print("用法: abq2022 cae noGUI=LLM_CAE_FE_2.py -- <file_path>")
    sys.exit(1)
file_path = sys.argv[-1]
file_path = os.path.abspath(file_path)
output_path = os.path.dirname(file_path)

print("file_path:", file_path)
print("output_path:", output_path)


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


p = mdb.models['Model-1'].parts['Layer_All']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].restore()
session.pngOptions.setValues(imageSize=(1200, 1200))

model_path = os.path.join(output_path, 'model')
session.printToFile(fileName=model_path, format=PNG, canvasObjects=(
    session.viewports['Viewport: 1'], ))

# =============================================================================
# material
# =============================================================================

ShearMod = 0.195000
Jm = 12.000000
M.Material(name='Rubber')
#: Gent Model----------------------------------------------------------------------
M.materials['Rubber'].Depvar(n=2)

M.materials['Rubber'].Hyperelastic(materialType=ISOTROPIC, 
testData=OFF, type=USER, properties=2, table=((ShearMod, Jm), ))

M.materials['Rubber'].Density(table=((1.1e-09, ), ))
#: Gent Model----------------------------------------------------------------------
M.HomogeneousSolidSection(name='Section-1', 
                          material='Rubber', 
                          thickness=None)

p = M.parts['Layer_All']
region=p.Set(cells=p.cells, name='Set-Body')
p.SectionAssignment(region, 
                    sectionName='Section-1', 
                    offset=0.0, 
                    offsetType=MIDDLE_SURFACE, 
                    offsetField='', 
                    thicknessAssignment=FROM_SECTION)


f = p.faces
faces = f.getByBoundingBox(-60,165,-60,60,165,60)
p.Set(faces=faces, name='Set-top')
faces = f.getByBoundingBox(-60,-15,-60,60,-15,60)
p.Set(faces=faces, name='Set-bottom')


# =============================================================================
# Step
# =============================================================================
M.StaticStep(name='Step-1', 
              previous='Initial', 
              timeIncrementationMethod=FIXED, 
              initialInc=0.1, 
              noStop=OFF, 
              nlgeom=ON)
# M.StaticStep(name='Step-2', 
#               previous='Step-1', 
#               timeIncrementationMethod=FIXED, 
#               initialInc=0.1, 
#               noStop=OFF)

M.fieldOutputRequests['F-Output-1'].setValues\
    (variables=('S', 'PE', 'PEEQ', 'PEMAG', 'LE', 
                'U', 'RF', 'CF', 'CSTRESS', 'CDISP', 
                'COORD'))
    
# =============================================================================
# Loading
# =============================================================================

region = a.instances['Layer_All-1'].sets['Set-bottom']
M.EncastreBC(name='BC-1', 
              createStepName='Step-1', 
              region=region, 
              localCsys=None)


region = a.instances['Layer_All-1'].sets['Set-top']
M.DisplacementBC(name='BC-2', 
                  createStepName='Step-1', 
                  region=region, 
                  u1=UNSET, u2=20.0, u3=UNSET, 
                  ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                  amplitude=UNSET, 
                  fixed=OFF, 
                  distributionType=UNIFORM, 
                  fieldName='', 
                  localCsys=None)
# M.boundaryConditions['BC-2'].deactivate('Step-2')




# region = a.instances['Layer_All-1'].sets['Set-top']
# M.DisplacementBC(name='BC-3', 
#                   createStepName='Step-2', 
#                   region=region, 
#                   u1=UNSET, u2=-20.0, u3=UNSET, 
#                   ur1=UNSET, ur2=UNSET, ur3=UNSET, 
#                   amplitude=UNSET, 
#                   fixed=OFF, 
#                   distributionType=UNIFORM, 
#                   fieldName='', 
#                   localCsys=None)





# =============================================================================
# Mesh
# =============================================================================
p = M.parts['Layer_All']
p.seedPart(size=3.0, 
            deviationFactor=0.3, 
            minSizeFactor=0.1)

c = p.cells
pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE)

p.setElementType(
regions=(p.cells,), 
elemTypes=(ElemType(elemCode=C3D8H, elemLibrary=STANDARD), 
ElemType(elemCode=C3D6H, elemLibrary=STANDARD),
ElemType(elemCode=C3D4H, elemLibrary=STANDARD)))

p.generateMesh()

# # # =============================================================================
# # # Set
# # # =============================================================================
# # p = mdb.models['Model-1'].parts['Layer_All']
# # v = p.vertices
# # verts = v.findAt(((55.980762, 79.945192, 15.0), ))
# # p.Set(vertices=verts, name='Set-d1')

# # p = mdb.models['Model-1'].parts['Layer_All']
# # v = p.vertices
# # verts = v.findAt(((-55.980762, 79.945192, 15.0), ))
# # p.Set(vertices=verts, name='Set-d2')



# p = mdb.models['Model-1'].parts['Layer_All']
# v = p.vertices
# verts = v.findAt(((33.63318,165.00,45.222891), ))
# p.Set(vertices=verts, name='Set-e1')


Job_name='Job-1'
mdb.Job(name=Job_name, model='Model-1', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine=Subroutine, 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=NumCpu, 
    numDomains=NumCpu, numGPUs=NumGpu)
mdb.jobs[Job_name].submit()
mdb.jobs[Job_name].waitForCompletion()




o3 = session.openOdb(name=os.path.join(output_path, Job_name+'.odb'))
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()

session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
    visibleEdges=NONE)


session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )

session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(
    contourStyle=CONTINUOUS, maxAutoCompute=OFF, maxValue=31.0, minValue=0)
session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(
    spectrum='White to black')

session.graphicsOptions.setValues(backgroundStyle=SOLID, 
    backgroundColor='#FFFFFF')
session.viewports['Viewport: 1'].restore()
session.viewports['Viewport: 1'].setValues(width=160.0)
session.viewports['Viewport: 1'].setValues(height=220.0)

for i in range(0,10,1):
    session.viewports[session.currentViewportName].odbDisplay.setFrame(
        step='Step-1', frame=i)
    session.printOptions.setValues(vpDecorations=OFF)
    session.pngOptions.setValues(imageSize=(1200, 1200))
    session.printToFile(fileName='frame_%s'%i, format=PNG, canvasObjects=(
        session.viewports['Viewport: 1'], ))


print('Save picture successfully')









# # # Post processing
# # # Angle 
# # import odbAccess
# # import math as m
# # odb = odbAccess.openOdb(path='D:/abaqus/Temp/Job-2.odb',
# #                     readOnly = True)
# # root = odb.rootAssembly

# # for i in range(11):
# #     lastFrame=odb.steps.values()[-1].frames[i]
    
# #     coord=lastFrame.fieldOutputs['COORD']
    
# #     nodeset=root.instances['LAYER_ALL-1'].nodeSets['SET-D1']
# #     d1 = coord.getSubset(region=nodeset).values[0].data
# #     x1, y1, z1 = d1[0], d1[1], d1[2]
    
# #     nodeset=root.instances['LAYER_ALL-1'].nodeSets['SET-D2']
# #     d2 = coord.getSubset(region=nodeset).values[0].data
# #     x2, y2, z2 = d2[0], d2[1], d2[2]
    
# #     # print(x1,y1,z1,'\t',x2,y2,z2)
# #     D = m.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
# #     # print(D)
# #     epsilon_y = (D-2*55.980762)/(2*55.980762)
# #     epsilon_x = (0.5*(y1+y2)-79.945192)/90
    
# #     possion = -epsilon_y/epsilon_x
# #     print(possion)


# import odbAccess
# import math as m
# odb = odbAccess.openOdb(path='D:/abaqus/Temp/Job-2.odb',
#                     readOnly = True)
# root = odb.rootAssembly

# for i in range(11):
#     lastFrame=odb.steps.values()[-1].frames[i]
    
#     coord=lastFrame.fieldOutputs['COORD']
    
#     nodeset=root.instances['LAYER_ALL-1'].nodeSets['SET-E1']
#     d1 = coord.getSubset(region=nodeset).values[0].data
#     x1, y1, z1 = d1[0], d1[1], d1[2]
    
#     theta = (m.atan(z1/x1)-m.atan(45.222891/33.63318))*180/m.pi
#     print (theta)

