#!/usr/bin/env python
# coding=utf-8
from lxml import etree
import numpy as np


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3

gml_ns = "http://www.opengis.net/gml"
min_x = min_y = min_z = float('inf')
max_x = max_y = max_z = -float('inf')
center_x =  980567.517053
center_y =  198976.869809
spread_x = 1000
spread_y = 1000
max_surfs = 20000
building_surfs = []

def visualize_building():
    global  building_surfs
    if not building_surfs:
        return
    # fig = plt.figure()
    # ax = mpl3.Axes3D(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    v_min_x = v_min_y = v_min_z = float('inf')
    v_max_x = v_max_y = v_max_z = -float('inf')
    for vert in building_surfs:
        ax.add_collection3d(mpl3.art3d.Line3DCollection(vert, colors='k', linewidths=1.0))
        ax.add_collection3d(mpl3.art3d.Poly3DCollection(vert, facecolors='b'))
        for v in vert:
            for x,y,z in v:
                v_min_x = min(v_min_x, x)
                v_max_x = max(v_max_x, x)
                v_min_y = min(v_min_y, y)
                v_max_y = max(v_max_y, y)
                v_min_z = min(v_min_z, z)
                v_max_z = max(v_max_z, z)

    print "DEBUG: min_mans:",v_max_x, v_min_x, v_min_y, v_max_y,v_min_z, v_max_z
    ax.set_xlim3d(left=v_min_x, right= v_max_x )
    ax.set_ylim3d(bottom=v_min_y, top=v_max_y)
    ax.set_zlim3d(bottom=v_min_z, top=v_max_z)
    plt.show()
    plt.close()

def addToSurfaceList(posList):
        global  max_x, max_y, max_z, min_x, min_y, min_z, building_surfs, center_x, center_y, spread_x, spread_y
        X, Y, Z = [], [], []
        c = 0
        addToList = True
        for value in posList:
            value = float(value)
            if c % 3 == 0:
                max_x = max(value, max_x)
                min_x = min(value, min_x)
                X.append(value)
                if not( (center_x-spread_x) <= value<= (center_x+spread_x)):
                    addToList = False
            elif c % 3 == 1:
                max_y = max(value, max_y)
                min_y = min(value, min_y)
                Y.append(value)
                if not ((center_y-spread_y) <= value<= (center_y+spread_y)):
                    addToList = False

            else:
                max_z = max(value, max_z)
                min_z = min(value, min_z)
                Z.append(value)
            c += 1
            if c > len(posList) - 3:
                break
        if (len(building_surfs) < max_surfs) and addToList:
            building_surfs.append( [list(zip(X, Y, Z))] )
        return addToList



def clear_element(elem):
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]



def iterative_bldg_parsing(infile, ns, showBuilding = False ):
    context = etree.iterparse(infile, events=('start','end'))
    global gml_ns
    t_building = 0
    t_wall = t_roof = t_ground = 0
    for event,elem in context:
        if event=='end' and elem.tag == "{%s}Building" % ns :
            t_building+=1
            for wall in elem.findall('.//{%s}WallSurface' % ns):
                #if build_min_indx<= t_building<= build_max_indx:
                for posList in wall.findall(".//{%s}posList" % gml_ns):
                    _ = addToSurfaceList( posList.text.split() )
                t_wall += 1
            # for ground in elem.findall('.//{%s}GroundSurface' % ns):
            #     if build_min_indx<= t_building<= build_max_indx:
            #         for posList in ground.findall(".//{%s}posList" % gml_ns):
            #             building_surfs.append( parse_vertex_list( posList.text.split() ) )
            #     t_ground += 1
            for roof in elem.findall('.//{%s}RoofSurface' % ns):
                #if build_min_indx<= t_building<= build_max_indx:
                for posList in roof.findall(".//{%s}posList" % gml_ns):
                    _ = addToSurfaceList( posList.text.split() )
                t_roof += 1
            clear_element(elem)
    print "Building# ", t_building
    print "Wall# ", t_wall
    #print "Ground# ", t_ground
    print "Roof# ", t_roof
    print "x-range: ",min_x, max_x, max_x - min_x
    print "y-range: ",min_y, max_y, max_y - min_y
    print "z-range: ",min_z, max_z, max_z - min_z
    if showBuilding:
        visualize_building( )
    # for b in  building_surfs:
    #     for k in b:
    #         print k





# def countBuilding(infile):
#     from xml.etree import ElementTree as ET
#
#     tree = ET.parse(infile)
#     root = tree.getroot()
#
#     namespaces = {
#         'ns0': "http://www.opengis.net/citygml/2.0",
#         'ns1': "http://www.opengis.net/gml",
#         'ns2': "http://www.opengis.net/citygml/building/2.0"
#     }
#
#     bldg_count = 0
#     for _ in root.findall('.//ns2:Building', namespaces):
#         bldg_count+=1
#     print bldg_count

# x-range:  1044807.62986 1044864.79805 57.1681927501
# y-range:  183953.991394 184001.039856 47.048462331
# z-range:  15.2975 45.7 30.4025


if __name__ == "__main__":
    ns = "http://www.opengis.net/citygml/building/1.0"
    infile = "./data/DA_WISE_GMLs/DA12_3D_Buildings_Merged.gml"
    iterative_bldg_parsing(infile , ns, showBuilding=True)
    #countBuilding(infile)

