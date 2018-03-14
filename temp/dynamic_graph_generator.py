'''
author: Md Shaifur Rahman (mdsrahman@cs.stonybrook.edu)
        Stony Brook University, NY.
'''
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3
import sys

class DynamicGraphGenerator(object):
    '''
    this class load the saved 3D-building file, calculate LOS and save the resulting graph
    '''
    def __init__(self):
        self.total_building = 0
        self.total_surface = 0

        self.ref_x = None
        self.ref_y = None

        self.max_x = -float('inf')
        self.max_y = -float('inf')

        self.min_z = None
        self.max_z = None

        self.bldg_bounding_box = None
        self.bldgs = []
        self.bldg_bbox_nsurf = []
        self.bldg_nsurf = []
        self.fso_tx = None
        self.MIN_X, self.MAX_X,\
        self.MIN_Y, self.MAX_Y,\
        self.MIN_Z, self.MAX_Z = range(6)

        return

    def loadBldgFile(self, input_file):
        self.bldg_bounding_box = np.zeros(shape = ( self.total_building, 6), dtype = np.float)
        with open(input_file, 'r') as bldg_file:
            for bindx in range(self.total_building):
                _, surf_count = bldg_file.readline().split(',')
                #--unpack bounding box----#
                x_bounds, y_bounds, z_bounds = bldg_file.readline().split(';')
                self.bldg_bounding_box[ bindx, :] = np.float(x_bounds.split(',')[0]) - self.ref_x, \
                                                    np.float(x_bounds.split(',')[1]) - self.ref_x, \
                                                    np.float(y_bounds.split(',')[0]) - self.ref_y, \
                                                    np.float(y_bounds.split(',')[1])-  self.ref_y, \
                                                    np.float(z_bounds.split(',')[0]), \
                                                    np.float(z_bounds.split(',')[1])
                #---unpack surface vertices-----#
                cur_bldg_surfs = []
                for sindx in range(int(surf_count)):
                    cur_surf_txt = bldg_file.readline().split(';')
                    cur_surf = []
                    for cur_vertices in cur_surf_txt:
                        x, y, z = cur_vertices.split(',')
                        x, y, z = np.float(x) - self.ref_x, np.float(y) - self.ref_y, np.float(z)
                        cur_surf.append((x, y, z))
                    cur_bldg_surfs.append(np.array( cur_surf, dtype = np.float) )
                self.bldgs.append(cur_bldg_surfs)
                self.bldg_bbox_nsurf.append(None)
                self.bldg_nsurf.append(None)

        return

    def loadStatFile(self, input_file):
        with open(input_file, 'r') as stat_file:
            line_txt = stat_file.readline().split(',')
            self.total_building = int(line_txt[1])
            self.total_surface = int(line_txt[2])
            x_lims, y_lims, z_lims = stat_file.readline().split(';')
            self.ref_x, max_gml_x  = float(x_lims.split(',')[0]), float(x_lims.split(',')[1])
            self.ref_y, max_gml_y  = float(y_lims.split(',')[0]), float(y_lims.split(',')[1])
            self.min_z, self.max_z = float(z_lims.split(',')[0]), float(z_lims.split(',')[1])
            self.max_x = max_gml_x - self.ref_x
            self.max_y = max_gml_y - self.ref_y
        return

    def load3DBuildingData(self, input_filepath):
        self.loadStatFile(input_filepath+'.stat')
        self.loadBldgFile(input_filepath+'.bldg')
        return

    def addFSOTowers(self, tower_height_ft):
        self.fso_tx = np.zeros( shape=(self.total_building, 3) , dtype = np.float ) #TODO: instead of pre-allocating, find fso-loc dynamically
        for bindx in range(self.total_building):
            bmin_x, bmax_x, bmin_y, bmax_y, bmin_z, bmax_z = self.bldg_bounding_box[bindx,: ]
            fso_x = (bmin_x + bmax_x)/2.0
            fso_y = (bmin_y + bmax_y)/2.0
            fso_z = bmax_z+tower_height_ft
            self.fso_tx[bindx, : ] = [fso_x, fso_y, fso_z]
            self.fso_los = []
        return

    def visualize3Dbuilding(self, showFSOLinks=False, deBugShowAllLinks=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for bindx, bldg_surfs in enumerate(self.bldgs):
            for surf in bldg_surfs:
                #ax.add_collection3d(mpl3.art3d.Line3DCollection([surf], colors='k', linewidths=1.0))
                ax.plot(surf[:,0], surf[:,1], surf[:, 2], linestyle ='-', color='k', linewidth=1.0)
                #--now close the polygon
                n = len(surf) - 1
                x1, y1, z1, xn, yn, zn = surf[ 0, 0], surf[ 0, 1], surf[ 0, 2], \
                                         surf[ n, 0], surf[ n, 1], surf[ n, 2]
                #ax.plot([x1, xn], [y1, yn], [z1, zn], linestyle ='-', color='k', linewidth=1.0) #TODO: Remove the comment to close the polygon

        ax.set_xlim3d(left=0-500,   right=self.max_x+500)
        ax.set_ylim3d(bottom=0-500, top=self.max_y+500)
        ax.set_zlim3d(bottom=self.min_z - 5, top=self.max_z+15)
        if showFSOLinks:
            for i,j in self.fso_los:
                ax.plot( [self.fso_tx[i, 0], self.fso_tx[j, 0] ], \
                         [self.fso_tx[i, 1], self.fso_tx[j, 1]], \
                         [self.fso_tx[i, 2], self.fso_tx[j, 2]],
                         linestyle='-', color='r', linewidth=1 )
                if deBugShowAllLinks:
                    for k in range(len(self.fso_tx)):
                        if i != k:
                            ax.plot([self.fso_tx[i, 0], self.fso_tx[k, 0]], \
                                    [self.fso_tx[i, 1], self.fso_tx[k, 1]], \
                                    [self.fso_tx[i, 2], self.fso_tx[k, 2]],
                                    linestyle=':', color='b', linewidth=0.5 )

        plt.show()
        #plt.savefig('./test.png',  dpi = 300)
        plt.close()
        return

    def visualizeBuildingBBox(self, showNormals = False, showFSOLinks = False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for bbox in self.bldg_bounding_box:
            bldg_surfs = self.getBoundingFaces(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5] )
            for surf in bldg_surfs:
                total_surf_points, _ = surf.shape
                n = total_surf_points - 1
                ax.plot(surf[:,0], surf[:, 1], surf[:, 2], linestyle = '-', color='k', linewidth = 1.0)
        if showNormals:
            for n_surfs in self.bldg_bbox_nsurf:
                surf_count, _ = n_surfs.shape
                for s_indx in range(surf_count):
                    _, _, _, nx, ny, nz = n_surfs[ s_indx, : ]
                    ax.plot([0.0, nx], [0.0, ny], [0.0, nz], linestyle=':', color='r', linewidth=1)

        if showFSOLinks:
            for i,j in self.fso_los:
                x1, y1, z1  = self.fso_tx[i, :]
                x2, y2, z2 =  self.fso_tx[j, :]
                ax.plot([x1, x2], [y1, y2], [z1, z2], linestyle='-', color='r', linewidth=1)
        ax.set_xlim3d(left=0-500,   right=self.max_x+500)
        ax.set_ylim3d(bottom=0-500, top=self.max_y+500)
        ax.set_zlim3d(bottom=self.min_z - 5, top=self.max_z+15)

        plt.show()
        #plt.savefig('./test.png',  dpi = 300)
        plt.close()
        return

    def getBoundingFaces(self, xmin, xmax, ymin, ymax, zmin, zmax ):
        '''
        1: xmin, ymin, zmin
        2: xmax, ymin, zmin
        3: xmax, ymin, zmax
        4: xmin, ymin, zmax
        5: xmin, ymax, zmin
        6: xmax, ymax, zmin
        7: xmax, ymax, zmax
        8: xmin, ymax, zmax
        face_1: 1, 2, 3, 4
        face_2: 2, 6, 7, 3
        face_3: 6, 5, 8, 7
        face_4: 5, 1, 4, 8
        face_5: 7, 8, 4, 3
        face_6: 2, 1, 5, 6
        :param bounding_bx:
        :return:
        '''
        # xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
        p1 = [xmin, ymin, zmin]
        p2 = [xmax, ymin, zmin]
        p3 = [xmax, ymin, zmax]
        p4 = [xmin, ymin, zmax]
        p5 = [xmin, ymax, zmin]
        p6 = [xmax, ymax, zmin]
        p7 = [xmax, ymax, zmax]
        p8 = [xmin, ymax, zmax]

        face_1 = np.array( [ p1, p2, p3, p4 ], dtype= np.float )
        face_2 = np.array( [ p2, p6, p7, p3 ], dtype= np.float )
        face_3 = np.array( [ p6, p5, p8, p7 ], dtype= np.float )
        face_4 = np.array( [ p5, p1, p4, p8 ], dtype= np.float )
        face_5 = np.array( [ p7, p8, p4, p3 ], dtype= np.float )
        face_6 = np.array( [ p2, p1, p5, p6 ], dtype= np.float )

        return [face_1, face_2, face_3, face_4, face_5, face_6]

    def isCollinear(self, p0, p1, p2):
        p_10 = p1 - p0
        p_20 = p2 - p0
        v = np.cross(p_10, p_20)
        if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
            return True
        return False

    def get3NonCollinearPoints(self, face):
        total_points = len(face)
        if total_points < 3:
            print "@get3NonCollinearPoints(..): #-of-points on face < 3 !"
            sys.exit(1)
        p1 = face[0]
        p2 = face[1]
        for i in range(2, total_points):
            p3 = face[i]
            if not self.isCollinear(p1, p2, p3):
                break
        return p1, p2, p3

    def getFaceNormal(self, face, interior_point = None):
        '''
        compute normal for a given face
        :return:
        '''
        #make two vectors
        p1, p2, p3 = self.get3NonCollinearPoints(face=face)
        v1 = p2 - p1
        v2 = p3-  p1
        n = np.cross(v1, v2)
        if not interior_point is None: #if an interior point is given, make sure the normal is outward
            dir_to_interior = interior_point - p1
            if np.dot(dir_to_interior, n) > 0.0:
                n = -n
        return n

    def getCentroidFromBoundingBox(self, xmin, xmax, ymin, ymax, zmin, zmax):
        cx = xmin + (xmax - xmin) / 2.0
        cy = ymin + (ymax - ymin) / 2.0
        cz = zmin + (zmax - zmin) / 2.0
        return cx, cy, cz

    def getSurfaceNormalRepresentation(self, faceList, interior_point = None):
        number_of_faces = len(faceList)
        bldg_nsurf = np.empty(shape = (number_of_faces, 6), dtype=np.float )
        bldg_nsurf[:] = np.nan
        for face_indx, face in enumerate(faceList):
            px, py, pz = face[0] #take the first point as reference
            nx, ny, nz = self.getFaceNormal(face=face, interior_point = interior_point)
            bldg_nsurf[face_indx, :] = px, py, pz, nx, ny, nz
        return bldg_nsurf

    def calculateBBoxSurfaceRepresentation(self, bindx):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bldg_bounding_box[bindx]
        bbox_faces = self.getBoundingFaces(xmin, xmax, ymin, ymax, zmin, zmax)
        cx, cy, cz = self.getCentroidFromBoundingBox(xmin, xmax, ymin, ymax, zmin, zmax)
        interior_point = [cx, cy, cz]
        bldg_nsurf = self.getSurfaceNormalRepresentation(faceList=bbox_faces, interior_point=interior_point)
        self.bldg_bbox_nsurf[bindx] = bldg_nsurf
        return

    def calculateBldgSurfaceRepresentation(self, bindx):
        # xmin, xmax, ymin, ymax, zmin, zmax = self.bldg_bounding_box[bindx]
        # cx, cy, cz = self.getCentroidFromBoundingBox(xmin, xmax, ymin, ymax, zmin, zmax)
        # interior_point = [cx, cy, cz]
        interior_point = None
        bldg_faces = self.bldgs[bindx]
        bldg_nsurf = self.getSurfaceNormalRepresentation(faceList=bldg_faces, interior_point=interior_point)
        self.bldg_nsurf[bindx] = bldg_nsurf
        return

    def getBldgSurfaceRepresentation(self, bindx):
        if self.bldg_nsurf[bindx] is None:
            self.calculateBldgSurfaceRepresentation(bindx)
        return self.bldg_nsurf[bindx]

    def getBBoxSurfaceRepresentation(self, bindx):
        if self.bldg_bbox_nsurf[bindx] is None:
            self.calculateBBoxSurfaceRepresentation(bindx)
        return self.bldg_bbox_nsurf[bindx]

    def isSurfaceIntersecting(self, p0, p1, surf_n):
        t_E = 0.0
        t_L = 1.0
        dS = p1 - p0
        sx, sy, sz, nx, ny, nz = surf_n
        sn = np.array([nx, ny, nz])
        p0_to_s = np.array([sx, sy, sz]) - p0
        n = np.dot(p0_to_s, sn)
        d = np.dot(dS, sn)
        if d==0.0: #parallel to face, assuming non-intersecting
            return False, None
        t = n / d
        if 0.0 <= t <= 1.0:
            intersecting_point = p0+t*dS
            return  True, intersecting_point
        return False, None

    def isBuildingBBoxIntersecting(self, p0, p1, bindx):
        bbox_nsurfs = self.getBBoxSurfaceRepresentation(bindx)
        return  self.isPolyhedronIntersecting(p0, p1, bbox_nsurfs)

    def isPointInsidePolygon(self, ip, surf, n_surf):
        #TODO: complete
        _, _, _, nx, ny, nz = n_surf
        n = np.array([nx, ny, nz], dtype = np.float)
        u = surf[1] - surf[0]
        total_points = len(surf)
        intersection_count = 0
        for i, p1 in enumerate(surf):
            j = (i+1)%total_points
            p2 = surf[j]
            v = p2-p1
            if self.isRayIntersectingLineSegment(ip,u,p1,v,n):
                intersection_count+=1
        if intersection_count%2==0:
            return  False
        return True

    def getBldgSurfaceAs2DPolygon(self, bindx, sindx):
        return  zip(self.bldgs[bindx][sindx][:, 0],self.bldgs[bindx][sindx][:, 1])

    def debugVisualizeLinePolygon(self, p0, p1, bindx, sindx = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surfs = []
        if sindx is not None:
            surfs.append(self.bldgs[bindx][sindx])
        else:
            surfs = self.bldgs[bindx]
        for surf in surfs:
            #ax.add_collection3d(mpl3.art3d.Line3DCollection([surf], colors='k', linewidths=1.0))
            ax.plot(surf[:,0], surf[:,1], surf[:, 2], linestyle ='-', color='k', linewidth=1.0)
            #--now close the polygon
            n = len(surf) - 1
            x1, y1, z1, xn, yn, zn = surf[ 0, 0], surf[ 0, 1], surf[ 0, 2], \
                                     surf[ n, 0], surf[ n, 1], surf[ n, 2]
            ax.plot([x1, xn], [y1, yn], [z1, zn], linestyle ='-', color='k', linewidth=1.0) #TODO: Remove the comment to close the polygon

        ax.plot([ p0[0], p1[0] ], [ p0[1], p1[1] ], [ p0[2], p1[2] ],\
                   linestyle='-', color='r', linewidth=1.0)


        ax.set_xlim3d(left=0-500,   right=self.max_x+500)
        ax.set_ylim3d(bottom=0-500, top=self.max_y+500)
        ax.set_zlim3d(bottom=self.min_z - 5, top=self.max_z+15)

        plt.show()
        #plt.savefig('./test.png',  dpi = 300)
        plt.close()
        return

    def isRayIntersectingLineSegment(self, p0, u, q0, v, n):
        perp_v = np.cross(v, n)
        d1 = np.dot(perp_v, u)
        if d1 == 0.0:  # parallel
            return False

        perp_u = np.cross(u, n)

        w = p0 - q0
        s = -np.dot(perp_v, w) / d1
        if s < 0.0:
            return False

        d2 = np.dot(perp_u, v)
        if d2 == 0.0:
            return False

        t = np.dot(perp_u, w) / np.dot(perp_u, v)
        if t > 1.0 or t < 0.0:
            return False

        return True

    def isBuildingSurfaceIntersecting(self, p0, p1, bindx):
        # print "DEBUG: building# ",bindx, "p0: ", p0, " p1: ",p1
        # self.debugVisualizeLinePolygon(p0, p1, bindx)
        # show_surfaces = raw_input('Want to show surfaces? (y/n)')
        # if show_surfaces == 'y':
        #     show_surfaces = True
        # else:
        #     show_surfaces = False
        n_surfs = self.getBldgSurfaceRepresentation( bindx)
        for sindx, n_surf in enumerate(n_surfs):
            #-------------------------debug_trap--------------#
            #1675.838184, 1815.891546, 44.3
            # debug_p1 = np.array([1675.838184, 1815.891546, 44.3], dtype=np.float)
            # if bindx==16 and sindx==43 and np.abs( np.sum(p1-debug_p1) ) < 0.001:
            #     debug_trap = 1.0
            #---------------------------------------------------#
            # if show_surfaces:
            #     print "\tDEBUG: current bindx, sindx: ", bindx, sindx
            #     print "\tDEBUG: building# ", bindx, "p0: ", p0, " p1: ", p1
            #     print "------------------\n\tcurrent surface#", sindx, " vals:\n", self.bldgs[bindx][sindx]
            #     self.debugVisualizeLinePolygon(p0, p1, bindx, sindx)
            isIntersecting, iPoint = self.isSurfaceIntersecting(p0, p1, n_surf)
            if isIntersecting:
                cur_surf = self.bldgs[bindx][sindx]
                if self.isPointInsidePolygon(iPoint, cur_surf, n_surf):
                    #print "\tDEBUG: is intersecting"
                    return True
        #print "\tDEBUG: building not intersecting.."
        return False

    def isPolyhedronIntersecting(self, p0, p1, surf_pn):
        t_E = 0.0
        t_L = 1.0
        dS = p1 - p0

        surf_count, _ = surf_pn.shape
        for sindx in range(surf_count):
            sx, sy, sz, nx, ny, nz =  surf_pn[sindx, :]
            sn = np.array([nx, ny, nz])
            p0_to_s = np.array( [sx, sy, sz] ) - p0
            n =  np.dot(p0_to_s, sn)
            # dS = p1- p0
            d = np.dot(dS, sn)
            if d ==  0.0:
                if n < 0.0:
                    return  False
                else:
                    continue
            t = n / d
            if d < 0.0:
                t_E = max(t, t_E)
                if t_E > t_L:
                    return  False
            else: # d > 0.0
                t_L = min(t, t_L)
                if t_E > t_L:
                    return  False
        return True

    def isLOS(self, p0, p1):
        for bindx in range(self.total_building):
            if self.isBuildingBBoxIntersecting(p0, p1,bindx):
                if self.isBuildingSurfaceIntersecting(p0, p1, bindx):
                    return False
        return True

    def calculateLOS(self, fso_tx_indx = None):
        total_fso_tx, _ = self.fso_tx.shape
        tx_indx_range = range(total_fso_tx-1)
        if not fso_tx_indx is None:
            tx_indx_range = [fso_tx_indx]
        for i in tx_indx_range:
            p0 = self.fso_tx[ i, :]
            for j in range(total_fso_tx):
                p1 = self.fso_tx[j, :]
                if self.isLOS(p0, p1):
                    self.fso_los.append((i,j))
        return


def driverDynamicGraphGenerator():
    input_file = 'world_trade_center'
    fso_tower_height_ft = 10.0

    dgg = DynamicGraphGenerator()
    dgg.load3DBuildingData(input_file)
    dgg.addFSOTowers(tower_height_ft=fso_tower_height_ft)
    dgg.calculateLOS(fso_tx_indx=16)
    #dgg.visualizeBuildingBBox(showNormals=False, showFSOLinks = True)

    dgg.visualize3Dbuilding(showFSOLinks=True, deBugShowAllLinks=False)
    return

if __name__ == '__main__':
    driverDynamicGraphGenerator()