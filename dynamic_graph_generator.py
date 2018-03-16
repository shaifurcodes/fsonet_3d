'''
author: Md Shaifur Rahman (mdsrahman@cs.stonybrook.edu)
        Stony Brook University, NY.
'''
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3
import sys
from timeit import default_timer as mytimer
import cProfile

class DynamicGraphGenerator(object):
    '''
    this class load the saved 3D-building file, calculate LOS and save the resulting graph
    '''
    def __init__(self):

        self.input_bldg_file = None
        self.total_building = 0
        self.total_surface = 0

        self.ref_x = None
        self.ref_y = None

        self.fso_tower_height_ft = 0.0
        self.min_bldg_perimeter_req_ft = 0.0
        self.building_per_hash_bin = 0

        self.fso_los = []

        self.max_x = -float('inf')
        self.max_y = -float('inf')

        self.min_z = None
        self.max_z = None

        self.bldg_bounding_box = None
        self.bldgs = []
        self.bldg_bbox_nsurf = []
        self.bldg_nsurf = []
        self.fso_tx = []
        self.MIN_X, self.MAX_X,\
        self.MIN_Y, self.MAX_Y,\
        self.MIN_Z, self.MAX_Z = range(6)
        self.bldg_hash = []
        self.bldg_hash_xcount = 0
        self.bldg_hash_ycount = 0
        self.bldg_hash_xdist = 0.0
        self.bldg_hash_ydist = 0.0
        self.max_short_link = 0.0
        self.max_long_link = 0.0

        return

    def setMaxLinkLength(self, max_long_link):
        self.max_long_link = max_long_link
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
        self.input_bldg_file = input_filepath
        self.loadStatFile(input_filepath+'.stat')
        self.loadBldgFile(input_filepath+'.bldg')
        return

    def isMinBldgPerimeter(self, bindx):
        [xmin, xmax, ymin, ymax, _, _] = self.bldg_bounding_box[bindx]
        perimeter = 2*((xmax - xmin)+(ymax-ymin))
        if perimeter < self.min_bldg_perimeter_req_ft:
            return  False
        return  True

    def saveFSOTXLocs(self):
        with open(self.input_bldg_file+".fso", "w") as fso_file:
            for bindx, fso_loc in enumerate(self.fso_tx):
                bid = bindx+1
                p = fso_loc[0, :]
                if np.isnan(p[0]):
                    bid = -bid #negative bid means no fso-tower here
                    fso_file.write(str(bid)+", 0, 0, 0"+"\n")
                else:
                    fso_file.write(str(bid)+", "+str(p[0])+", "+str(p[1])+", "+str(p[2])+"\n")
        return

    def addFSOTowers(self, tower_height_ft, min_bldg_perimeter_req_ft):
        self.fso_tower_height_ft = tower_height_ft
        self.min_bldg_perimeter_req_ft = min_bldg_perimeter_req_ft

        self.fso_tx = []
        with open(self.input_bldg_file+'.roof', 'r') as surf_file:
            for bindx in range(self.total_building):
                fso_tx_x, fso_tx_y, fso_tx_z = 0.0, 0.0, -float('inf')
                _, surf_count = surf_file.readline().split(',')
                for sindx in range(int(surf_count)):
                    cur_surf_txt = surf_file.readline().split(';')
                    for cur_vertices in cur_surf_txt:
                        x, y, z = cur_vertices.split(',')
                        x, y, z = np.float(x) - self.ref_x, np.float(y) - self.ref_y, np.float(z)
                        if fso_tx_z < z:
                            fso_tx_x, fso_tx_y, fso_tx_z = x, y, z
                if self.isMinBldgPerimeter(bindx) is False:
                    self.fso_tx.append(np.array( [(np.nan, np.nan, np.nan)], dtype=np.float) )
                else:
                    self.fso_tx.append( np.array([ (fso_tx_x, fso_tx_y, fso_tx_z+self.fso_tower_height_ft) ],\
                                             dtype = np.float) )
        self.saveFSOTXLocs()
        return

    def visualize3Dbuilding(self, showFSOLinks=False, showFSOTowers = False):
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
                #---! uncomment the line below to show -----closed polygon
                #ax.plot([x1, xn], [y1, yn], [z1, zn], linestyle ='-', color='k', linewidth=1.0)


        ax.set_xlim3d(left=0-500,   right=self.max_x+500)
        ax.set_ylim3d(bottom=0-500, top=self.max_y+500)
        ax.set_zlim3d(bottom=self.min_z - 5, top=self.max_z+15)
        if showFSOLinks:
            for i,j in self.fso_los:
                ax.plot( [self.fso_tx[i][0, 0], self.fso_tx[j][0, 0] ], \
                         [self.fso_tx[i][0, 1], self.fso_tx[j][0, 1]], \
                         [self.fso_tx[i][0, 2], self.fso_tx[j][0, 2]],
                         linestyle=':', color='b', linewidth=1.0 )
        if showFSOTowers:
            for cur_fso_tx in self.fso_tx:
                [x,y,z] = cur_fso_tx[0,:]
                if np.isnan(x): continue
                ax.plot( [x, x], [y, y], [z, z-self.fso_tower_height_ft], linestyle ='-', color='r', linewidth=4.0)
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
            isIntersecting, iPoint = self.isSurfaceIntersecting(p0, p1, n_surf)
            if isIntersecting:
                cur_surf = self.bldgs[bindx][sindx]
                if self.isPointInsidePolygon(iPoint, cur_surf, n_surf):
                    return True
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

    def getNPairsForLOSCalc(self):
        pair_count = 0
        for i in range(self.total_building-1):
            p0 =  self.fso_tx[i][0, :]
            if np.isnan(p0[0]):
                continue
            for j in range(i+1, self.total_building):
                p1 = self.fso_tx[j][0, :]
                if np.isnan(p1[0]):
                    continue
                if self.isLink(p0, p1):
                    pair_count += 1
        return pair_count

    def isLOS(self, p0, p1):
        #start_t = mytimer()
        stat_bbox_check = 0
        stat_bldg_check = 0

        ibldgs_list = self.getIntersectingBuildingIDs(p0, p1)
        for bindx in ibldgs_list:
            stat_bbox_check += 1
            if self.isBuildingBBoxIntersecting(p0, p1,bindx):
                stat_bldg_check += 1
                if self.isBuildingSurfaceIntersecting(p0, p1, bindx):
                    return False
        return True

    def isLink(self, p0, p1):
        d = p1 - p0
        if np.sum(np.dot(d, d)) <= self.max_long_link*self.max_long_link:
            return  True
        return False

    def calculateLOS(self, max_long_link_ft = None, building_per_hash_bin = 5):
        if not max_long_link_ft is None:
            self.max_long_link = max_long_link_ft
        self.building_per_hash_bin = building_per_hash_bin
        self.hashXYBuilding()

        total_fso_tx = len( self.fso_tx )
        stat_fso_tx_pair_count = 0
        stat_los_time = 0.0
        stat_total_los_pairs = self.getNPairsForLOSCalc()
        with open(self.input_bldg_file+'.dyn', 'w') as output_dyn_file:
            for i in range(total_fso_tx-1):
                p0 = self.fso_tx[i][ 0, :]
                if np.isnan(p0[0]): continue
                for j in range(i+1,total_fso_tx):
                    p1 = self.fso_tx[j][0, :]
                    if np.isnan(p1[0]): continue
                    if not self.isLink(p0, p1):
                        continue
                    start_t = mytimer()
                    stat_fso_tx_pair_count += 1
                    if self.isLOS(p0, p1):
                        self.fso_los.append((i,j))
                        output_dyn_file.write(str(i+1)+", "+str(j+1)+"\n")
                        output_dyn_file.flush()
                    stat_los_time += mytimer() - start_t
                if stat_fso_tx_pair_count > 0:
                    stat_cur_avg_los_calc_time = stat_los_time / stat_fso_tx_pair_count
                    stat_expected_remaining_time = (stat_total_los_pairs - stat_fso_tx_pair_count)*stat_cur_avg_los_calc_time
                    print "Progress: fso_tx: ", i+1, "/", total_fso_tx, \
                        " LOS-pairs: ", stat_fso_tx_pair_count, "/", stat_total_los_pairs,\
                        " Avg. time per LOS-pair: ", stat_cur_avg_los_calc_time, "sec "\
                        " Expected Remaining time: ",  stat_expected_remaining_time,"sec"
        return

    def hashXYBuilding(self):
        ratio_xy = 1.0*np.ceil(self.max_x) / np.ceil( self.max_y )
        self.bldg_hash_ycount = int(np.ceil( np.sqrt(1.0 * self.total_building / (self.building_per_hash_bin * ratio_xy)) ) )
        self.bldg_hash_xcount = int(np.ceil(ratio_xy*self.bldg_hash_ycount))
        self.bldg_hash_xdist = 1.0 * np.ceil(self.max_x) / self.bldg_hash_xcount
        self.bldg_hash_ydist = 1.0 * np.ceil(self.max_y) / self.bldg_hash_ycount
        #--------init hash table------#
        for i in range(self.bldg_hash_xcount+1):
            cur_hash = []
            for j in range(self.bldg_hash_ycount+1):
                cur_hash.append([])
            self.bldg_hash.append(cur_hash)

        for bindx, bbox in enumerate( self.bldg_bounding_box ):
            xmin, xmax, ymin, ymax, _, _ = bbox
            grid_x_min = int( xmin / self.bldg_hash_xdist)
            grid_x_max = int( xmax / self.bldg_hash_xdist)
            grid_y_min = int( ymin / self.bldg_hash_ydist)
            grid_y_max = int( ymax / self.bldg_hash_ydist)
            for i in range(grid_x_min, grid_x_max+1):
                for j in range(grid_y_min, grid_y_max+1):
                    self.bldg_hash[i][j].append(bindx)
        return

    def findVisitedGrids(self, x1, y1, x2, y2):
        points = []

        dx = x2 - x1
        dy = y2 - y1
        x = x1
        y = y1

        xstep = 1
        ystep = 1

        points.append((x1, y1))
        if dy < 0:
            ystep = -1
            dy = -dy

        if dx < 0:
            xstep = -1
            dx = -dx

        ddy = 2 * dy
        ddx = 2 * dx
        if ddx >= ddy:
            errorprev = error = dx
            for i in range(dx):
                x += xstep
                error += ddy
                if error > ddx:
                    y += ystep
                    error -= ddx
                    if error + errorprev < ddx:
                        points.append((x, y - ystep))
                    elif error + errorprev > ddx:
                        points.append((x - xstep, y))
                    else:
                        points.append((x, y - ystep))
                        points.append((x - xstep, y))
                points.append((x, y))
                errorprev = error
        else:
            errorprev = error = dy
            for i in range(dy):
                y += ystep
                error += ddx
                if error > ddy:
                    x += xstep
                    error -= ddy
                    if error + errorprev < ddy:
                        points.append((x - xstep, y))
                    elif error + errorprev > ddy:
                        points.append((x, y - ystep))
                    else:
                        points.append((x - xstep, y))
                        points.append((x, y - ystep))
                points.append((x, y))
                errorprev = error
        return points

    def getIntersectingBuildingIDs(self, p0, p1):
        x0, y0, _ = p0
        x1, y1, _ = p1
        p0_gridx, p0_gridy = int(x0 / self.bldg_hash_xdist), int(y0 / self.bldg_hash_ydist)
        p1_gridx, p1_gridy = int(x1 / self.bldg_hash_xdist), int(y1 / self.bldg_hash_ydist)
        all_grids = self.findVisitedGrids(p0_gridx, p0_gridy, p1_gridx, p1_gridy)
        visited_bldgs = []
        for cur_grid in all_grids:
            i,j = cur_grid
            cur_bldgs = list(self.bldg_hash[i][j])
            visited_bldgs += cur_bldgs
        visited_bldgs_set = list( set(visited_bldgs) )
        return visited_bldgs_set

def driverDynamicGraphGenerator():

    #-----params--------------------#
    input_file = 'world_trade_center'
    max_link_length_km = 1.0 # affects run time
    fso_tower_height_ft = 30.0
    building_perimeter_req_ft = 80.0
    building_per_bin = 5
    #-------end params-----------------#

    start_t = mytimer()
    dgg = DynamicGraphGenerator()
    dgg.load3DBuildingData(input_file)
    dgg.setMaxLinkLength(max_link_length_km*3280.84)
    dgg.addFSOTowers(tower_height_ft=fso_tower_height_ft,
                     min_bldg_perimeter_req_ft=building_perimeter_req_ft)
    dgg.calculateLOS(building_per_hash_bin=building_per_bin)
    print 'Execution time:', np.round((mytimer() - start_t), 3), "seconds"
    #dgg.visualize3Dbuilding(showFSOLinks=False, showFSOTowers=True ) #comment this before final run
    return

if __name__ == '__main__':
    profileCode = False
    if profileCode:
        cProfile.run('driverDynamicGraphGenerator()', 'expProfile.cprof')
    else:
        driverDynamicGraphGenerator()
