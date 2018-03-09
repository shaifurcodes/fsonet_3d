'''
author: Md Shaifur Rahman (mdsrahman@cs.stonybrook.edu)
        Stony Brook University, NY.
'''
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3

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
        self.fso_tx = None
        self.MIN_X, self.MAX_X,\
        self.MIN_Y, self.MAX_Y,\
        self.MIN_Z, self.MAX_Z = range(6)

        return

    def loadBldgFile(self, input_file):
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

        self.bldg_bounding_box = np.zeros(shape = ( self.total_building, 6), dtype = np.float)
        return

    def load3DBuildingData(self, input_filepath):
        self.loadStatFile(input_filepath+'.stat')
        self.loadBldgFile(input_filepath+'.bldg')
        return

    def addFSOTowers(self, tower_height_ft):
        self.fso_tx = np.zeros( shape=(self.total_building, 3) , dtype = np.float )
        for bindx in range(self.total_building):
            bmin_x, bmax_x, bmin_y, bmax_y, bmin_z, bmax_z = self.bldg_bounding_box[bindx,: ]
            fso_x = (bmin_x + bmax_x)/2.0
            fso_y = (bmin_y + bmax_y)/2.0
            fso_z = bmax_z+tower_height_ft
            self.fso_tx[bindx, : ] = [fso_x, fso_y, fso_z]
        return
    def visualize3Dbuilding(self, showFsoTowers = False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for bindx, bldg_surfs in enumerate(self.bldgs):
            for surf in bldg_surfs:
                ax.add_collection3d(mpl3.art3d.Line3DCollection([surf], colors='k', linewidths=1.0))
                ax.add_collection3d(mpl3.art3d.Poly3DCollection([surf], facecolors='w'))
        ax.set_xlim3d(left=0-500,   right=self.max_x+500)
        ax.set_ylim3d(bottom=0-500, top=self.max_y+500)
        ax.set_zlim3d(bottom=self.min_z - 5, top=self.max_z+15)
        # if showFsoTowers:
        #     for i in range(self.total_building-1):
        #         for j in range(i+1, self.total_building):
        #             ax.plot( [self.fso_tx[i, 0], self.fso_tx[j, 0] ], \
        #                      [self.fso_tx[i, 1], self.fso_tx[j, 1]], \
        #                      [self.fso_tx[i, 2], self.fso_tx[j, 2]],
        #                      linestyle=':', color='r', linewidth=1
        #                      )
        # ax.auto_scale_xyz([0, 10], [0, 10], [0, 0.100])
        plt.show()
        #plt.savefig('./test.png',  dpi = 300)
        plt.close()
        return
def driverDynamicGraphGenerator():
    input_file = 'world_trade_center'
    fso_tower_height_ft = 10.0

    dgg = DynamicGraphGenerator()
    dgg.load3DBuildingData(input_file)
    dgg.addFSOTowers(tower_height_ft=fso_tower_height_ft)

    dgg.visualize3Dbuilding(showFsoTowers=True)
    return

if __name__ == '__main__':
    driverDynamicGraphGenerator()