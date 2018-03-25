import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

class CoverageMaker(object):
    def __init__(self, coverage_radius, grid_size, area_x, area_y, loc_file, out_coverage_file):
        self.r = coverage_radius
        self.grid_size = grid_size
        self.area_x = area_x
        self.area_y = area_y
        self.out_coverage_file = out_coverage_file

        self.coverage_grid_size = 0
        self.max_x_grid = 0
        self.max_y_grid = 0


        self.grid_coverage =  None
        self.loc_file = loc_file
        self.locs = None
        self.isLocTaken = None
        self.setOverlapMatrix()
        self.setCoverageMatrix()
        self.loadLoc()
        self.loc_heap =  None
        return

    def setOverlapMatrix(self):
        self.coverage_grid_size = int( np.ceil( self.r / self.grid_size ) )
        self.coverage_mask = np.full( (1+2*self.coverage_grid_size, 1+2*self.coverage_grid_size), True, dtype = np.bool )
        center_x = center_y = self.grid_size*self.coverage_grid_size+self.grid_size/2.0
        dim_x, dim_y =  self.coverage_mask.shape
        for i in range(dim_x):
            for j in range(dim_y):
                x = i*self.grid_size+self.grid_size/2.0
                y = j*self.grid_size+self.grid_size/2.0
                d_sq = (x-center_x)**2+(y-center_y)**2
                if d_sq > self.r**2:
                    self.coverage_mask[i, j ] = False
        return

    def setCoverageMatrix(self):
        dim_x = int( np.ceil(self.area_x / self.grid_size) )
        dim_y = int( np.ceil(self.area_y / self.grid_size) )
        self.grid_coverage = np.full((dim_x, dim_y), True, dtype= np.bool)
        self.max_x_grid, self.max_y_grid = dim_x-1, dim_y-1
        return

    def loadLoc(self):
        temp = np.loadtxt(self.loc_file, delimiter=',', dtype=np.float)
        self.locs = temp[:, 1:3]
        x, _ = self.locs.shape
        self.isLocTaken = np.full( (x), False, dtype=np.bool)
        return

    def getCoverageScore(self, indx):
        x, y = self.locs[indx, :]
        gx, gy = int(np.floor(x / self.grid_size)), int(np.floor(y / self.grid_size))
        min_cx = min_cy = 0
        max_cx, max_cy = self.coverage_mask.shape
        min_gx = gx - self.coverage_grid_size
        min_gy = gy - self.coverage_grid_size
        max_gx = gx + self.coverage_grid_size
        max_gy = gy + self.coverage_grid_size
        if min_gx < 0:
            min_cx = 0 - min_gx
            min_gx = 0
        if min_gy < 0:
            min_cy = 0 - min_gy
            min_gy = 0
        if max_gx > self.max_x_grid:
            max_cx = max_cx - (max_gx - self.max_x_grid)
            max_gx = self.max_x_grid
        if max_gy > self.max_y_grid:
            max_cy = max_cy - (max_gy - self.max_y_grid)
            max_gy = self.max_y_grid

        temp = np.logical_and(self.grid_coverage[min_gx:max_gx + 1, min_gy:max_gy + 1],
                              self.coverage_mask[min_cx: max_cx, min_cy: max_cy])
        return -np.sum(temp)


    def getCurrentMaxCoverIndx(self):
        #TODO: for each loc, find the extent on grid_coverge matrix and count the (False, True) i.e. uncovered values and push into the heapQ
        self.loc_heap = []
        indx = np.argwhere(self.isLocTaken==False)
        for i in indx[:, 0]:
            score = self.getCoverageScore(i)
            heappush( self.loc_heap, ( score, i) )
            #print min_gx, min_gy, max_gx, max_gy
        score, indx = heappop(self.loc_heap)
        return indx, -score

    def updateCover(self, indx):
        x, y = self.locs[indx, :]
        gx, gy = int(np.floor(x / self.grid_size)), int(np.floor(y / self.grid_size))
        min_cx = min_cy = 0
        max_cx, max_cy = self.coverage_mask.shape
        min_gx = gx - self.coverage_grid_size
        min_gy = gy - self.coverage_grid_size
        max_gx = gx + self.coverage_grid_size
        max_gy = gy + self.coverage_grid_size
        if min_gx < 0:
            min_cx = 0 - min_gx
            min_gx = 0
        if min_gy < 0:
            min_cy = 0 - min_gy
            min_gy = 0
        if max_gx > self.max_x_grid:
            max_cx = max_cx - (max_gx - self.max_x_grid)
            max_gx = self.max_x_grid
        if max_gy > self.max_y_grid:
            max_cy = max_cy - (max_gy - self.max_y_grid)
            max_gy = self.max_y_grid
        prev_val = np.sum(self.grid_coverage)
        self.grid_coverage[min_gx:max_gx + 1, min_gy:max_gy + 1] = np.where(self.coverage_mask[min_cx: max_cx, min_cy: max_cy]==True, False, self.grid_coverage[min_gx:max_gx + 1, min_gy:max_gy + 1])

        new_val = np.sum(self.grid_coverage)
        print prev_val, "-->", new_val
        return

    def saveCovers(self):
        with open(self.out_coverage_file, 'w') as f:
            indx = np.argwhere(self.isLocTaken == True)
            for i in indx[:, 0]:
                x, y = self.locs[i,:]
                f.write(str(x)+", "+str(y)+"\n")
        return

    def runSetCover(self):
        x, y = self.grid_coverage.shape
        total_uncovered_grid = x*y
        while True:
            current_uncoverd = np.sum(self.grid_coverage)
            if current_uncoverd ==0:
                break
            indx, score = self.getCurrentMaxCoverIndx()
            print current_uncoverd, indx, score, round( 100*current_uncoverd/total_uncovered_grid, 2),"%"
            if score == 0:
                break
            self.updateCover(indx)
            self.isLocTaken[indx] = True
        self.saveCovers()
        return

    def saveBuilidngRoofs(self, roof_file, out_file, total_bldg, ref_x, ref_y):
        with open(roof_file, 'r') as infile, open(out_file, "w") as outfile:
            for bindx in range(tot_bldg):
                line = infile.readline().split(',')
                total_surfs = int(line[1])
                for sindx in range(total_surfs):
                    cur_surf_points = infile.readline().split(';')
                    total_cur_points = len(cur_surf_points)
                    for pindx in range(total_cur_points + 1):
                        pindx = pindx % total_cur_points
                        x, y, z = cur_surf_points[pindx].split(',')
                        x, y, z = float(x) - ref_x, float(y) - ref_y, float(z)
                        ftext = str(x) + ", " + str(y) + "\n"
                        outfile.write(ftext)
                        outfile.flush()
                    outfile.write("\n")
                    outfile.flush()

    def debugVisualize(self):
        fig, ax = plt.subplots()
        indx = np.argwhere(self.isLocTaken == True)
        for i in indx[:, 0]:
            x , y = self.locs[i]
            ax.add_artist(plt.Circle((x, y), self.r, color='b', fill=False))
        ax.set_xlim((0, self.area_x+10))
        ax.set_ylim((0, self.area_y+10))
        fig.savefig('plotcircles2.png')
        return

if __name__ == '__main__':

    coverage_radius_meter = 100.0*3.28084
    grid_size_ft = 10.0
    area_x_km = 5.0
    area_y_km = 3.0
    loc_file = './all_wtc_data/world_trade_center.fso'
    out_coverage_file = './all_wtc_data/world_trade_center.covnu'

    gv = CoverageMaker(coverage_radius=coverage_radius_meter,
                       grid_size=grid_size_ft,
                       area_x=area_x_km*3280.84,
                       area_y=area_y_km*3280.84,
                       loc_file=loc_file,
                       out_coverage_file=out_coverage_file)

    #gv.runSetCover()
    roof_file = './all_wtc_data/world_trade_center.roof'
    out_file = './all_wtc_data/world_trade_center.roofnu'
    tot_bldg = 9731
    ref_x = 979573.179945
    ref_y = 196978.123482
    gv.saveBuilidngRoofs(roof_file, out_file, tot_bldg, ref_x, ref_y)
    #gv.debugVisualize()
    #print gv.isLocTaken


