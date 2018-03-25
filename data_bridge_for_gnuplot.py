import PyGnuplot as gp
import numpy as np
# gp.s([X,Y,Z])
# gp.c('plot "tmp.dat" u 1:2 w lp')
# gp.c('replot "tmp.dat" u 1:3 w lp')
# gp.p('myfigure.ps')
'''
task:
1. load building structure from bldg, stat file
2. show in gnuplot
'''

class GnuVisualize(object):
    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.bldg_file =  input_filename+".bldg"
        self.stat_file = input_filename+".stat"
        self.grnd_file = input_filename+".dyn"
        self.roof_file = input_filename+".roof"
        self.fso_file = input_filename+".fso"
        self.los_file = input_filename+".dyn"
        self.ref_x = self.ref_y = 0.0
        self.total_building = 0
        self.loadStatFile()
        print  "DEBUG:", self.total_building, self.ref_x, self.ref_y
        return


    def loadStatFile(self):
        with open(self.stat_file, 'r') as f:
            line= f.readline().split(',')
            self.total_building =  int(line[1])
            xs, ys, zs = f.readline().split(';')
            xmin, _ = xs.split(',')
            ymin, _ = ys.split(',')
            self.ref_x, self.ref_y = float(xmin), float(ymin)
        return

    def loadBldgSurfaces(self):
        with open(self.bldg_file, 'r') as infile, open(self.input_filename+".bldgnu", "w") as outfile:
            for bindx in range(self.total_building):
                _, total_surfs = infile.readline().split(',')
                _ = infile.readline() #skip bounding box line
                total_surfs = int(total_surfs)
                for sindx in range(total_surfs):
                    cur_surf_points = infile.readline().split(';')
                    total_cur_points = len(cur_surf_points)
                    for pindx in range( total_cur_points+1 ):
                        pindx = pindx%total_cur_points
                        x, y, z = cur_surf_points[pindx].split(',')
                        x, y, z = float(x)-self.ref_x, float(y)-self.ref_y, float(z)
                        ftext = str(x)+", "+str(y)+", "+str(z)+"\n"
                        outfile.write(ftext)
                        outfile.flush()
                    outfile.write("\n")
                    outfile.flush()
        return

    def loadFSOLos(self, max_len_km):
        max_len_ft = max_len_km*3280.84
        fso_loc = np.loadtxt(self.fso_file, dtype=np.float, delimiter=',')
        with open(self.los_file, 'r') as fin, open(self.input_filename+".losnu", 'w') as fout:
            for line in fin:
                p1, p2 = line.split(',')
                p1, p2 = int(p1), int(p2)
                x1, y1, z1 = fso_loc[p1 - 1, 1:]
                x2, y2, z2 = fso_loc[p2 - 1, 1:]
                dist_sq = (x1-x2)**2.0+(y1-y2)**2.0+(z1-z2)**2.0
                if dist_sq > max_len_ft*max_len_ft: continue
                fout.write( str(x1)+", "+str(y1)+", "+str(z1)+", " )
                fout.write( str(x2)+", "+str(y2)+", "+str(z2) +"\n")
        return

def numpyGraphTest():
    x = 6000*3000
    #a = np.full((x, 1000), False, dtype = np.bool)
    #print a.shape
    return
if __name__ == '__main__':
    inputfile = './all_wtc_data/world_trade_center'
    max_fso_link_km = 0.4
    gv = GnuVisualize(inputfile)
    gv.loadBldgSurfaces()
    gv.loadFSOLos(max_fso_link_km)
    #numpyGraphTest()