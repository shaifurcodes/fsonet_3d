import numpy as np
class DynGraphMaker(object):
    def __init__(self, total_nodes, target_file_path, short_link, long_link):
        self.total_nodes = total_nodes
        self.target_file_path = target_file_path
        self.fso_file = self.target_file_path+".fso"
        self.los_file = self.target_file_path+".dyn"
        self.cover_file = self.target_file_path+".covnu"
        self.short_d = short_link
        self.long_d = long_link


        temp = np.loadtxt(self.fso_file,  delimiter=',', dtype=np.float)
        xy_locs = temp[:, 1:3 ]
        self.node_locs = np.vstack(( np.array([0.0, 0.0]), xy_locs ))

        n, _ = self.node_locs.shape
        self.backbone = np.full((n, n), False, dtype=np.bool)
        self.los = np.full((n, n), np.inf, dtype=np.float)
        with open(self.los_file, 'r') as f:
            for line in f:
                i, j = line.split(',')
                i,j = int(i), int(j)
                x1, y1 = self.node_locs[i, :]
                x2, y2 = self.node_locs[j, :]
                d = np.sqrt( (x1-x2)**2+(y1-y2)**2 )
                if d <= self.long_d:
                    self.los[i, j]  = d
                    self.los[j, i]  = d
                if d<= self.short_d:
                    #print d
                    self.backbone[i, j] = True
                    self.backbone[j, i] = True

        temp = np.loadtxt(self.cover_file, delimiter=',', dtype=np.float)
        self.bs_nodes = []
        for i in temp[:, 0]:
            self.bs_nodes.append( int(i) )
        print "Total Backbone edges:", np.sum(self.backbone)/2
        return

    def selectGateways(self, garea_x, garea_y ):
        #TODO: iterate over every area of the above size, bin all locations, randomly select one bs with highest node degree in los
        max_x = np.max(self.node_locs[:, 0])
        max_y = np.max(self.node_locs[:, 1])
        self.gateways = []
        for x in np.arange(garea_x/2.0, max_x+garea_x/2.0, garea_x):
            for y in np.arange(garea_y/2.0, max_y+garea_x/2.0, garea_y,):
                min_d_sq = np.inf
                min_loc_x = min_loc_y = -1
                min_loc_id = None
                for cur_bs in self.bs_nodes:
                    x1, y1 = self.node_locs[cur_bs]
                    d_sq = (x-x1)**2+(y-y1)**2
                    if d_sq < min_d_sq:
                        min_loc_x, min_loc_y, min_loc_id = x1, y1, cur_bs
                        min_d_sq = d_sq
                self.gateways.append(( min_loc_id, min_loc_x, min_loc_y))
        with open(self.target_file_path+".gatenu", 'w') as f:
            for i in self.gateways:
                id, x,y = i
                f.write(str(id)+','+str(x)+','+str(y)+'\n')
        return

    def saveBackboneGraphForGnuplot(self):
        with open(self.target_file_path+".backnu",'w') as f:
            for i in self.bs_nodes:
                for j in self.bs_nodes:
                    if i>=j :
                        continue
                    if self.backbone[i, j] == False:
                        continue
                    x1, y1 = self.node_locs[i, :]
                    x2, y2 = self.node_locs[j, :]
                    f.write(str(x1)+", "+str(y1)+", "+str(x2)+", "+str(y2)+"\n")
        return

if __name__ == '__main__':
    total_nodes = 9731
    short_link_m = 100.0
    long_link_m = 1000.0
    gateway_grid_x_ft = 4000.0
    gateway_grid_y_ft = 4000.0
    target_file_path = './all_wtc_data/world_trade_center'
    dgm = DynGraphMaker(total_nodes=total_nodes, target_file_path = target_file_path, short_link=short_link_m*3.28084, long_link=long_link_m*3.28084)
    #dgm.saveBackboneGraphForGnuplot()
    dgm.selectGateways(garea_x=gateway_grid_x_ft, garea_y=gateway_grid_y_ft)
