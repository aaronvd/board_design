import pcbnew
import csv
import ast

m = 1000000000.0    #kicad internal units are nm
degrees = 10.0      #kicad internal units are 0.1 degree

class KiCadBoard():

    def __init__(self, filename=None):
        if filename is None:
            self.BOARD = pcbnew.GetBoard()
        else:
            self.BOARD = pcbnew.LoadBoard(filename)

    def list_shape(self, nested_list):
        dims = []
        temp = nested_list
        while True:
            try:
                dims.append(len(temp))
                temp = temp[0]
            except:
                break
        return dims

    def load_list(self, filename):
        '''
         Loads in CSV file, supporting nested lists (i.e. CSVs exported from multidimensional arrays)
        '''
        list1 = []
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                list1.append(row)
        try:
            list1 = [[float(v) for v in r] for r in list1]
        except:
            list1 = [[ast.literal_eval(i) for i in v] for v in list1]
        print('List Dimensions:' + self.list_shape(list1))
        return list1

    def get_layertable(self):
        '''
        creates dictionary for conveniently referencing layers, e.g. as layertable["F.Cu"]
        '''
        self.layertable = {}
        numlayers = pcbnew.PCB_LAYER_ID_COUNT
        for i in range(numlayers):
            self.layertable[self.BOARD.GetLayerName(i)] = i

    def edge_cuts(self, corner_list, refresh=False):
        '''
        Creates polygon defining board edge geometry
        '''
        edgecut = self.layertable['Edge.Cuts']
        for i in range(len(corner_list)-1):
            seg = pcbnew.DRAWSEGMENT(self.BOARD)
            self.BOARD.Add(seg)
            seg.SetStart(pcbnew.wxPoint(corner_list[i][0]*m, corner_list[i][1]*m))
            seg.SetEnd(pcbnew.wxPoint(corner_list[i+1][0]*m, corner_list[i+1][1]*m))
            seg.SetLayer(edgecut)

        if refresh:
            pcbnew.Refresh()

    def add_via(self, x, y, via_diameter, top_layer, bottom_layer, net, via_width, via_type='through', refresh=False):
        newvia = pcbnew.VIA(self.BOARD)
        self.BOARD.Add(newvia)
        newvia.SetPosition(pcbnew.wxPoint(x*m, y*m))
        if via_type == 'through':
            newvia.SetViaType(pcbnew.VIA_THROUGH)
        elif via_type == 'blind':
            newvia.SetViaType(pcbnew.VIA_BLIND_BURIED)
        newvia.SetDrill(int(via_diameter*m))
        newvia.SetWidth(int(via_width*m))
        newvia.SetNet(net)
        newvia.SetLayerPair(self.layertable[top_layer], self.layertable[bottom_layer]) #THIS NEEDS TO GO LAST
        
        if refresh:
            pcbnew.Refresh()

    def add_module(self, x, y, footprint_lib, component_name, refresh=False):
        '''
        Adds component corresponding to component_name from library footprint_lib at position (x, y)
        '''
        io = pcbnew.PCB_IO()
        mod = io.FootprintLoad(footprint_lib, component_name)
        pt = pcbnew.wxPoint(x*m, y*m)
        mod.SetPosition(pt)
        self.BOARD.Add(mod)
        
        if refresh:
            pcbnew.Refresh()

    def write_text(self, text_string, text_x, text_y, size=0.003, thickness=.00015, rotation=180, layer="F.SilkS", refresh=False):
        '''
        Places text on front silkscreen layer (by default)
        '''
        text = pcbnew.TEXTE_PCB(self.BOARD)
        text.SetText(text_string)
        text.SetPosition(pcbnew.wxPoint(text_x*m, text_y*m))
        text.Rotate(pcbnew.wxPoint(text_x*m, text_y*m), rotation*degrees)
        text.SetTextSize(pcbnew.wxSize(size*m, size*m))
        text.SetThickness(int(thickness*m))
        text.SetLayer(self.layertable[layer])
        self.BOARD.Add(text)
        
        if refresh:
            pcbnew.Refresh()

    def add_track(self, endpoints_array, trace_width, layer, net, refresh=False):
        '''
        Adds copper traces defined by endpoints_array of size (n_traces, 2, 2)
        Second dimension of endpoints_array defines trace start (x, y)
        Third dimension of endpoints_array defines trace end (x, y)
        '''
        start_x = endpoints_array[0][0]
        start_y = endpoints_array[0][1]
        end_x = endpoints_array[1][0]
        end_y = endpoints_array[1][1]

        track = pcbnew.TRACK(self.BOARD)
        track.SetStart(pcbnew.wxPoint(start_x*m, start_y*m))
        track.SetEnd(pcbnew.wxPoint(end_x*m, end_y*m))
        track.SetWidth(int(trace_width*m))
        track.SetLayer(self.layertable[layer])
        track.SetNet(net)
        
        self.BOARD.Add(track)
        
        if refresh:
            pcbnew.Refresh()

    def make_outline(self, vertex_array):
        '''
        Creates and returns polygon outline from (n_corners+1 x 2) array
        '''
        sp = pcbnew.SHAPE_POLY_SET()
        sp.NewOutline()
        for i in range(len(vertex_array)-1):
            sp.Append(int(vertex_array[i][0]*m), int(vertex_array[i][1]*m))
        sp.thisown = 0
        return sp

    def add_zone(self, vertex_array, net, keepout=False, layer="F.Cu", refresh=False):
        '''
        Creates and adds zone from (n_corners+1 x 2) vertex array
        '''
        if not isinstance(vertex_array, pcbnew.SHAPE_POLY_SET):
            vertex_array = self.make_outline(vertex_array)
        
        zone = pcbnew.ZONE_CONTAINER(self.BOARD)
        zone.SetOutline(vertex_array)
        zone.SetLayer(self.layertable[layer])
        zone.SetIsKeepout(keepout)
        zone.SetDoNotAllowCopperPour(keepout)
        zone.SetNet(net)
        zone.thisown = 0
        
        self.BOARD.Add(zone)
        
        if refresh:
            pcbnew.Refresh()









