import os
import pcbnew
import csv
import ast

m = 1000000000.0    #kicad internal units are nm
degrees = 10.0      #kicad internal units are 0.1 degree

class KiCadBoard():

    def __init__(self, filename=None):
        self.board_data = {}
        if filename is None:
            self.BOARD = pcbnew.GetBoard()
        else:
            self.BOARD = pcbnew.LoadBoard(filename)
        self.get_layertable()
        self.nets = self.BOARD.GetNetsByName()
        if 'GND' in [str(i) for i in self.nets.keys()]:
            self.gnd_net = self.nets.find('GND').value()[1]
        self.component_list = []
        for module in self.BOARD.GetFootprints():
            self.component_list.append(module.GetReference())

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
        
        print('List Dimensions: {}'.format(self.list_shape(list1)))
        return list1

    def load_data(self, data_directory):
        with open('{}/board_params.csv'.format(data_directory), 'r') as infile:
            r = csv.reader(infile)
            self.params = {rows[0]:float(rows[1]) for rows in r if len(rows) == 2}

        self.corners = self.load_list('{}/Positions/board_corners.csv'.format(data_directory))
        
        positions_directory = '{}/Positions'.format(data_directory)
        files = os.listdir(positions_directory)
        for f in files:
            data = f.split('.')
            if data[1] == 'csv':
                self.board_data[data[0]] = self.load_list('{}/{}'.format(positions_directory, f))

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
            seg = pcbnew.PCB_SHAPE(self.BOARD)
            seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
            seg.SetStart(pcbnew.VECTOR2I(pcbnew.wxPoint(int(corner_list[i][0]*m), int(corner_list[i][1]*m))))
            seg.SetEnd(pcbnew.VECTOR2I(pcbnew.wxPoint(int(corner_list[i+1][0]*m), int(corner_list[i+1][1]*m))))
            seg.SetLayer(edgecut)
            self.BOARD.Add(seg)

        if refresh:
            pcbnew.Refresh()

    def add_via(self, x, y, via_diameter, top_layer, bottom_layer, net, via_width, via_type='through', refresh=False):
        newvia = pcbnew.PCB_VIA(self.BOARD)
        newvia.SetPosition(pcbnew.VECTOR2I(pcbnew.wxPoint(x*m, y*m)))
        if via_type == 'through':
            newvia.SetViaType(pcbnew.VIATYPE_THROUGH)
        elif via_type == 'blind':
            newvia.SetViaType(pcbnew.VIATYPE_BLIND_BURIED)
        newvia.SetDrill(int(via_diameter*m))
        newvia.SetWidth(int(via_width*m))
        newvia.SetNet(net)
        newvia.SetLayerPair(self.layertable[top_layer], self.layertable[bottom_layer]) 
        self.BOARD.Add(newvia)
        
        if refresh:
            pcbnew.Refresh()

    def add_module(self, x, y, footprint_lib, component_name, refresh=False):
        '''
        Adds component corresponding to component_name from library footprint_lib at position (x, y)
        '''
        io = pcbnew.PCB_IO()
        mod = io.FootprintLoad(footprint_lib, component_name)
        pt = pcbnew.wxPoint(x*m, y*m)
        mod.SetPosition(pcbnew.VECTOR2I(pt))
        self.BOARD.Add(mod)
        
        if refresh:
            pcbnew.Refresh()

    def write_text(self, text_string, text_x, text_y, size=0.003, thickness=.00015, rotation=180, layer="F.Silkscreen", refresh=False):
        '''
        Places text on front silkscreen layer (by default)
        '''
        text = pcbnew.PCB_TEXT(self.BOARD)
        text.SetText(text_string)
        position_temp = pcbnew.VECTOR2I(pcbnew.wxPoint(text_x*m, text_y*m))
        text.SetPosition(position_temp)
        text.SetTextAngle(pcbnew.EDA_ANGLE(rotation, pcbnew.DEGREES_T))
        text.SetTextSize(pcbnew.VECTOR2I(pcbnew.wxSize(size*m, size*m)))
        text.SetLayer(self.layertable[layer])
        self.BOARD.Add(text)
        
        if refresh:
            pcbnew.Refresh()

    def move_module(self, module_reference, x, y, rotation=None, flip=False, refresh=False):
        module = self.BOARD.FindFootprintByReference(module_reference)
        position_temp = pcbnew.VECTOR2I(pcbnew.wxPoint(x*m, y*m))
        module.SetPosition(position_temp)

        if flip:
            module.Flip(position_temp, False)

        if rotation is not None:
            module.SetOrientation(pcbnew.EDA_ANGLE(rotation, pcbnew.DEGREES_T))

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

        track = pcbnew.PCB_TRACK(self.BOARD)
        track.SetStart(pcbnew.VECTOR2I(pcbnew.wxPoint(start_x*m, start_y*m)))
        track.SetEnd(pcbnew.VECTOR2I(pcbnew.wxPoint(end_x*m, end_y*m)))
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
        
        zone = pcbnew.ZONE(self.BOARD)
        zone.SetOutline(vertex_array)
        zone.SetLayer(self.layertable[layer])
        zone.SetDoNotAllowCopperPour(keepout)
        if net is not None:
            zone.SetNet(net)
        zone.thisown = 0
        
        self.BOARD.Add(zone)
        
        if refresh:
            pcbnew.Refresh()










