'''
Example script demonstrating usage of the kicad_board module.
Execute from within KiCad's Python with 'import Board_Example'
'''
import sys
sys.path.append('..')
import kicad_board as kb
import pcbnew

board = kb.KiCadBoard()
board.load_data('../Design Data')

######################################################################

##                           Specify cuts                           ##

######################################################################
corners = [[0, 0], [0, board.params['Ly_board']],
                                 [board.params['Lx_board'], board.params['Ly_board']], 
                                 [board.params['Lx_board'], 0], 
                                 [0, 0]]
board.edge_cuts(corners, refresh=True)

######################################################################

##                          Place vias                              ##

######################################################################
for wg in board.board_data['siw']:
    for poly in wg:
        for via in poly:
            board.add_via(via[0],
                          via[1], 
                          board.params['via_diameter'],
                          'F.Cu',
                          'B.Cu',
                          board.gnd_net,
                          2.3*board.params['via_diameter'])
pcbnew.Refresh()

######################################################################

##                       Place components                           ##

######################################################################
varactor_list = [i for i in board.component_list if i.startswith('D')]

for i in range(len(varactor_list)):
    board.move_module(varactor_list[i], 
                      board.board_data['varactor'][i][0][0][0],
                      board.board_data['varactor'][i][0][0][1],
                      rotation=0,
                      flip=True)

header_list = [i for i in board.component_list if i.startswith('J')]
board.move_module(header_list[0],
                  board.params['Lx_board']/2,
                  board.params['Ly_board']/8)
pcbnew.Refresh()

######################################################################

##                          Place test                              ##

######################################################################
board.write_text('Example Board', board.params['Lx_board']/2, board.params['Ly_board']-board.params['Ly_board']/8, refresh=True)

######################################################################

##                       Make copper zones                          ##

######################################################################

#### Construct top layer zones
top_layer_board_polygon = board.make_outline(corners)

for element in board.board_data['cELC']:
    polygon_temp = board.make_outline(element[1])
    top_layer_board_polygon.BooleanSubtract(polygon_temp, 0)

board.add_zone(top_layer_board_polygon, board.gnd_net, refresh=True)

for i in range(len(board.board_data['cELC'])):
    net_string = 'Net-(D'+str(i+1)+'-Pad1)'
    net_temp = board.nets.find(net_string).value()[1]
    polygon_temp = board.make_outline(board.board_data['cELC'][i][0])
    board.add_zone(polygon_temp, net_temp)

pcbnew.Refresh()

#### Construct bottom layer zones
bottom_layer_board_polygon = board.make_outline(corners)
board.add_zone(corners, board.gnd_net, layer='B.Cu', refresh=True)
