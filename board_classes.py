import csv
import os
from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt

cm = 0.01

### EACH ITEM IS A LENGTH N_POLYGONS LIST OF N_VERTICES X 2 ARRAYS

class Board():

    def __init__(self, Lx_board, Ly_board):

        self.params = locals()
        self.params.pop('self')
        self.items = {}
        self.corners = [np.array([[0, 0], [0, self.params['Ly_board']],
                                 [self.params['Lx_board'], self.params['Ly_board']], 
                                 [self.params['Lx_board'], 0], 
                                 [0, 0]])]

    def add(self, component):
        self.items[component.params['name']] = component
    
    def rotate(self, theta, point_list=None):
        '''
        Rotates an object defined by the N_polygons x N_vertices x 2 array point_list about the origin by an angle theta.

        theta: angle in degrees
        point_list: corners defining polygon
        '''
        rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                    [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        
        if point_list is not None:
            for i in range(len(point_list)):
                point_list[i] = np.matmul(rotation_matrix, point_list[i][:,:,None])[:,:,0]
        else:
            raise Exception('Must supply point_list.')

    def reflect_x(self, point_list=None):
        '''
        Reflects an object defined by the N_polygons x N_vertices x 2 array point_list over the x axis.
        '''
        reflection_matrix = np.array([[1, 0], [0, -1]])

        if point_list is not None:
            for i in range(len(point_list)):
                point_list[i] = np.matmul(reflection_matrix, point_list[i][:,:,None])[:,:,0]
            return point_list
        else:
            raise Exception('Must supply point_list.')

    def reflect_y(self, point_list=None):
        '''
        Reflects an object defined by the N_polygons x N_vertices x 2 array point_list over the y axis.
        '''
        reflection_matrix = np.array([[-1, 0], [0, 1]])

        if point_list is not None:
            for i in range(len(point_list)):
                point_list[i] = np.matmul(reflection_matrix, point_list[i][:,:,None])[:,:,0]
            return point_list
        else:
            raise Exception('Must supply point_list.')

    def move(self, x=0, y=0, point_list=None):
        '''
        Moves an object defined by the N_polygons x N_vertices x 2 array point_list by an amount (x, y)
        '''
        if point_list is not None:
            for i in range(len(point_list)):
                point_list[i] = point_list[i] + np.array([x, y])[None,:]
            return point_list
        else:
            raise Exception('Must supply point_list.')

    def vertex_to_endpoints(self, vertex_list):
        '''
        Creates an (N-1) x (x, y) x (start, end) array of line segments from an N x 2 array of corners.
        vertex_list: N_polygons x N_vertices x 2 ordered array of corners. Include the starting corner if creating closed loop.
        
        '''
        if vertex_list is not None:
            endpoint_list = []
            for i in range(len(vertex_list)):
                N_corners = vertex_list[i].shape[0]
                endpoint_array = np.empty((N_corners-1, 2, 2), dtype=np.float32)
                
                for j in range(N_corners-1):
                    endpoint_array[j,:,:] = np.transpose(np.array([vertex_list[i][j,:], vertex_list[i][j+1,:]]))
                endpoint_list.append(endpoint_array)
                        
            return endpoint_list
        else:
            raise Exception('Must supply vertex_list.')

    def make_via_list(self, endpoint_list, pitch, even=True):
        '''
        Generates N x 2 array of via positions from N_lines x (x, y) x (start, end) array of line segments.

        pitch: via pitch (float)
        even: if True, rounds via pitch so that vias are evenly spaced between endpoints
        '''
        if endpoint_list is not None:
            point_list = []
            for i in range(len(endpoint_list)):
                point_array = np.array([]).reshape(0, 2)
                for j in range(endpoint_list[i].shape[0]):
                    vector = endpoint_list[i][j,:,1] - endpoint_list[i][j,:,0]
                    vector_length = np.linalg.norm(vector)
                    unit_vector = vector/vector_length
                    
                    N = np.ceil(vector_length/pitch).astype(np.int32)
                    
                    if even is True:
                        pitch_even = vector_length/(N.astype(np.float32)) #sets via pitch so that vias are evenly spaced over length of line
                    else:
                        pitch_even = pitch

                    points_temp = np.array([]).reshape((0, 2))
                    for n in range(N):
                    
                        new_point = endpoint_list[i][j,:,0] + n*pitch_even*unit_vector

                        points_temp = np.append(points_temp, new_point[None,:], axis=0)

                    point_array = np.append(point_array, points_temp, axis=0)
                point_list.append(point_array)
            
            return point_list
        else:
            raise Exception('Must supply endpoint_list.')

    def plot(self, ax=None):
        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        pool = cycle(cycle_colors)

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        for i in range(len(self.corners)):
            ax.fill(self.corners[i][:,0], self.corners[i][:,1], facecolor='none', edgecolor='black')
        for key, item in self.items.items():
            item.plot(ax=ax, color=next(iter(pool)), plot_type='all')

    def tolist(self):
        for key, component in self.items.items():
            component.tolist_items = [[polygon.tolist() for polygon in component_item] for component_item in component.items]

    def export_items(self, filepath):
        # export board parameters dictionary
        filename = filepath + '/' + 'board_params.csv'
        with open(filename, 'w', newline='') as file:
            w = csv.writer(file)
            for key, val in self.params.items():
                w.writerow([key, val])
        
        positions_filepath = '{}/Positions'.format(filepath)
        if not os.path.exists(positions_filepath):
            os.makedirs(positions_filepath)

        # export board corners
        filename = positions_filepath + '/' + 'board_corners.csv'
        corner_list = []
        for i in range(len(self.corners)):
            corner_list.append(self.corners[i].tolist())
        with open(filename, 'w', newline='') as file:
            write = csv.writer(file)
            write.writerows(corner_list)
        
        self.tolist()
        for key, item in self.items.items():
            item.export_items(positions_filepath)

    def list_shape(nested_list):
        dims = []
        temp = nested_list
        while True:
            try:
                dims.append(len(temp))
                temp = temp[0]
            except:
                break
        return dims

    def __str__(self):
        param_table = '{:<25} {:<25}\n'.format('PARAMETER', 'VALUE')     # print column names
    
        # print each data item.
        for key, value in self.params.items():
            param_table += '{:<25} {:<25}\n'.format(key, value)
        comps = ''
        for key, value in self.items.items():
            comps = comps + key + '; '
        param_table += '\n{:<25}\n'.format('COMPONENTS')
        param_table += '{:<25}\n'.format(comps)
        return param_table

class Component():      ## EACH COMPONENT SHOULD HAVE ROTATE, REFLECT, MOVE, AND ADD (TO ITEM LIST) OPERATIONS

    def __init__(self, name=None):
        self.params = locals()
        self.params.pop('self')
        self.items = []
        self.make_point_list()
    
    def make_point_list(self):
        self.point_list = [np.array([0, 0])[None,:]]
        return

    def rotate(self, theta):
        '''
        Rotates an object defined by the N_polygons x N_vertices x 2 array point_list about the origin by an angle theta.

        theta: angle in degrees
        point_list: corners defining polygon
        '''
        rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                    [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        
        for i in range(len(self.point_list)):
            self.point_list[i] = np.matmul(rotation_matrix, self.point_list[i][:,:,None])[:,:,0]
        return self

    def reflect_x(self):
        '''
        Reflects an object defined by the N_polygons x N_vertices x 2 array point_list over the x axis.
        '''
        reflection_matrix = np.array([[1, 0], [0, -1]])

        for i in range(len(self.point_list)):
            self.point_list[i] = np.matmul(reflection_matrix, self.point_list[i][:,:,None])[:,:,0]
        return self

    def reflect_y(self):
        '''
        Reflects an object defined by the N_polygons x N_vertices x 2 array point_list over the y axis.
        '''
        reflection_matrix = np.array([[-1, 0], [0, 1]])

        for i in range(len(self.point_list)):
            self.point_list[i] = np.matmul(reflection_matrix, self.point_list[i][:,:,None])[:,:,0]
        return self

    def move(self, x=0, y=0):
        '''
        Moves an object defined by the N_polygons x N_vertices x 2 array point_list by an amount (x, y)
        '''
        for i in range(len(self.point_list)):
            self.point_list[i] = self.point_list[i] + np.array([x, y])[None,:]
        return self
    
    def add(self):
        self.items.append(self.point_list)
        return self

    def reset(self):
        self.make_point_list()
        return self

    def plot(self, ax=None, color='red', plot_type='template'):
        '''
        Plots polygon or scatter plot, depending on type of object.
        plot_type: 'template' or 'all' -- plots the point list or set of all point lists in items, respectively.
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
        if plot_type=='template':
            for i in range(len(self.point_list)):
                    if self.params['type']=='scatter':
                        ax.scatter(self.point_list[i][:,0], self.point_list[i][:,1], s=5, color=color)
                    else:
                        ax.fill(self.point_list[i][:,0], self.point_list[i][:,1], facecolor='none', edgecolor=color)
        elif plot_type=='all':
            for t in range(len(self.items)):
                for i in range(len(self.items[t])):
                    if self.params['type']=='scatter':
                        ax.scatter(self.items[t][i][:,0], self.items[t][i][:,1], s=5, color=color)
                    else:
                        ax.fill(self.items[t][i][:,0], self.items[t][i][:,1], facecolor='none', edgecolor=color)
        plt.axis('equal')
    
    def export_items(self, filepath):
        filename = filepath + '/' + self.params['name'] + '.csv'
        with open(filename, 'w', newline='') as file:
            write = csv.writer(file)
            write.writerows(self.tolist_items)

    def __str__(self):
        param_table = '{:<25} {:<25}\n'.format('PARAMETER', 'VALUE')     # print column names
    
        # print each data item.
        for key, value in self.params.items():
            param_table += '{:<25} {:<25}\n'.format(key, value)
        return param_table

class Transition(Component):

    def __init__(self, L_track, L_taper, L_tip, w_gap, w_track, w_tip, w1, w2, name='transition'):
        self.params = locals()
        self.params.pop('self')
        self.params['type'] = 'polygon'
        self.items = []
        self.make_point_list()

    def make_point_list(self):
        '''
        Constructs ordered vertex list describing coax transition polygon, starting at top left at origin and moving clockwise
        See dimension labels in companion figure
        Creates (2 x N x 2) vertex array point_list
        First dimension specifies upper and lower polygons
        Uses self.params dictionary
        Required dictionary keys: L_track, L_taper, L_tip, w_gap, w_track, w1, w2 
        '''
        x_start = 0
        y_start = 0

        point_list_upper = np.array([x_start, y_start])[None,:]

        deltas = np.array([[self.params['L_track'], 0],
                           [self.params['L_taper'], self.params['w1']/2 - self.params['w_track']/2 - self.params['w_gap']],
                           [0, -(self.params['w1'] - self.params['w2'])/2],
                           [-self.params['L_taper'], -(self.params['w2'] - self.params['w_track'])/2],
                           [-(self.params['L_track'] - self.params['L_tip']), 0],
                           [-self.params['L_tip'], -(self.params['w_track'] - self.params['w_tip'])/2],
                           [0, (self.params['w_track'] - self.params['w_tip'])/2 + self.params['w_gap']]
                            ])

        for i in range(deltas.shape[0]):
            point_list_upper = np.append(point_list_upper, 
                                               point_list_upper[i,:][None,:] + deltas[i,:][None,:],
                                               axis = 0)
        
        point_list_lower = np.copy(point_list_upper)
        point_list_lower = Board.reflect_x(self, point_list=point_list_lower[None,:,:])[0,:,:]
        point_list_lower = Board.move(self, x=0, y=-(2*self.params['w_gap'] + self.params['w_track']), point_list=point_list_lower[None,:,:])[0,:,:]

        self.point_list = [point_list_upper, point_list_lower]


class Termination(Component):

    def __init__(self, w_gap_top, L_track, L_taper, w_gap, w_track, w1, w2, name='termination'):
        self.params = locals()
        self.params.pop('self')
        self.params['type'] = 'polygon'
        self.items = []
        self.make_point_list()

    def make_point_list(self):
        '''
        Constructs ordered vertex list describing termination polygon, starting at top left at origin and moving clockwise
        See dimension labels in companion figure
        Creates vertex array point_list
        Uses self.params dictionary
        Required dictionary keys: w_gap_top, L_track, L_taper, w_gap, w_track, w1, w2
        '''
        x_start = 0
        y_start = 0

        point_list = np.array([x_start, y_start])[None,:]

        deltas = np.array([[self.params['w_gap_top'] + self.params['L_track'], 0],
                           [self.params['L_taper'], self.params['w1']/2 - self.params['w_gap'] - self.params['w_track']/2],
                           [0, -(self.params['w1'] - self.params['w2'])/2],
                           [-self.params['L_taper'], -(self.params['w2'] - self.params['w_track'])/2],
                           [-self.params['L_track'], 0],
                           [0, -self.params['w_track']],
                           [self.params['L_track'], 0],
                           [self.params['L_taper'], -(self.params['w2'] - self.params['w_track'])/2],
                           [0, -(self.params['w1'] - self.params['w2'])/2],
                           [-self.params['L_taper'], self.params['w1']/2 - self.params['w_track']/2 - self.params['w_gap']],
                           [-self.params['L_track'] - self.params['w_gap_top'], 0],
                           [0, self.params['w_track'] + 2*self.params['w_gap']]
                            ])
        for i in range(deltas.shape[0]):
            point_list = np.append(point_list, 
                                        point_list[i,:][None,:] + deltas[i,:][None,:],
                                        axis = 0)
        self.point_list = [point_list]

class Choke(Component):

    def __init__(self, theta_filter, r_filter, choke_pad_diameter, name='choke', n_points_curve1=50, n_points_curve2=50, filter_scale=1.2):
        self.params = locals()
        self.params.pop('self')
        self.params['type'] = 'polygon'
        self.items = []
        self.make_point_list()

    def make_point_list(self):
        '''
        Makes corner array defining RF decoupling filters
        Starts at top left corner, places center of pad at origin
        Uses self.RF_choke_params dictionary
        Required dictionary keys: theta_filter, r_filter, choke_pad_diameter
        '''
        
        theta0 = np.radians((180 - self.params['theta_filter'])/2 + self.params['theta_filter'])
        theta_list1 = np.linspace(theta0, theta0 - np.radians(self.params['theta_filter']), self.params['n_points_curve1'])
        theta_list2 = np.linspace(theta0 - np.radians(self.params['theta_filter']), -np.pi - (np.pi - theta0), self.params['n_points_curve2'])

        point_list_inner = np.array([]).reshape(0, 2)
        for i in range(self.params['n_points_curve1']):
            point_list_inner = np.append(point_list_inner,
                                                        np.array([self.params['r_filter'] * np.cos(theta_list1[i]),
                                                            self.params['r_filter'] * np.sin(theta_list1[i])])[None,:], axis=0)

        for i in range(self.params['n_points_curve2']):
            point_list_inner = np.append(point_list_inner,
                                                np.array([self.params['choke_pad_diameter']/2 * np.cos(theta_list2[i]),
                                                            self.params['choke_pad_diameter']/2 * np.sin(theta_list2[i])])[None,:], axis=0)

        point_list_inner = np.append(point_list_inner,
                                                    np.array([self.params['r_filter'] * np.cos(theta_list1[0]),
                                                            self.params['r_filter'] * np.sin(theta_list1[0])])[None,:], axis=0)
        
        point_list_outer = self.params['filter_scale'] * point_list_inner
        filter_r_gap = self.params['filter_scale'] * self.params['r_filter'] - self.params['r_filter']
        point_list_outer = point_list_outer - np.array([0, filter_r_gap/2])[None,:]

        self.params['filter_r_gap'] = filter_r_gap
        self.point_list = [point_list_inner, point_list_outer]

class cELC(Component):

    def __init__(self, wx_cELC, wy_cELC, wx_gap, wy_gap, w_center, w_gap, w_arm1, w_arm2, cELC_varactor_gap, w_cELC_varactor, name='cELC'):
        self.params = locals()
        self.params.pop('self')
        self.params['type'] = 'polygon'
        self.items = []
        self.make_point_list()

    def make_point_list(self):
        '''
        Constructs two ordered vertex lists describing cELC polygons, starting at top left and moving clockwise, origin at center.
        See dimension labels in companion figure
        Creates two vertex polygons: outer vertex list and inner vertex list
        Uses self.params dictionary
        Required dictionary keys: wx_cELC, wy_cELC, wx_gap, wy_gap, w_center, w_gap, w_arm1, w_arm2, cELC_varactor_gap, w_cELC_varactor
        '''

        self.params['wx_tot'] = self.params['wx_cELC'] + 2*self.params['wx_gap']
        self.params['wy_tot'] = self.params['wy_cELC'] + 2*self.params['wy_gap']
        stub_length = (self.params['wy_gap'] - self.params['cELC_varactor_gap'])/2

        x_start = -self.params['wx_cELC']/2
        y_start = self.params['wy_cELC']/2
        point_list_inner = np.array([x_start, y_start])[None,:]
        deltas = np.array([[self.params['wx_cELC'], 0],
                           [0, -(self.params['wy_cELC'] - self.params['w_gap'])/2],
                           [-self.params['w_arm2'], 0],
                           [0, (self.params['wy_cELC'] - self.params['w_gap'] - 2*self.params['w_arm1'])/2],
                           [-(self.params['wx_cELC'] - 2*self.params['w_arm2'] - self.params['w_center'])/2, 0],
                           [0, -(self.params['wy_cELC'] - 2*self.params['w_arm1'])],
                           [(self.params['wx_cELC'] - 2*self.params['w_arm2'] - self.params['w_center'])/2, 0],
                           [0, (self.params['wy_cELC'] - self.params['w_gap'] - 2*self.params['w_arm1'])/2],
                           [self.params['w_arm2'], 0],
                           [0, -(self.params['wy_cELC'] - self.params['w_gap'])/2],
                           [-(self.params['wx_cELC'] - self.params['w_cELC_varactor'])/2, 0],
                           [0, -stub_length],
                           [-self.params['w_cELC_varactor'], 0],
                           [0, stub_length],
                           [-(self.params['wx_cELC'] - self.params['w_cELC_varactor'])/2, 0],
                           [0, (self.params['wy_cELC'] - self.params['w_gap'])/2],
                           [self.params['w_arm2'], 0],
                           [0, -(self.params['wy_cELC'] - self.params['w_gap'] - 2*self.params['w_arm1'])/2],
                           [(self.params['wx_cELC'] - 2*self.params['w_arm2'] - self.params['w_center'])/2, 0],
                           [0, self.params['wy_cELC'] - 2*self.params['w_arm1']],
                           [-(self.params['wx_cELC'] - 2*self.params['w_arm2'] - self.params['w_center'])/2, 0],
                           [0, -(self.params['wy_cELC'] - self.params['w_gap'] - 2*self.params['w_arm1'])/2],
                           [-self.params['w_arm2'], 0],
                           [0, (self.params['wy_cELC'] - self.params['w_gap'])/2]
                            ])
        for i in range(deltas.shape[0]):
            point_list_inner = np.append(point_list_inner, 
                                        point_list_inner[i,:][None,:] + deltas[i,:][None,:],
                                        axis = 0)

        x_start = -self.params['wx_tot']/2
        y_start = self.params['wy_tot']/2
        point_list_outer = np.array([x_start, y_start])[None,:]
        deltas = np.array([[self.params['wx_tot'], 0],
                           [0, -self.params['wy_tot']],
                           [-(self.params['wx_tot'] - self.params['w_cELC_varactor'])/2, 0],
                           [0, stub_length],
                           [-self.params['w_cELC_varactor'], 0],
                           [0, -stub_length],
                           [-(self.params['wx_tot'] - self.params['w_cELC_varactor'])/2, 0],
                           [0, self.params['wy_tot']]
                            ])
        for i in range(deltas.shape[0]):
            point_list_outer = np.append(point_list_outer, 
                                        point_list_outer[i,:][None,:] + deltas[i,:][None,:],
                                        axis = 0)

        self.point_list = [point_list_inner, point_list_outer]

class SIW(Component):

    def __init__(self, L_wg, w_wg, L_taper, L_track, w_wall, pitch, even=True, mode='closed', name='SIW'):
        self.params = locals()
        self.params['Lx_tot'] = L_wg + 2*L_taper + 2*L_track
        self.params.pop('self')
        self.params['type'] = 'scatter'
        self.items = []
        self.make_point_list()

    def make_point_list(self):
        '''
        Constructs two ordered vertex lists describing SIW polygon(s), starting at top left and moving clockwise, origin at center.
        See dimension labels in companion figure
        Creates one or two via lists, depending on mode.
        Uses self.params dictionary
        Required dictionary keys: L_wg, w_wg, L_taper, L_track, w_wall
        mode: 'closed', 'open', 'half-open-left', or 'half-open-right' (string)
        '''
        x_start = 0
        y_start = 0

        point_list_1 = np.array([x_start, y_start])[None,:]

        if self.params['mode'] == 'reflect':
            deltas = np.array([[self.params['L_track'], 0],
                               [self.params['L_taper'], (self.params['w_wg'] - self.params['w_wall'])/2],
                               [self.params['L_wg'], 0],
                               [0, -self.params['w_wg']],
                               [-self.params['L_wg'], 0],
                               [-self.params['L_taper'], (self.params['w_wg'] - self.params['w_wall'])/2],
                               [-self.params['L_track'], 0]
                               ])
            for i in range(deltas.shape[0]):
                point_list_1 = np.append(point_list_1, 
                                            point_list_1[i,:][None,:] + deltas[i,:][None,:],
                                            axis=0)
            self.point_list = [point_list_1]
        else:
            deltas = np.array([[self.params['L_track'], 0],
                            [self.params['L_taper'], (self.params['w_wg'] - self.params['w_wall'])/2],
                            [self.params['L_wg'], 0],
                            [self.params['L_taper'], -(self.params['w_wg'] - self.params['w_wall'])/2],
                            [self.params['L_track'], 0]
                                ])
            for i in range(deltas.shape[0]):
                point_list_1 = np.append(point_list_1, 
                                            point_list_1[i,:][None,:] + deltas[i,:][None,:],
                                            axis=0)
            point_list_2 = np.copy(point_list_1)
            point_list_2 = Board.move(self, x=0, y=-self.params['w_wall'], point_list=Board.reflect_x(self, point_list=point_list_2[None,:,:]))[0,:,:]
            point_list_2 = np.flip(point_list_2, axis=0)

            if self.params['mode']=='open':
                self.point_list = [point_list_1, point_list_2]
            elif self.params['mode']=='half-open-left':
                self.point_list = [np.append(point_list_1, point_list_2, axis=0)]
            elif self.params['mode']=='half-open-right':
                self.point_list = [np.append(point_list_2, point_list_1, axis=0)]
            elif self.params['mode']=='closed':
                point_list = np.append(point_list_1, point_list_2, axis=0)
                self.point_list = [np.append(point_list, np.array([x_start, y_start])[None,:], axis=0)]
        
        self.vertex_to_endpoints()
        self.make_via_list()

    def vertex_to_endpoints(self):
        '''
        Creates an (N-1) x (x, y) x (start, end) array of line segments from an N x 2 array of corners.
        point_list: N_polygons x N_vertices x 2 ordered array of corners. Include the starting corner if creating closed loop.
        
        '''
        self.endpoint_list = []
        for i in range(len(self.point_list)):
            N_corners = self.point_list[i].shape[0]
            endpoint_array = np.empty((N_corners-1, 2, 2), dtype=np.float32)
            
            for j in range(N_corners-1):
                endpoint_array[j,:,:] = np.transpose(np.array([self.point_list[i][j,:], self.point_list[i][j+1,:]]))
            self.endpoint_list.append(endpoint_array)
                        
    def make_via_list(self, omit_startpoint=True):
        '''
        Generates N x 2 array of via positions from N_lines x (x, y) x (start, end) array of line segments.

        pitch: via pitch (float)
        even: if True, rounds via pitch so that vias are evenly spaced between endpoints
        '''
        self.point_list = []
        for i in range(len(self.endpoint_list)):
            point_array = np.array([]).reshape(0, 2)
            for j in range(self.endpoint_list[i].shape[0]):
                vector = self.endpoint_list[i][j,:,1] - self.endpoint_list[i][j,:,0]
                vector_length = np.linalg.norm(vector)
                unit_vector = vector/vector_length
                
                N = np.ceil(vector_length/self.params['pitch']).astype(np.int32)
                
                if self.params['even'] is True:
                    pitch_even = vector_length/(N.astype(np.float32)) #sets via pitch so that vias are evenly spaced over length of line
                else:
                    pitch_even = self.params['pitch']

                points_temp = np.array([]).reshape((0, 2))
                for n in range(N):
                
                    new_point = self.endpoint_list[i][j,:,0] + n*pitch_even*unit_vector

                    points_temp = np.append(points_temp, new_point[None,:], axis=0)

                point_array = np.append(point_array, points_temp, axis=0)
            if omit_startpoint:
                point_array = point_array[1:,:]
            self.point_list.append(point_array)

class Patch(Component):

    def __init__(self, W, L, W_is, L_is, W_ms, L_ms, keepout_ratio, name='patch'):
        self.params = locals()
        self.params.pop('self')
        self.params['type'] = 'polygon'
        self.items = []
        self.make_point_list()

    def make_point_list(self):
        '''
        Constructs two ordered vertex lists describing the patch polygon and a keepout region, starting at top left and moving clockwise, origin at center.
        See dimension labels in companion figure.
        Creates two vertex polygons: outer vertex list and inner vertex list
        Uses self.params dictionary
        Required dictionary keys: W, L, W_is, L_is, L_ms, keepout_ratio
        '''

        x_start = -self.params['W']/2
        y_start = self.params['L']/2
        point_list_inner = np.array([x_start, y_start])[None,:]
        deltas = np.array([[self.params['W'], 0],
                           [0, -self.params['L']],
                           [-(self.params['W']-2*self.params['W_is']-self.params['W_ms'])/2, 0],
                           [0, self.params['L_is']],
                           [-self.params['W_is'], 0],
                           [0, -self.params['L_is']-self.params['L_ms']],
                           [-self.params['W_ms'], 0],
                           [0, self.params['L_is']+self.params['L_ms']],
                           [-self.params['W_is'], 0],
                           [0, -self.params['L_is']],
                           [-(self.params['W']-2*self.params['W_is']-self.params['W_ms'])/2, 0],
                           [0, self.params['L']],
                            ])
        for i in range(deltas.shape[0]):
            point_list_inner = np.append(point_list_inner, 
                                        point_list_inner[i,:][None,:] + deltas[i,:][None,:],
                                        axis = 0)

        x_start = -self.params['W']/2 * self.params['keepout_ratio']
        y_start = self.params['L']/2 * self.params['keepout_ratio']
        point_list_outer = np.array([x_start, y_start])[None,:]
        deltas = np.array([[self.params['W'], 0],
                           [0, -self.params['L']],
                           [-(self.params['W']-self.params['W_ms'])/2, 0],
                           [0, -self.params['L_ms']],
                           [-self.params['W_ms'], 0],
                           [0, self.params['L_ms']],
                           [-(self.params['W']-self.params['W_ms'])/2, 0],
                           [0, self.params['L']],
                            ]) * self.params['keepout_ratio']
        for i in range(deltas.shape[0]):
            point_list_outer = np.append(point_list_outer, 
                                        point_list_outer[i,:][None,:] + deltas[i,:][None,:],
                                        axis = 0)

        self.point_list = [point_list_inner, point_list_outer]