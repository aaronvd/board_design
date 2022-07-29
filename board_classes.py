import csv
import numpy as np
from matplotlib import pyplot as plt

cm = 0.01

### EACH ITEM IS AN N_POLYGONS X N_VERTICES X 2 ARRAY

class Board():

    def __init__(self, Lx_board, Ly_board):

        self.params = locals()
        self.params.pop('self')
        self.items = {}
        self.corners = np.array([[0, 0], [0, self.params['Ly_board']],
                                 [self.params['Lx_board'], self.params['Ly_board']], 
                                 [self.params['Lx_board'], 0], 
                                 [0, 0]])

    def add(self, component):
        self.items[component.params['name']] = component.items
    
    def rotate(self, theta, vertex_list=None):
        '''
        Rotates an object defined by the N_polygons x N_vertices x 2 array vertex_list about the origin by an angle theta.

        theta: angle in degrees
        vertex_list: corners defining polygon
        '''
        rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                    [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        
        if vertex_list is not None:
            for i in range(vertex_list.shape[0]):
                vertex_list[i,:,:] = np.matmul(rotation_matrix, vertex_list[i,:,:,None])[:,:,0]
        else:
            raise Exception('Must supply vertex_list.')

    def reflect_x(self, vertex_list=None):
        '''
        Reflects an object defined by the N_polygons x N_vertices x 2 array vertex_list over the x axis.
        '''
        reflection_matrix = np.array([[1, 0], [0, -1]])

        if vertex_list is not None:
            for i in range(vertex_list.shape[0]):
                vertex_list[i,:,:] = np.matmul(reflection_matrix, vertex_list[i,:,:,None])[:,:,0]
            return vertex_list
        else:
            raise Exception('Must supply vertex_list.')

    def reflect_y(self, vertex_list=None):
        '''
        Reflects an object defined by the N_polygons x N_vertices x 2 array vertex_list over the y axis.
        '''
        reflection_matrix = np.array([[-1, 0], [0, 1]])

        if vertex_list is not None:
            for i in range(vertex_list.shape[0]):
                vertex_list[i,:,:] = np.matmul(reflection_matrix, vertex_list[i,:,:,None])[:,:,0]
            return vertex_list
        else:
            raise Exception('Must supply vertex_list.')

    def move(self, x=0, y=0, vertex_list=None):
        '''
        Moves an object defined by the N_polygons x N_vertices x 2 array vertex_list by an amount (x, y)
        '''
        if vertex_list is not None:
            for i in range(vertex_list.shape[0]):
                vertex_list[i,:,:] = vertex_list[i,:,:] + np.array([x, y])[None,:]
            return vertex_list
        else:
            raise Exception('Must supply vertex_list.')

    def vertex_to_endpoints(self, vertex_list):
        '''
        Creates an (N-1) x (x, y) x (start, end) array of line segments from an N x 2 array of corners.
        vertex_list: N x 2 ordered array of corners. Include the starting corner if creating closed loop.
        
        '''
        N_corners = vertex_list.shape[0]
        endpoint_array = np.empty((N_corners-1, 2, 2), dtype=np.float32)
        
        for i in range(N_corners-1):
            endpoint_array[i,:,:] = np.transpose(np.array([vertex_list[i,:], vertex_list[i+1,:]]))
                
        return endpoint_array

    def make_via_list(self, endpoints_array, pitch, even=True):
        '''
        Generates N x 2 array of via positions from N_lines x (x, y) x (start, end) array of line segments.

        pitch: via pitch (float)
        even: if True, rounds via pitch so that vias are evenly spaced between endpoints
        '''
        
        points_array = np.array([]).reshape(0, 2)
        for i in range(endpoints_array.shape[0]):
            vector = endpoints_array[i,:,1] - endpoints_array[i,:,0]
            vector_length = np.linalg.norm(vector)
            unit_vector = vector/vector_length
            
            N = np.ceil(vector_length/pitch).astype(np.int32)
            
            if even is True:
                pitch_even = vector_length/(N.astype(np.float32)) #sets via pitch so that vias are evenly spaced over length of line
            else:
                pitch_even = pitch

            points_list = np.array([]).reshape((0, 2))
            for n in range(N):
            
                new_point = endpoints_array[i,:,0] + n*pitch_even*unit_vector

                points_list = np.append(points_list, new_point[None,:], axis=0)

            points_array = np.append(points_array, points_list, axis=0)
        
        return points_array
    
    def export_items(self, filepath):
        for item in self.items:
            filename = filepath + item + '.csv'
            with open(filename, 'w') as file:
                write = csv.writer(file)
                write.writerows(np.stack(self.items[item], axis=0).tolist())

class Component():      ## EACH COMPONENT SHOULD HAVE ROTATE, REFLECT, MOVE, AND ADD (TO ITEM LIST) OPERATIONS

    def __init__(self):
        self.vertex_list = None
        self.items = []
    
    def make_vertex_list(self):
        return

    def rotate(self, theta):
        '''
        Rotates an object defined by the N_polygons x N_vertices x 2 array vertex_list about the origin by an angle theta.

        theta: angle in degrees
        vertex_list: corners defining polygon
        '''
        rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                                    [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        
        for i in range(self.vertex_list.shape[0]):
            self.vertex_list[i,:,:] = np.matmul(rotation_matrix, self.vertex_list[i,:,:,None])[:,:,0]
        return self

    def reflect_x(self):
        '''
        Reflects an object defined by the N_polygons x N_vertices x 2 array vertex_list over the x axis.
        '''
        reflection_matrix = np.array([[1, 0], [0, -1]])

        for i in range(self.vertex_list.shape[0]):
            self.vertex_list[i,:,:] = np.matmul(reflection_matrix, self.vertex_list[i,:,:,None])[:,:,0]
        return self

    def reflect_y(self):
        '''
        Reflects an object defined by the N_polygons x N_vertices x 2 array vertex_list over the y axis.
        '''
        reflection_matrix = np.array([[-1, 0], [0, 1]])

        for i in range(self.vertex_list.shape[0]):
            self.vertex_list[i,:,:] = np.matmul(reflection_matrix, self.vertex_list[i,:,:,None])[:,:,0]
        return self

    def move(self, x=0, y=0):
        '''
        Moves an object defined by the N_polygons x N_vertices x 2 array vertex_list by an amount (x, y)
        '''
        for i in range(self.vertex_list.shape[0]):
            self.vertex_list[i,:,:] = self.vertex_list[i,:,:] + np.array([x, y])[None,:]
        return self
    
    def add(self):
        self.items.append(self.vertex_list)
        return self

    def reset(self):
        self.make_vertex_list()
        return self

    def plot(self):
        plt.figure(figsize=(5,5))
        for i in range(self.vertex_list.shape[0]):
            plt.fill(self.vertex_list[i,:,0], self.vertex_list[i,:,1], facecolor='none', edgecolor='red')
        plt.show()
    
    def export_items(self, filepath, filename):
        filename = filepath + filename + '.csv'
        with open(filename, 'w') as file:
            write = csv.writer(file)
            write.writerows(np.stack(self.items, axis=0).tolist())

class Transition(Component):

    def __init__(self, L_track, L_taper, L_tip, w_gap, w_track, w_tip, w1, w2, name='transition'):
        self.params = locals()
        self.params.pop('self')
        self.items = []
        self.make_vertex_list()

    def make_vertex_list(self):
        '''
        Constructs ordered vertex list describing coax transition polygon, starting at top left at origin and moving clockwise
        See dimension labels in companion figure
        Creates (2 x N x 2) vertex array vertex_list
        First dimension specifies upper and lower polygons
        Uses self.params dictionary
        Required dictionary keys: L_track, L_taper, L_tip, w_gap, w_track, w1, w2 
        '''
        x_start = 0
        y_start = 0

        vertex_list_upper = np.array([x_start, y_start])[None,:]

        deltas = np.array([[self.params['L_track'], 0],
                           [self.params['L_taper'], self.params['w1']/2 - self.params['w_track']/2 - self.params['w_gap']],
                           [0, -(self.params['w1'] - self.params['w2'])/2],
                           [-self.params['L_taper'], -(self.params['w2'] - self.params['w_track'])/2],
                           [-(self.params['L_track'] - self.params['L_tip']), 0],
                           [-self.params['L_tip'], -(self.params['w_track'] - self.params['w_tip'])/2],
                           [0, (self.params['w_track'] - self.params['w_tip'])/2 + self.params['w_gap']]
                            ])

        for i in range(deltas.shape[0]):
            vertex_list_upper = np.append(vertex_list_upper, 
                                               vertex_list_upper[i,:][None,:] + deltas[i,:][None,:],
                                               axis = 0)
        
        vertex_list_lower = ( Board.reflect_x(vertex_list_upper) - 
                                    np.array([0, 2*self.params['w_gap'] + self.params['w_track']])[None,:]  )

        self.vertex_list = np.stack((vertex_list_upper, vertex_list_lower), axis=0)


class Termination(Component):

    def __init__(self, w_gap_top, L_track, L_taper, w_gap, w_track, w1, w2, name='termination'):
        self.params = locals()
        self.params.pop('self')
        self.items = []
        self.make_vertex_list()

    def make_vertex_list(self):
        '''
        Constructs ordered vertex list describing termination polygon, starting at top left at origin and moving clockwise
        See dimension labels in companion figure
        Creates vertex array vertex_list
        Uses self.params dictionary
        Required dictionary keys: w_gap_top, L_track, L_taper, w_gap, w_track, w1, w2
        '''
        x_start = 0
        y_start = 0

        self.vertex_list = np.array([x_start, y_start])[None,:]

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
            self.vertex_list = np.append(self.vertex_list, 
                                        self.vertex_list[i,:][None,:] + deltas[i,:][None,:],
                                        axis = 0)
        self.vertex_list = self.vertex_list[None,:,:]

class Choke(Component):

    def __init__(self, theta_filter, r_filter, choke_pad_diameter, name='choke', n_points_curve1=50, n_points_curve2=50, filter_scale=1.2):
        self.params = locals()
        self.params.pop('self')
        self.items = []
        self.make_vertex_list()

    def make_vertex_list(self):
        '''
        Makes corner array defining RF decoupling filters
        Starts at top left corner, places center of pad at origin
        Uses self.RF_choke_params dictionary
        Required dictionary keys: theta_filter, r_filter, choke_pad_diameter
        '''
        
        theta0 = np.radians((180 - self.params['theta_filter'])/2 + self.params['theta_filter'])
        theta_list1 = np.linspace(theta0, theta0 - np.radians(self.params['theta_filter']), self.params['n_points_curve1'])
        theta_list2 = np.linspace(theta0 - np.radians(self.params['theta_filter']), -np.pi - (np.pi - theta0), self.params['n_points_curve2'])

        vertex_list_inner = np.array([]).reshape(0, 2)
        for i in range(self.params['n_points_curve1']):
            vertex_list_inner = np.append(vertex_list_inner,
                                                        np.array([self.params['r_filter'] * np.cos(theta_list1[i]),
                                                            self.params['r_filter'] * np.sin(theta_list1[i])])[None,:], axis=0)

        for i in range(self.params['n_points_curve2']):
            vertex_list_inner = np.append(vertex_list_inner,
                                                np.array([self.params['choke_pad_diameter']/2 * np.cos(theta_list2[i]),
                                                            self.params['choke_pad_diameter']/2 * np.sin(theta_list2[i])])[None,:], axis=0)

        vertex_list_inner = np.append(vertex_list_inner,
                                                    np.array([self.params['r_filter'] * np.cos(theta_list1[0]),
                                                            self.params['r_filter'] * np.sin(theta_list1[0])])[None,:], axis=0)
        
        filter_scale = 1.2
        vertex_list_outer = filter_scale * vertex_list_inner
        filter_r_gap = filter_scale * self.params['r_filter'] - self.params['r_filter']
        vertex_list_outer = vertex_list_outer - np.array([0, filter_r_gap/2])[None,:]

        self.params['filter_r_gap'] = filter_r_gap
        self.vertex_list = np.stack((vertex_list_inner, vertex_list_outer), axis=0)

class cELC(Component):

    def __init__(self, wx_cELC, wy_cELC, wx_gap, wy_gap, w_center, w_gap, w_arm1, w_arm2, cELC_varactor_gap, w_cELC_varactor, name='cELC'):
        self.params = locals()
        self.params.pop('self')
        self.items = []
        self.make_vertex_list()

    def make_vertex_list(self):
        '''
        Constructs two ordered vertex lists describing cELC polygons, starting at top left and moving clockwise, origin at center.
        See dimension labels in companion figure
        Creates two vertex lists: outer_vertex_list and inner_vertex_list
        Uses self.params dictionary
        Required dictionary keys: wx_cELC, wy_cELC, wx_gap, wy_gap, w_center, w_gap, w_arm1, w_arm2, cELC_varactor_gap, w_cELC_varactor
        '''

        self.params['wx_tot'] = self.params['wx_cELC'] + 2*self.params['wx_gap']
        self.params['wy_tot'] = self.params['wy_cELC'] + 2*self.params['wy_gap']
        stub_length = (self.params['wy_gap'] - self.params['cELC_varactor_gap'])/2

        vertex_list_inner = np.array([[-self.params['wx_cELC']/2, -self.params['wy_cELC']/2],
                            [-self.params['w_cELC_varactor']/2, -self.params['wy_cELC']/2],
                            [-self.params['w_cELC_varactor']/2, -self.params['wy_cELC']/2 - stub_length],
                            [self.params['w_cELC_varactor']/2, -self.params['wy_cELC']/2 - stub_length],
                            [self.params['w_cELC_varactor']/2, -self.params['wy_cELC']/2],
                            [self.params['wx_cELC']/2, -self.params['wy_cELC']/2],
                            [self.params['wx_cELC']/2, -self.params['w_gap']/2],
                            [self.params['wx_cELC']/2-self.params['w_arm2'], -self.params['w_gap']/2],
                            [self.params['wx_cELC']/2-self.params['w_arm2'], -self.params['wy_cELC']/2+self.params['w_arm1']],
                            [self.params['w_center']/2, -self.params['wy_cELC']/2+self.params['w_arm1']],
                            [self.params['w_center']/2, self.params['wy_cELC']/2-self.params['w_arm1']],
                            [self.params['wx_cELC']/2-self.params['w_arm2'], self.params['wy_cELC']/2-self.params['w_arm1']],
                            [self.params['wx_cELC']/2-self.params['w_arm2'], self.params['w_gap']/2],
                            [self.params['wx_cELC']/2, self.params['w_gap']/2],
                            [self.params['wx_cELC']/2, self.params['wy_cELC']/2],
                            [-self.params['wx_cELC']/2, self.params['wy_cELC']/2],
                            [-self.params['wx_cELC']/2, self.params['w_gap']/2],
                            [-self.params['wx_cELC']/2+self.params['w_arm2'], self.params['w_gap']/2],
                            [-self.params['wx_cELC']/2+self.params['w_arm2'], self.params['wy_cELC']/2-self.params['w_arm1']],
                            [-self.params['w_center']/2, self.params['wy_cELC']/2-self.params['w_arm1']],
                            [-self.params['w_center']/2, -self.params['wy_cELC']/2+self.params['w_arm1']],
                            [-self.params['wx_cELC']/2+self.params['w_arm2'], -self.params['wy_cELC']/2+self.params['w_arm1']],
                            [-self.params['wx_cELC']/2+self.params['w_arm2'], -self.params['w_gap']/2],
                            [-self.params['wx_cELC']/2, -self.params['w_gap']/2],
                            [-self.params['wx_cELC']/2, -self.params['wy_cELC']/2]])

        vertex_list_outer = np.array([[-self.params['wx_tot']/2, -self.params['wy_tot']/2],
                            [-self.params['w_cELC_varactor']/2, -self.params['wy_tot']/2],
                            [-self.params['w_cELC_varactor']/2, -self.params['wy_tot']/2 + stub_length],
                            [self.params['w_cELC_varactor']/2, -self.params['wy_tot']/2 + stub_length],
                            [self.params['w_cELC_varactor']/2, -self.params['wy_tot']/2],
                            [self.params['wx_tot']/2, -self.params['wy_tot']/2],
                            [self.params['wx_tot']/2, self.params['wy_tot']/2],
                            [-self.params['wx_tot']/2, self.params['wy_tot']/2],
                            [-self.params['wx_tot']/2, -self.params['wy_tot']/2]])

        self.vertex_list = np.stack((vertex_list_inner, vertex_list_outer), axis=0)

        

