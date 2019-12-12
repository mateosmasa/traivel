"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import itertools
import pandas as pd
import math
import json
import argparse
import recommedation_system 


def load_foot(foot,pois,col):
    pois = sorted(list(pois.Tripadvisor.unique()))
    foot = foot[foot[col].isin(pois)].loc[:,[col]+pois].sort_values(by=col,
                ascending=True).drop(columns=col)
    indexes = {key+1:value for key,value in enumerate(foot.columns)}
    shape = foot.shape
    return foot.values,indexes,shape
    

def transform_foot(foot,shape):
    z = np.zeros((1,shape[1]),dtype=float)
    foot = np.append(z,foot, axis=0)
    z = np.zeros((foot.shape[0],1),dtype=float)
    return np.append(z,foot, axis=1)

def search_lowest_distance(metro,foot):
    trans_type = (foot<=metro).astype(str)
    trans_type[trans_type == 'True'] = "Foot"
    trans_type[trans_type == 'False'] = "Metro"
    lowest = np.where(foot<=metro, foot, -1)
    lowest[np.where(lowest==-1)] = metro[np.where(lowest==-1)]
    return lowest.tolist(),trans_type


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))

def create_data_model(data_):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = data_
      
    data['tourist'] = 1
    data['depot'] = 0
    return data


def print_transport(index_list,trans_type):
    printed = ''
    for i in range(len(index_list)):
         printed += ' From {tup[0]} to {tup[1]} mode: {trans} '.format(trans = 
                            trans_type[index_list[i]], tup = index_list[i])
    if printed != '':
       print(printed)
    else:
       print("Not optimal route with this time travel")
       
def get_route(index_list_,indexes,max_route_distance):
    return json.dumps({'pois': (list(map(indexes.get, index_list_))), 
                'tiempo' : max_route_distance},ensure_ascii=False)
    
       
      
def print_solution(data, manager, routing, solution,trans_typ,indexes):
    """Prints solution on console."""
    max_route_distance = 0
    index = routing.Start(0)
    index_list = []
    plan_output = 'Route for tourist {}:\n'.format(0)
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} -> '.format(manager.IndexToNode(index))
        index_list.append(index)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, 0)
    index_list_ = [i for i in index_list if i !=0]
    index_list = pairwise(index_list_)
    index_list_t = [(indexes[i[0]] ,indexes[i[1]]) for i in index_list]
    print(index_list_t)
    print(list(map(indexes.get, index_list_)))
    plan_output += '{}\n'.format(manager.IndexToNode(index))
    plan_output += 'Time of the route: {}m\n'.format(route_distance)
    print(plan_output)
    max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route times: {}min'.format(max_route_distance)) 
    print_transport(index_list,trans_type)
    return get_route(index_list_,indexes,max_route_distance)

#df = df[df[["A","B"]].apply(tuple, 1).isin(AB_col)]

def run(data_,trans_type,time_travel,penalty,indexes):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(data_)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['tourist'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

  # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
    time_travel,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    
    
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

     
         
    # Solve the problem.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print('The Objective Value is {0}'.format(solution.ObjectiveValue()))
    # Print solution on console.
    if solution:
        return print_solution(data, manager, routing, solution,trans_type,indexes)
    else:
        print('Infeasible')


def main(data_,trans_type,time_travel,penalty,indexes,pois_json):
    opt_json = run(data_,trans_type,time_travel,penalty,indexes)
    print(opt_json)
    return opt_json,pois_json

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--timetravel", required=True, type=int,
	help="max time travel")
    ap.add_argument("-sp", "--speed", required=True, type=float,
	help="walking speed")
    ap.add_argument("-fm", "--footmatrix", required=True,
	help="path foot matrix")
    ap.add_argument("-mm", "--metromatrix", required=False, default="default",
	help="path foot matrix")

    ap.add_argument("-f", "--file", required=True,
    	help="path file pois")
    ap.add_argument("-s", "--selection", required=True,
    	help="type of search")
    ap.add_argument("-n", "--pois", required=True, type=int,
    	help="number of pois")
    ap.add_argument("-o", "--output", default="default", required=False,
    	help="output path")
    ap.add_argument("-d", "--desc", default="default", required=True,
    	help="description path")
    args = vars(ap.parse_args())
    
    pois = recommedation_system.main(args['file'],args['selection'],args['pois'],args['output'],args['desc'])
    
    pois_json = pois[['Tripadvisor','Latitud','Longitud','description','descripction_spanish']].to_json(orient='columns',force_ascii=False)

    time_travel = args['timetravel']
    foot = pd.read_csv(args['footmatrix'],sep="|")  
    foot,indexes,shape =load_foot(foot,pois,'Origen')
    foot = (transform_foot(foot,shape)/args['speed'])*60
    penalty = math.ceil(foot.max())*10
    if  args['metromatrix'] != 'default' :
        metro = pd.read_csv(args['metromatrix'],sep="|") 
        metro,indexes,shape =load_foot(metro,pois,'name_x')
        metro = transform_foot(metro,shape)
        data_,trans_type=search_lowest_distance(metro,foot)
    else:
        data_,trans_type= foot.tolist(),np.full(foot.shape,"Foot")
    main(data_,trans_type,time_travel,penalty,indexes,pois_json)
