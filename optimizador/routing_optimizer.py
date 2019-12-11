"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import itertools


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


def print_solution(data, manager, routing, solution,trans_type):
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
    index_list = [i for i in index_list if i !=0]
    index_list = pairwise(index_list)
    plan_output += '{}\n'.format(manager.IndexToNode(index))
    plan_output += 'Distance of the route: {}m\n'.format(route_distance)
    print(plan_output)
    max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))
    print_transport(index_list,trans_type)




def main(data_,trans_type,time_travel,penalty):
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
        print_solution(data, manager, routing, solution,trans_type)
    else:
        print('Infeasible')


if __name__ == '__main__':
    
    time_travel = 500
    penalty = 1000
    
    foot = np.array([[0,0,0,0],
     [0,0,200,400],
     [0,200, 0,400],
     [0,400,400,0]])
    metro = np.array([[0,0,0,0],
     [0,0,100,500],
     [0,100, 0,400],
     [0,500,400,0]])

    data_,trans_type=search_lowest_distance(metro,foot)
    main(data_,trans_type,time_travel,penalty)