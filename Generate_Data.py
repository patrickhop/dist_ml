import random
import sys

# recursive function that takes in an empty list, and returns one with iid data, with mean zero noise                                                                                                                                                                                                                                         
def make_data(mu, std, data, length):
    if length > 0:
        a_1 = random.uniform(0,10)
        a_2 = random.uniform(0,10)
        b = (2 * a_1 + random.gauss(mu, std)) + (4 * a_2 + random.gauss(mu, std)) # response has gaussian noise                                                                                                                                                                                                                               
        data.append([a_1, a_2, b]) # [datapoint, response]                                                                                                                                                                                                                                                                                    
        return make_data(mu, std, data, length - 1)
    else: return data

print "generating data..."

data =  make_data(0, .25, [], 250)
data_file = open(sys.argv[1], "w")

for result in data:
    for value in result:
        data_file.write(str(value) + " ")
    data_file.write("\n")

print "data generated."
