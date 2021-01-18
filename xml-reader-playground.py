################################## Testing ##################
import os
from xml.etree.ElementTree import ElementTree

file_name = 'DATA.xml'
full_file = os.path.join('data', file_name)

tree = ElementTree()

tree.parse(full_file)


firstTrace = tree.find('creation')
print(firstTrace)


# for trace in traces:
#     print(trace)