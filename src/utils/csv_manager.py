import os
import csv
def append_to_csv(path_name,value_list : list,create=False,row_name =None):
    if create:

        dir_path = "/".join(path_name.split("/")[:-1])
        os.makedirs(dir_path, exist_ok=True)


    with open(path_name,"a") as f :
        row = ",".join(str(value) for value in value_list)
        if row_name is not  None:
            f.write(row_name + "," + row +"\n")
        else:
            f.write(row + "\n")
            
def list_from_csv(file):
    with open(file, "r") as f:
        reader = csv.reader(f)
        return list(reader)