import glob

# Load the files
def get_file_names(path):
    ext = ['txt']    # Add formats here
    files = []
    [files.extend(glob.glob(path + '*.' + e)) for e in ext]
    return files

'''  
# Change to zero the first possition
def change_to_zero_1(file_name):
    with open(file=file_name) as f:
        lines = f.readlines()
        new_lines = ['']
        line_help = ''
        for line in lines:   
            # From 0 to 9 it the first number changes to 0
            for i in range(1): 
                first='9' #str(i)
                if line[0] == first and line[1] == ' ':
                    line_help = line
                    print(line)
                    final_line = line_help.replace(first,'85', 1)
                    print(final_line)
                    new_lines.append(final_line)      
        new_file = 'path/' + file_name # Same name
        g = open(new_file, "a")
        g.writelines(new_lines)

'''

# Change to zero the second position        
def change_to_zero_2(file_name):
    with open(file=file_name) as f:
        lines = f.readlines()
        new_lines = []
        line_help = ''   
    for line in lines:
        # From 0 to 9 it the second number changes to 0
        for j in range(1): 
            second='5' #str(j)
            for k in range(1):
                third='6' #str(k)
                if line[0] == second and line[1] == third:
                    line_help = line
                    print(line)
                    final_line = line_help.replace((second+third),'132', 1)
                    print(final_line)
                    new_lines.append(final_line) 
    new_file = 'path/' + file_name # Same name
    g = open(new_file, "a")
    g.writelines(new_lines)


def run():
    validation_path = ''
    paths = [validation_path]
    for path in paths:
        files = get_file_names(path=path)
        for file in files:
            #change_to_zero_1(file)
            change_to_zero_2(file)
run()



