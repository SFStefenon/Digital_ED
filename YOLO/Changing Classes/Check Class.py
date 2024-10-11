import glob

# Load the files
def get_file_names(path):
    ext = ['txt']    # Add formats here
    files = []
    [files.extend(glob.glob(path + '*.' + e)) for e in ext]
    return files
            
# Change to zero the second position        
def check_numbers(file_name):
    with open(file=file_name) as f:
        lines = f.readlines()
        new_lines = []
        line_help = ''   
    for line in lines:
        # From 0 to 9 it the second number changes to 0
        for j in range(9): 
            second=str(j)
            for k in range(9):
                third=str(k)
                if line[0] == second and line[1] == third:
                    print(file_name)
       
def run():
    validation_path = ''
    paths = [validation_path]
    for path in paths:
        files = get_file_names(path=path)
        for file in files:
            check_numbers(file)
run()



